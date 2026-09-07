"""
AmbientField.py -- an exogenous non-stationarity that is RECOVERABLE by
construction, unlike the opponent.

WHY NOT THE OPPONENT
--------------------
Measured on this task, the 'hidden' opponent preset removes transmission
capacity for 51.6% of grid steps.  A do-nothing policy -- which uses no
information at all, and therefore cannot be suffering an information loss --
lost 53% of its survival.  That drop is pure physics: no policy, omniscient or
otherwise, routes power through an open line.  So the opponent stacks two gaps:

    gap A  (information)  the loss from not OBSERVING the disturbance  -- closable
    gap B  (feasibility)  the loss from the grid being WEAKER          -- closable by nobody

With gap B large, no algorithm can reach the undisturbed baseline, and any
result reads as "we made the task harder" rather than "non-stationarity breaks
MARL".

THE FIELD
---------
Line thermal ratings depend on ambient conditions: wind cools conductors, still
heat derates them.  This is Dynamic Line Rating, a deployed technology, and the
+-20-40% swing against static ratings is well documented.  Weather systems cross
a region as approximately plane fronts.  So:

    limit_l(t) = nominal_l * ( 1 + A cos( 2 pi ( k.x_l - c t ) / L + phi ) )

with x_l the line's geographic midpoint from the grid layout.  The cosine is
mean-zero over both space and time, so AVERAGE CAPACITY IS UNCHANGED and gap B
collapses to a small Jensen term (rho is convex in 1/limit) that the oracle arm
measures rather than assumes.  Nothing is ever disconnected, so the coupling
structure of the grid stays valid throughout.

WHY IT BREAKS EXISTING ALGORITHMS -- provably, not empirically
-------------------------------------------------------------
At ONE location the observed series is a sinusoid of temporal frequency c/L.
From it you recover A and c/L and NOTHING ELSE: c and L are not separable, and
the heading k is completely unidentifiable -- a 1-D scalar series at a point
cannot determine a 2-D velocity.  Two separated observers give the phase
difference 2 pi k.(x1-x2)/L, one equation in two unknowns.  THREE non-collinear
observers determine k and L uniquely.

  memoryless MAPPO/MASAC/QMIX : no memory, cannot even estimate c/L
  recurrent MAPPO, local obs  : tracks the field HERE, but the heading is
                                unidentifiable from one location -- so it can
                                react and never anticipate
  CTDE / centralised critic   : centralisation is training-only; execution is
                                still local

The heading is what says WHICH ZONE IS HIT NEXT.  Storage must charge before the
derating arrives, curtailment ramps take time, and the rho>safe_max_rho gate only
wakes agents once it is already late -- so anticipation is worth real return,
and anticipation is exactly what is not locally identifiable.

THE ORACLE ARM
--------------
oracle='full' appends the true latent to every agent's observation.  It is the
ceiling: whatever it reaches is what perfect inference buys.

    gap B = baseline(field off)  -  oracle(field on)   <- irreducible
    gap A = oracle(field on)     -  blind(field on)    <- what a method can win

If gap B is not small, turn `amplitude` down until it is.  Measure it before
building any method on top.
"""

import functools

import numpy as np
from gymnasium.spaces import Box

from .PZMAEnvWithHeuristics import PZMAEnvRecoDNLimit


class _PlaneWaveEstimator:
    """Recover the advecting field from line-rating measurements.

    Each line's CURRENT rating is directly measurable without any privileged
    access: rho = a_or / limit, so limit = a_or / rho and the field factor is
    limit / nominal. The hard part is not the present, it is the FUTURE -- which
    needs the wavevector, and that is what may not be locally identifiable.

    Model, at a fixed time t:   f(x) - 1 = A cos(u.x + psi_t)
    which is LINEAR in (a, b) for f - 1 = a cos(u.x) + b sin(u.x) once u is
    fixed. So: grid over candidate wavevectors u, precompute the pseudo-inverse
    of each candidate's design matrix, and per step the fit is one matvec per
    candidate. Amplitude A = hypot(a, b), phase psi_t = atan2(-b, a).

    The temporal frequency comes from tracking psi_t: psi advances by -omega per
    step, so omega is the unwrapped phase rate. Prediction is then
        f(x, t+h) = 1 + A cos(u.x + psi_t - omega h).

    HONEST CAVEAT on the identifiability claim: a single POINT cannot determine
    a 2-D wavevector -- that part is exact. But a zone owns ~20-46 lines spread
    over a fair fraction of a wavelength, so a zone-local fit is ill-conditioned
    rather than impossible. The separation between `local` and `consensus` is
    therefore a CONDITIONING argument, not an impossibility proof. Run both and
    report the difference instead of asserting one.
    """

    def __init__(self, positions, diag, n_head=24, n_wave=7, warmup=8):
        self.pos = np.asarray(positions, dtype=float)
        self.diag = float(diag)
        self.warmup = int(warmup)

        headings = np.linspace(0.0, np.pi, n_head, endpoint=False)
        waves = np.geomspace(0.3 * diag, 3.0 * diag, n_wave)
        self.cands, self._pinv, self._design = [], [], []
        for L in waves:
            for th in headings:
                u = np.array([np.cos(th), np.sin(th)]) * (2.0 * np.pi / L)
                proj = self.pos @ u
                X = np.stack([np.cos(proj), np.sin(proj)], axis=1)   # (n, 2)
                G = X.T @ X + 1e-9 * np.eye(2)
                self.cands.append((u, L, th))
                self._design.append(X)
                self._pinv.append(np.linalg.solve(G, X.T))           # (2, n)

        self.n_seen = 0
        self.A = 0.0
        self.psi = 0.0
        self.omega = 0.0
        self.best = 0
        self._psi_hist = []

    def update(self, f_meas):
        """f_meas: measured rating factor per position. Returns residual RMS."""
        y = np.asarray(f_meas, dtype=float) - 1.0
        if not np.all(np.isfinite(y)) or y.size != self.pos.shape[0]:
            return float("nan")

        best_r, best_i, best_ab = np.inf, 0, (0.0, 0.0)
        for i, P in enumerate(self._pinv):
            a, b = P @ y
            r = float(np.mean((self._design[i] @ np.array([a, b]) - y) ** 2))
            if r < best_r:
                best_r, best_i, best_ab = r, i, (a, b)

        a, b = best_ab
        self.best = best_i
        self.A = float(np.hypot(a, b))
        psi = float(np.arctan2(-b, a))

        self._psi_hist.append(psi)
        if len(self._psi_hist) > 12:
            self._psi_hist.pop(0)
        if len(self._psi_hist) >= 3:
            d = np.diff(np.unwrap(np.asarray(self._psi_hist)))
            self.omega = float(-np.median(d))          # psi advances by -omega
        self.psi = psi
        self.n_seen += 1
        return float(np.sqrt(best_r))

    def ready(self):
        return self.n_seen >= self.warmup

    def predict(self, positions, h):
        """Rating factor at `positions`, h steps ahead."""
        u = self.cands[self.best][0]
        proj = np.atleast_2d(positions) @ u
        return 1.0 + self.A * np.cos(proj + self.psi - self.omega * h)

    def heading_deg(self):
        return float(np.degrees(self.cands[self.best][2])) % 180.0


DEFAULT_FIELD = dict(
    amplitude=0.25,        # +-25% rating swing; DLR literature says 20-40%
    wavelength_frac=1.0,   # wavelength as a fraction of the layout diagonal
    speed_frac=1.0 / 48.0, # layout diagonals per step -> 48 steps = 4 h period
    randomize=True,        # redraw heading and phase every episode
    jitter=0.0,            # relative jitter on amplitude/wavelength/speed
    seed=0,
    oracle="none",         # 'none' | 'local' | 'full'
    horizons=(0, 12, 24),  # steps ahead exposed by the oracle (0, 1 h, 2 h)
    # -- anticipatory controller (PACT architecture: host RL untouched) --
    controller="none",     # 'none' | 'oracle' | 'consensus' | 'local'
    ctrl_gain=0.5,         # correction per unit of anticipated rho excess
    ctrl_horizon=12,       # steps ahead to anticipate (12 = 1 h)
    ctrl_target=0.90,      # rho above which anticipated loading is "excess"
    ctrl_max_delta=0.5,    # cap on |curtailment correction|
    ctrl_report_every=5000,
)

PRESETS = {
    "off":      None,
    "mild":     dict(amplitude=0.12),
    "standard": dict(amplitude=0.25),
    "strong":   dict(amplitude=0.40),
    # Faster front: the same amplitude but less time to react.
    "fast":     dict(amplitude=0.25, speed_frac=1.0 / 24.0),
}


def get_field_preset(name, oracle="none", controller="none", **overrides):
    if name is None or name == "off":
        if (controller or "none") != "none":
            raise ValueError("--field_controller needs a field; --field is 'off'")
        return None
    if name not in PRESETS:
        raise ValueError(f"unknown field preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = dict(DEFAULT_FIELD)
    cfg.update(PRESETS[name] or {})
    cfg["oracle"] = oracle
    cfg["controller"] = controller or "none"
    cfg.update(overrides)
    return cfg


def line_positions(env_g2op):
    """Geographic midpoint of every line, from the grid's own layout.

    A missing layout is FATAL rather than silently replaced by a proxy: a
    geometric stand-in for a declared operator is exactly the substitution that
    produced a negative fit gain in the PACT work.
    """
    layout = getattr(env_g2op, "grid_layout", None)
    if not layout:
        raise RuntimeError(
            "env.grid_layout is empty -- AmbientField needs real substation "
            "coordinates to define a spatial front. Refusing to fabricate one.")

    names = list(env_g2op.name_sub)
    missing = [n for n in names if n not in layout]
    if missing:
        raise RuntimeError(f"grid_layout is missing {len(missing)} substations, "
                           f"e.g. {missing[:5]}")

    sub = np.array([layout[n] for n in names], dtype=float)   # (n_sub, 2)
    mid = 0.5 * (sub[env_g2op.line_or_to_subid] + sub[env_g2op.line_ex_to_subid])

    span = sub.max(0) - sub.min(0)
    diag = float(np.hypot(*span))
    if not np.isfinite(diag) or diag <= 0:
        raise RuntimeError(f"degenerate grid layout (diagonal={diag})")
    return mid, sub, diag


class AmbientFieldEnv(PZMAEnvRecoDNLimit):
    def __init__(self, ambient_field=None, **kwargs):
        super().__init__(**kwargs)

        cfg = dict(DEFAULT_FIELD)
        cfg.update(ambient_field or {})
        self._fcfg = cfg
        self._oracle = cfg.get("oracle", "none") or "none"
        self._horizons = tuple(cfg.get("horizons", (0,)))

        self._nominal = np.asarray(
            self.env_g2op.get_thermal_limit(), dtype=float).copy()
        self._linepos, _subpos, self._diag = line_positions(self.env_g2op)

        # Zone centroids: where each agent "stands" in the field.
        self._agent_pos = {}
        for i, zname in enumerate(self.zone_names, start=1):
            idx = [int(x) for x in self.zones_dict[zname].get("line_in_zone_idx", [])]
            self._agent_pos[f"agent_{i}"] = (self._linepos[idx].mean(0) if idx
                                             else self._linepos.mean(0))
        if self.use_redispatching_agent:
            self._agent_pos["redispatching_agent"] = self._linepos.mean(0)

        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self._t = 0
        self._z = None
        self._draw_latent()

        n_h = len(self._horizons)
        self._n_oracle = (0 if self._oracle == "none"
                          else n_h if self._oracle == "local"
                          else n_h + 7)
        self._obs_space_cache = {}

        self._wrap_env()
        self._apply_field()
        self._init_controller()

    # -- the latent -----------------------------------------------------
    def _draw_latent(self):
        c = self._fcfg
        rng = self._rng
        j = float(c.get("jitter", 0.0))

        def jit(v):
            return v * (1.0 + j * rng.uniform(-1, 1)) if j > 0 else v

        if c.get("randomize", True):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            phi = rng.uniform(0.0, 2.0 * np.pi)
        else:
            theta, phi = 0.0, 0.0

        self._z = dict(
            A=jit(float(c["amplitude"])),
            L=jit(float(c["wavelength_frac"])) * self._diag,
            c=jit(float(c["speed_frac"])) * self._diag,
            khat=np.array([np.cos(theta), np.sin(theta)]),
            phi=phi,
            theta=theta,
        )

    def _factor_at(self, pos, t):
        """Rating multiplier at position(s) `pos` and time `t`."""
        z = self._z
        proj = np.atleast_2d(pos) @ z["khat"]
        val = z["A"] * np.cos(2.0 * np.pi * (proj - z["c"] * t) / z["L"] + z["phi"])
        return 1.0 + val

    def _apply_field(self):
        f = self._factor_at(self._linepos, self._t)
        # Guard: a non-positive rating would make rho infinite.
        self.env_g2op.set_thermal_limit(self._nominal * np.clip(f, 0.05, None))

    # -- hook every grid transition, heuristic steps included ------------
    def _wrap_env(self):
        inner = self.env_g2op
        orig_step, orig_reset = inner.step, inner.reset

        def stepping(action):
            self._t += 1
            self._apply_field()          # in force FOR this step
            return orig_step(action)

        def resetting(*a, **k):
            self._t = 0
            if self._fcfg.get("randomize", True):
                self._draw_latent()
            out = orig_reset(*a, **k)
            self._apply_field()
            return out

        inner.step = stepping
        inner.reset = resetting

    # -- oracle observation ---------------------------------------------
    def _oracle_feats(self, agent_id):
        if self._oracle == "none":
            return None
        pos = self._agent_pos[agent_id]
        vals = [float(self._factor_at(pos, self._t + h)[0] - 1.0)
                for h in self._horizons]
        if self._oracle == "full":
            z = self._z
            vals += [z["A"],
                     z["L"] / self._diag,
                     z["c"] / self._diag,
                     float(z["khat"][0]), float(z["khat"][1]),
                     float(np.sin(z["phi"])), float(np.cos(z["phi"]))]
        return np.asarray(vals, dtype=np.float32)

    def _to_gym_obs(self, grid2op_obs):
        obs = super()._to_gym_obs(grid2op_obs)
        if self._oracle == "none":
            return obs
        for aid in list(obs.keys()):
            extra = self._oracle_feats(aid)
            obs[aid] = np.concatenate(
                [np.asarray(obs[aid], dtype=np.float32), extra]).astype(np.float32)
        return obs

    def observation_space(self, agent_id):
        if self._oracle == "none":
            return super().observation_space(agent_id)
        if agent_id not in self._obs_space_cache:
            base = super().observation_space(agent_id)
            k = self._n_oracle
            self._obs_space_cache[agent_id] = Box(
                low=np.concatenate([base.low, np.full(k, -5.0, dtype=base.dtype)]),
                high=np.concatenate([base.high, np.full(k, 5.0, dtype=base.dtype)]),
                dtype=base.dtype)
        return self._obs_space_cache[agent_id]

    # -- anticipatory controller ----------------------------------------
    def _init_controller(self):
        c = self._fcfg
        self._ctrl = (c.get("controller") or "none").lower()
        self._ctrl_gain = float(c.get("ctrl_gain", 0.5))
        self._ctrl_h = int(c.get("ctrl_horizon", 12))
        self._ctrl_target = float(c.get("ctrl_target", 0.90))
        self._ctrl_max = float(c.get("ctrl_max_delta", 0.5))
        self._ctrl_report = int(c.get("ctrl_report_every", 5000))

        # Which action entries are curtailment: curtail_zone has low 0, storage
        # and redispatch have low < 0. Derived from the spaces, so it stays
        # correct if the zone partition changes.
        self._curt_mask = {}
        for aid in self.agents:
            low = np.asarray(self.action_space(aid).low)
            self._curt_mask[aid] = (low >= 0.0)

        # Lines each agent can sense / is judged on.
        self._agent_lines = {}
        for i, zname in enumerate(self.zone_names, start=1):
            idx = [int(x) for x in self.zones_dict[zname].get("line_large_idx", [])]
            self._agent_lines[f"agent_{i}"] = np.asarray(idx, dtype=int)
        if self.use_redispatching_agent:
            self._agent_lines["redispatching_agent"] = np.arange(self.env_g2op.n_line)

        self._est = None
        self._est_local = {}
        if self._ctrl == "consensus":
            self._est = _PlaneWaveEstimator(self._linepos, self._diag)
        elif self._ctrl == "local":
            for aid, idx in self._agent_lines.items():
                if idx.size >= 6:
                    self._est_local[aid] = _PlaneWaveEstimator(
                        self._linepos[idx], self._diag)

        self._ctrl_stats = dict(n=0, n_nonzero=0, dabs=0.0, resid=0.0,
                                mae=0.0, n_mae=0)

    def _measure_field(self, g2op_obs):
        """Field factor per line from the observation alone: limit = a_or / rho."""
        rho = np.asarray(g2op_obs.rho, dtype=float)
        a_or = np.abs(np.asarray(g2op_obs.a_or, dtype=float))
        ok = (rho > 1e-6) & (a_or > 1e-6) & np.isfinite(rho) & np.isfinite(a_or)
        f = np.ones(self.env_g2op.n_line, dtype=float)
        f[ok] = (a_or[ok] / rho[ok]) / self._nominal[ok]
        return f, ok

    def _predicted_factor(self, agent_id, lines, h):
        """Rating factor h steps ahead for `lines`, per the active driver."""
        if self._ctrl == "oracle":
            return self._factor_at(self._linepos[lines], self._t + h)
        if self._ctrl == "consensus":
            if self._est is not None and self._est.ready():
                return self._est.predict(self._linepos[lines], h)
            return None
        if self._ctrl == "local":
            e = self._est_local.get(agent_id)
            if e is not None and e.ready():
                return e.predict(self._linepos[lines], h)
            return None
        return None

    def _controller_delta(self, gym_act_dict):
        """Anticipatory curtailment correction. Floor property: a zero excess
        (or an unready estimator) returns the action untouched, byte for byte."""
        obs = getattr(self, "_previous_act", None)
        if obs is None:
            return gym_act_dict

        f_now, ok = self._measure_field(obs)
        if self._ctrl == "consensus" and self._est is not None:
            m = ok.copy()
            r = self._est.update(np.where(m, f_now, 1.0))
            self._ctrl_stats["resid"] += 0.0 if r != r else r
        elif self._ctrl == "local":
            for aid, e in self._est_local.items():
                idx = self._agent_lines[aid]
                e.update(np.where(ok[idx], f_now[idx], 1.0))

        # Estimator accuracy against the truth, for reporting only.
        if self._ctrl in ("consensus", "local"):
            pred = self._predicted_factor(
                "agent_1" if self._ctrl == "local" else None,
                np.arange(self.env_g2op.n_line), self._ctrl_h) \
                if self._ctrl == "consensus" else None
            if pred is not None:
                truth = self._factor_at(self._linepos, self._t + self._ctrl_h)
                self._ctrl_stats["mae"] += float(np.mean(np.abs(pred - truth)))
                self._ctrl_stats["n_mae"] += 1

        rho = np.asarray(obs.rho, dtype=float)
        out = dict(gym_act_dict)
        for aid, act in gym_act_dict.items():
            lines = self._agent_lines.get(aid)
            mask = self._curt_mask.get(aid)
            if lines is None or mask is None or not mask.any() or lines.size == 0:
                continue
            f_h = self._predicted_factor(aid, lines, self._ctrl_h)
            if f_h is None:
                continue
            f_c = np.where(ok[lines], f_now[lines], 1.0)
            # If flow held constant, rho scales as limit_now / limit_future.
            rho_pred = rho[lines] * np.clip(f_c, 1e-3, None) / np.clip(f_h, 1e-3, None)
            excess = float(np.max(rho_pred) - self._ctrl_target)   # MAX: binding line
            self._ctrl_stats["n"] += 1
            if excess <= 0.0:
                continue                                            # exactly blind
            d = float(np.clip(self._ctrl_gain * excess, 0.0, self._ctrl_max))
            a = np.array(act, dtype=np.float32, copy=True)
            a[mask] = np.clip(a[mask] - d, 0.0, 1.0)   # lower curtail = curtail more
            out[aid] = a
            self._ctrl_stats["n_nonzero"] += 1
            self._ctrl_stats["dabs"] += d

        s = self._ctrl_stats
        if self._ctrl_report and s["n"] and s["n"] % self._ctrl_report < len(self.agents):
            frac = s["n_nonzero"] / max(1, s["n"])
            print(f"[field-ctrl pid{os.getpid()}] driver={self._ctrl} "
                  f"steps={s['n']} delta_nonzero_frac={frac:.3f} "
                  f"delta_abs={s['dabs']/max(1,s['n_nonzero']):.4f} "
                  f"pred_mae={(s['mae']/s['n_mae']) if s['n_mae'] else float('nan'):.4f}")
        return out

    def _from_gym_act(self, gym_act_dict):
        if getattr(self, "_ctrl", "none") != "none":
            gym_act_dict = self._controller_delta(gym_act_dict)
        return super()._from_gym_act(gym_act_dict)

    def controller_stats(self):
        s = dict(self._ctrl_stats)
        s["delta_nonzero_frac"] = s["n_nonzero"] / max(1, s["n"])
        s["delta_abs"] = s["dabs"] / max(1, s["n_nonzero"])
        s["pred_mae"] = s["mae"] / s["n_mae"] if s["n_mae"] else float("nan")
        return s

    # -- reporting -------------------------------------------------------
    def field_summary(self):
        z = self._z
        return dict(amplitude=z["A"],
                    wavelength_frac=z["L"] / self._diag,
                    speed_frac=z["c"] / self._diag,
                    period_steps=z["L"] / z["c"] if z["c"] else float("inf"),
                    heading_deg=float(np.degrees(z["theta"])) % 360.0,
                    oracle=self._oracle,
                    n_oracle_feats=self._n_oracle,
                    controller=getattr(self, "_ctrl", "none"),
                    ctrl_gain=getattr(self, "_ctrl_gain", None),
                    ctrl_horizon=getattr(self, "_ctrl_h", None),
                    layout_diag=self._diag)


# ---------------------------------------------------------------------------
def self_test(env_name=None, steps=6):
    """Verify set_thermal_limit actually bites mid-episode.

    The whole design rests on it, so check rather than assume: after setting a
    limit L, grid2op must report rho == a_or / L.
    """
    import os

    import grid2op
    from grid2op.Action import PlayableAction
    try:
        from lightsim2grid import LightSimBackend as backend_cls
    except ImportError:
        from grid2op.Backend import PandaPowerBackend as backend_cls

    if env_name is None:
        env_name = os.path.join(grid2op.get_current_local_dir(), "l2rpn_idf_2023")

    env = grid2op.make(env_name, action_class=PlayableAction, backend=backend_cls())
    obs = env.reset()
    nominal = np.asarray(env.get_thermal_limit(), dtype=float).copy()
    print(f"env            : {env_name}")
    print(f"n_line         : {env.n_line}")
    print(f"nominal limits : min={nominal.min():.1f} max={nominal.max():.1f}")

    mid, sub, diag = line_positions(env)
    print(f"grid_layout    : {len(sub)} substations, diagonal={diag:.1f}")
    print(f"line midpoints : x[{mid[:,0].min():.0f},{mid[:,0].max():.0f}] "
          f"y[{mid[:,1].min():.0f},{mid[:,1].max():.0f}]")

    scaled = nominal * 0.5
    env.set_thermal_limit(scaled)
    obs, _, done, _ = env.step(env.action_space({}))
    got = np.asarray(env.get_thermal_limit(), dtype=float)
    ok_set = np.allclose(got, scaled, rtol=1e-5)
    live = obs.a_or > 1e-6
    ratio = obs.rho[live] / (obs.a_or[live] / scaled[live])
    ok_rho = bool(np.allclose(ratio, 1.0, rtol=1e-3))

    print(f"\nset_thermal_limit persisted : {ok_set}")
    print(f"rho == a_or / new_limit     : {ok_rho}   "
          f"(ratio min={ratio.min():.4f} max={ratio.max():.4f} over "
          f"{live.sum()} energised lines)")

    env.set_thermal_limit(nominal)
    rhos = []
    for _ in range(steps):
        f = 1.0 + 0.25 * np.cos(np.linspace(0, 2 * np.pi, env.n_line))
        env.set_thermal_limit(nominal * f)
        obs, _, done, _ = env.step(env.action_space({}))
        rhos.append(float(obs.rho.max()))
        if done:
            obs = env.reset()
    print(f"max rho over {steps} modulated steps: "
          f"{['%.3f' % r for r in rhos]}")
    env.close()

    verdict = ok_set and ok_rho
    print("\n" + ("PASS -- thermal limits are live mid-episode; the design works"
                  if verdict else
                  "FAIL -- set_thermal_limit does not take effect; the field NS "
                  "cannot be built this way"))
    return verdict


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ambient field NS: verify and inspect.")
    ap.add_argument("--self-test", action="store_true",
                    help="check that set_thermal_limit bites mid-episode")
    ap.add_argument("--env", default=None)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test(a.env) else 1)
    for name in sorted(PRESETS):
        print(f"{name:<10} {get_field_preset(name)}")
