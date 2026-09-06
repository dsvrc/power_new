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


DEFAULT_FIELD = dict(
    amplitude=0.25,        # +-25% rating swing; DLR literature says 20-40%
    wavelength_frac=1.0,   # wavelength as a fraction of the layout diagonal
    speed_frac=1.0 / 48.0, # layout diagonals per step -> 48 steps = 4 h period
    randomize=True,        # redraw heading and phase every episode
    jitter=0.0,            # relative jitter on amplitude/wavelength/speed
    seed=0,
    oracle="none",         # 'none' | 'local' | 'full'
    horizons=(0, 12, 24),  # steps ahead exposed by the oracle (0, 1 h, 2 h)
)

PRESETS = {
    "off":      None,
    "mild":     dict(amplitude=0.12),
    "standard": dict(amplitude=0.25),
    "strong":   dict(amplitude=0.40),
    # Faster front: the same amplitude but less time to react.
    "fast":     dict(amplitude=0.25, speed_frac=1.0 / 24.0),
}


def get_field_preset(name, oracle="none", **overrides):
    if name is None or name == "off":
        return None
    if name not in PRESETS:
        raise ValueError(f"unknown field preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = dict(DEFAULT_FIELD)
    cfg.update(PRESETS[name] or {})
    cfg["oracle"] = oracle
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
