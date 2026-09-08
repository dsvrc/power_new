"""
Reconfig.py -- exogenous substation reconfiguration: a non-stationarity that is
recoverable BY CONSTRUCTION, because every configuration it uses was screened
for benignness before being allowed in.

WHY THIS AND NOT THE FIELD
--------------------------
The DLR field (AmbientField.py) failed its own ceiling test: an oracle handed the
true field measured +5% return and +1% survival over blind. It attacked
rho = flow/limit by shrinking `limit`, so it pushed the system against a HARD
cliff, and curtailment plus a few storage units cannot offset a 25% capacity loss
on a binding corridor. Perfect information does not help when you cannot act far
enough. gap_B was ~everything.

A busbar split redistributes flows WITHOUT removing capacity. Measured by
screen_topology.py over 90 candidates:

    22 configurations across 10 substations move the sensitivities 2.4-20.8%
    while costing a median 1.5% of do-nothing survival; ten of them cost ZERO.

Do-nothing uses no information at all, so any survival it keeps is capacity that
is genuinely still there. Screening on it is what makes gap_B small by
construction rather than by hope -- and the screen is the evidence.

WHY EXISTING METHODS SHOULD FAIL
--------------------------------
A reconfiguration changes the SENSITIVITY of every line flow to every agent's
injection. It is not a feature you can read off a frame: obs_attr_to_keep_default
carries no topo_vect, so it surfaces only as "my actions stopped having the
effect they used to".

  memoryless MAPPO/MASAC/QMIX : a sensitivity cannot be inferred from one frame
  recurrent, local obs        : can identify only the columns it EXCITES. Agent i
                                varies its own injections so it can learn
                                d(flow)/d(a_i). It never controls a_j, so
                                d(flow)/d(a_j) is unreachable at any sample size
                                or SNR -- a STRUCTURAL limit, not a conditioning
                                one. (The plane-wave field failed exactly here:
                                a zone's own lines spanned enough of a wavelength
                                that a local estimator matched the pooled one,
                                measured ratio 1.11, and the coordination story
                                collapsed.)
  CTDE critic                 : centralisation is training-time only
  PACT with a fixed W         : W is declared once and never fitted (spec 2.1),
                                so a reconfiguration silently invalidates the very
                                operator the channel inverse divides by

SEVERITY DIAL
-------------
Tiers filter the screened set by cost and potency. Turning severity up changes
HOW OFTEN and HOW MANY reconfigurations fire -- it never relaxes the benignness
screen, so it cannot quietly reintroduce the field's failure.
"""

import json
import os

import numpy as np
from gymnasium.spaces import Box

from .PZMAEnvWithHeuristics import PZMAEnvRecoDNLimit


DEFAULT_SCREEN = "topology_screen.json"

DEFAULT_SIGNFLIP = "signflip_screen.json"

# tier -> (max survival drop, min sensitivity change)
#
# These three select on FLOW REDISTRIBUTION MAGNITUDE, which authority.py showed
# is correlated with constraint displacement Delta -- i.e. they select for the
# very thing that destroys recoverability. Measured: the `standard` tier gave
# Delta = 0.106 against an actuator reach of 0.03, so kappa = 0.02, and its
# oracle arm finished 8% BELOW blind at 1M frames. Kept for the ablation that
# reproduces that failure; do not build on them.
TIERS = {
    "conservative": (0.00, 0.02),   # measured: 10 configs, zero survival cost
    "standard":     (0.02, 0.05),   # measured: 8 configs, kappa 0.02
    "aggressive":   (0.10, 0.02),   # measured: 22 configs
}

# The tier that is recoverable BY CONSTRUCTION: small Delta (the grid is no
# harder) plus flipped control-authority signs (your lever works backwards).
# An oracle knowing the current signs matches no-NS, so gap_B = 0 definitionally
# rather than by screening luck. Built by screen_signflip.py.
SIGNFLIP_TIER = "signflip"

DEFAULT_RECONFIG = dict(
    tier="standard",
    screen_path=None,        # defaults to topology_screen.json, cwd then module dir
    interval_steps=100,      # mean steps between reconfiguration events
    interval_jitter=0.5,     # relative uniform jitter on the interval
    n_active=2,              # how many substations may be off-nominal at once
    revert_prob=0.3,         # chance an event restores a substation to nominal
    oracle="none",           # 'none' | 'full' (true bus vectors in the obs)
    seed=0,
    report_every=5000,
)


def _find_screen(path):
    here = os.path.dirname(os.path.abspath(globals().get("__file__", ".")))
    cands = [path] if path else []
    cands += [DEFAULT_SCREEN,
              os.path.join(os.getcwd(), DEFAULT_SCREEN),
              os.path.join(here, DEFAULT_SCREEN)]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise RuntimeError(
        "No topology_screen.json found. Run screen_topology.py first -- the "
        "screened set is what makes this NS recoverable, and using unscreened "
        "reconfigurations would reproduce the DLR field's failure. Looked in: "
        + ", ".join(str(c) for c in cands if c))


def load_signflip(screen_path=None, verbose=True):
    """The sign-flip set: Delta ~ 0 and control-authority signs reversed."""
    p = _find_screen(screen_path or DEFAULT_SIGNFLIP)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    keep = d.get("screened") or []
    if not keep:
        raise RuntimeError(
            f"{p} contains no passing configuration. Run screen_signflip.py and "
            f"read its two per-criterion counts -- if none has small |Delta|, "
            f"this task cannot host a recoverable NS by reconfiguration and the "
            f"route is raising actuator authority instead.")
    out = [dict(sub_id=int(c["sub_id"]),
                config=np.asarray(c["config"], dtype=int),
                drop=float(c.get("survival_drop") or 0.0),
                sens=float(c.get("flip_rate", 0.0)),
                delta_rho=float(c.get("delta_rho", float("nan")))) for c in keep]
    if verbose:
        subs = sorted({k["sub_id"] for k in out})
        print(f"[reconfig] tier=signflip -> {len(out)} configs over {len(subs)} "
              f"substations {subs}")
        print(f"[reconfig]   |Delta|   median "
              f"{np.median([k['delta_rho'] for k in out]):.4f} rho-units "
              f"(actuator reach ~0.03 -> kappa >= 1)")
        print(f"[reconfig]   flip rate median "
              f"{100*np.median([k['sens'] for k in out]):.0f}% of agents")
        print(f"[reconfig]   screen file    {p}")
    return out, d


def load_screened(tier="standard", screen_path=None, verbose=True):
    """Screened configurations for a tier, re-filtered from all_candidates.

    Re-filters rather than trusting the file's own `screened` list, because the
    first screening run used an ABSOLUTE rho<=1 gate -- and do-nothing itself
    peaks near rho 1.56, since grid2op tolerates several overflow steps before
    tripping. That gate rejected all 90 candidates, including ones whose
    survival exactly equalled nominal. Safety is judged relative to the nominal
    run here.
    """
    p = _find_screen(screen_path)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)

    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")
    max_drop, min_sens = TIERS[tier]

    nom_rho = d.get("nominal_max_rho")
    rho_thr = (nom_rho * 1.25) if nom_rho else float("inf")

    keep = []
    for c in d.get("all_candidates", []):
        if c.get("illegal") or c.get("exceptions"):
            continue
        if c.get("max_rho", 0.0) > rho_thr:
            continue
        if not (c.get("survival_drop") == c.get("survival_drop")):
            continue
        if c["survival_drop"] > max_drop or c["sensitivity_change"] < min_sens:
            continue
        keep.append(dict(sub_id=int(c["sub_id"]),
                         config=np.asarray(c["config"], dtype=int),
                         drop=float(c["survival_drop"]),
                         sens=float(c["sensitivity_change"])))
    if not keep:
        raise RuntimeError(
            f"tier {tier!r} matched no screened configuration in {p}. Loosen the "
            f"tier or re-run screen_topology.py -- do NOT bypass the screen.")
    if verbose:
        subs = sorted({k["sub_id"] for k in keep})
        print(f"[reconfig] tier={tier} -> {len(keep)} configs over {len(subs)} "
              f"substations {subs}")
        print(f"[reconfig]   survival drop  median "
              f"{100*np.median([k['drop'] for k in keep]):+.1f}%  "
              f"max {100*max(k['drop'] for k in keep):+.1f}%")
        print(f"[reconfig]   sensitivity    median "
              f"{100*np.median([k['sens'] for k in keep]):.1f}%  "
              f"max {100*max(k['sens'] for k in keep):.1f}%")
        print(f"[reconfig]   screen file    {p}  "
              f"(nominal_max_rho={nom_rho}, source={d.get('ptdf_source')})")
    return keep, d


class ReconfigEnv(PZMAEnvRecoDNLimit):
    def __init__(self, reconfig=None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(DEFAULT_RECONFIG)
        cfg.update(reconfig or {})
        self._rcfg = cfg

        if cfg["tier"] == SIGNFLIP_TIER:
            self._pool, _meta = load_signflip(cfg.get("screen_path"))
        else:
            self._pool, _meta = load_screened(cfg["tier"], cfg.get("screen_path"))
        self._subs = sorted({c["sub_id"] for c in self._pool})
        self._by_sub = {s: [c for c in self._pool if c["sub_id"] == s]
                        for s in self._subs}

        self._nominal_cfg = {s: np.ones(int(self.env_g2op.sub_info[s]), dtype=int)
                             for s in self._subs}
        self._active = {s: self._nominal_cfg[s].copy() for s in self._subs}

        self._oracle = (cfg.get("oracle") or "none").lower()
        self._sub_slices = {}
        off = 0
        for s in self._subs:
            n = int(self.env_g2op.sub_info[s])
            self._sub_slices[s] = (off, off + n)
            off += n
        self._n_oracle = off if self._oracle == "full" else 0
        self._obs_space_cache = {}

        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self._t = 0
        self._next_event = self._sample_interval()
        self._stats = dict(events=0, applied=0, reverts=0, illegal=0, steps=0)
        self._wrap_env()

    # -- schedule --------------------------------------------------------
    def _sample_interval(self):
        base = float(self._rcfg["interval_steps"])
        j = float(self._rcfg.get("interval_jitter", 0.0))
        return max(5, int(base * (1.0 + j * self._rng.uniform(-1, 1))))

    def _n_off_nominal(self):
        return sum(1 for s in self._subs
                   if not np.array_equal(self._active[s], self._nominal_cfg[s]))

    def _choose_event(self):
        """Pick the next reconfiguration, or a revert. Returns (sub_id, cfg)."""
        off = [s for s in self._subs
               if not np.array_equal(self._active[s], self._nominal_cfg[s])]
        # Revert if we are at the concurrency cap, or by chance.
        if off and (self._n_off_nominal() >= int(self._rcfg["n_active"])
                    or self._rng.random() < float(self._rcfg["revert_prob"])):
            s = off[self._rng.integers(len(off))]
            return s, self._nominal_cfg[s].copy(), True
        avail = [s for s in self._subs
                 if np.array_equal(self._active[s], self._nominal_cfg[s])]
        if not avail:
            return None, None, False
        s = avail[self._rng.integers(len(avail))]
        opts = self._by_sub[s]
        return s, opts[self._rng.integers(len(opts))]["config"].copy(), False

    # -- hook every grid transition, heuristic steps included -------------
    def _wrap_env(self):
        inner = self.env_g2op
        orig_step, orig_reset = inner.step, inner.reset

        def stepping(action):
            self._t += 1
            self._stats["steps"] += 1
            if self._t >= self._next_event:
                self._next_event = self._t + self._sample_interval()
                sub, cfg, is_revert = self._choose_event()
                if sub is not None:
                    try:
                        action = action + inner.action_space(
                            {"set_bus": {"substations_id": [(sub, cfg)]}})
                        self._active[sub] = cfg
                        self._stats["events"] += 1
                        self._stats["reverts" if is_revert else "applied"] += 1
                    except Exception:
                        self._stats["illegal"] += 1
            obs, rew, done, info = orig_step(action)
            if info.get("is_illegal"):
                self._stats["illegal"] += 1
            self._maybe_report()
            return obs, rew, done, info

        def resetting(*a, **k):
            self._t = 0
            self._next_event = self._sample_interval()
            # grid2op restores nominal topology on reset.
            for s in self._subs:
                self._active[s] = self._nominal_cfg[s].copy()
            return orig_reset(*a, **k)

        inner.step = stepping
        inner.reset = resetting

    def _maybe_report(self):
        n = int(self._rcfg.get("report_every", 0))
        if n and self._stats["steps"] % n == 0:
            s = self._stats
            print(f"[reconfig pid{os.getpid()}] steps={s['steps']} "
                  f"events={s['events']} applied={s['applied']} "
                  f"reverts={s['reverts']} illegal={s['illegal']} "
                  f"off_nominal_now={self._n_off_nominal()}/{len(self._subs)}")

    # -- oracle observation ----------------------------------------------
    def _oracle_vec(self):
        v = np.zeros(self._n_oracle, dtype=np.float32)
        for s in self._subs:
            lo, hi = self._sub_slices[s]
            v[lo:hi] = (np.asarray(self._active[s], dtype=np.float32) - 1.0)
        return v

    def _to_gym_obs(self, grid2op_obs):
        obs = super()._to_gym_obs(grid2op_obs)
        if self._oracle == "none":
            return obs
        extra = self._oracle_vec()
        for aid in list(obs.keys()):
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
                low=np.concatenate([base.low, np.zeros(k, dtype=base.dtype)]),
                high=np.concatenate([base.high, np.ones(k, dtype=base.dtype)]),
                dtype=base.dtype)
        return self._obs_space_cache[agent_id]

    # -- reporting --------------------------------------------------------
    def reconfig_summary(self):
        return dict(tier=self._rcfg["tier"], n_configs=len(self._pool),
                    substations=self._subs,
                    interval_steps=self._rcfg["interval_steps"],
                    n_active=self._rcfg["n_active"],
                    revert_prob=self._rcfg["revert_prob"],
                    oracle=self._oracle, n_oracle_feats=self._n_oracle)

    def reconfig_stats(self):
        return dict(self._stats)


def get_reconfig_preset(tier, oracle="none", **overrides):
    if tier is None or tier == "off":
        if (oracle or "none") != "none":
            raise ValueError("--reconfig_oracle needs --reconfig != off")
        return None
    if tier != SIGNFLIP_TIER and tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from "
                         f"{sorted(TIERS) + [SIGNFLIP_TIER]}")
    cfg = dict(DEFAULT_RECONFIG)
    cfg["tier"] = tier
    cfg["oracle"] = oracle or "none"
    cfg.update(overrides)
    return cfg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the screened reconfiguration set.")
    ap.add_argument("--screen", default=None)
    a = ap.parse_args()
    for t in ("conservative", "standard", "aggressive"):
        try:
            load_screened(t, a.screen)
        except Exception as e:
            print(f"[reconfig] tier={t}: {e}")
        print()
