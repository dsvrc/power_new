#!/usr/bin/env python
"""
diagnose_ns.py -- settle, by measurement, whether exogenous non-stationarity is
real in the Redispatching/Curtailment task, how severe it is, and whether the
agents can even see it.

Three questions, none of them answered by guessing:

  Q1  Is exogenous NS (opponent attacks / maintenance) actually running under
      the training configuration?  `env_g2op_config` is empty, so grid2op falls
      through to the dataset's own config.py -- which this script reads AND
      corroborates by rollout, rather than assuming either way.

  Q2  How much of the *learning problem* does it touch?  Not the fraction of
      simulated wall-clock time that is degraded, but the fraction of the frames
      an algorithm actually trains on.  These differ: agents are only consulted
      when rho > safe_max_rho, and outages push rho up, so the heuristic gate
      may concentrate exogenous events into the training distribution.  The
      ratio between the two is reported as the CONCENTRATION FACTOR.

  Q3  Is it observable?  A zone agent sees line_status only over its own
      line_large_idx.  An attack inside that set is a visible mode switch, which
      a memoryless MLP can condition on.  An attack outside it is a hidden
      regime switch, which a memoryless policy provably cannot adapt to -- it
      can only learn the regime-marginal compromise.  This script computes,
      per zone, what fraction of observed attacks were visible.

Attribution of cause is done by a PAIRED COUNTERFACTUAL (same chronics, opponent
on vs off) rather than by trusting the semantics of grid2op's info dict keys,
which vary across versions.  Raw info keys are reported for transparency but no
headline number depends on them.

Usage, on the machine holding the grid2op data:

    GRID2OP_DATA_PATH=/scratch/... python diagnose_ns.py --episodes 12
    python diagnose_ns.py --episodes 4 --quick        # smoke test
"""

import argparse
import collections
import json
import os
import sys
import time
import traceback

import numpy as np

# --------------------------------------------------------------------------
# Configuration mirroring the real training run
#   my_power_grid.yaml  +  configs/expes_config.yaml  +  main.py overrides
# --------------------------------------------------------------------------
TRAIN_CFG = dict(
    zone_names=[f"Zone{j}" for j in range(11)],
    use_global_obs=False,
    use_redispatching_agent=True,
    local_rewards=None,
    shuffle_chronics=True,
    regex_filter_chronics=".*-02-.*$",   # February only, from expes_config.yaml
    safe_max_rho=0.9,
    curtail_margin=30,
)

_TIME_SERIE_ID = "time serie id"


# --------------------------------------------------------------------------
# Imports of the REAL training classes (never reimplemented here)
# --------------------------------------------------------------------------
def import_env_classes():
    import grid2op
    from benchmarl.environments.G2OpPowerGrid import PZMAEnvWithHeuristics as mod
    from benchmarl.environments.G2OpPowerGrid.PZMAEnvWithHeuristics import (
        PZMAEnvRecoDNLimit,
    )
    return grid2op, mod, PZMAEnvRecoDNLimit


def opponent_off_kwargs():
    """grid2op's documented recipe for a genuinely inert opponent."""
    from ns_opponent import opponent_off
    return opponent_off()


# --------------------------------------------------------------------------
# The instrumented environment
# --------------------------------------------------------------------------
def make_probe_class(base_cls):
    class Probe(base_cls):
        """Counts every grid step and every agent decision point.

        A 'decision point' is a step where heuristic_actions returns [] -- i.e.
        exactly the moments the RL agents are asked to act, which is exactly
        what becomes a training frame.
        """

        def __init__(self, **kw):
            super().__init__(**kw)
            self.reset_counters()
            self._wrap_grid_step()

        # -- counters ---------------------------------------------------
        def reset_counters(self):
            self.grid_steps = 0
            self.grid_steps_lineout = 0
            self.grid_steps_attack_flag = 0
            self.grid_steps_maint = 0

            self.decisions = 0
            self.decisions_lineout = 0
            self.decisions_attack_flag = 0
            self.decisions_maint = 0

            self.attacked_lines = collections.Counter()
            self.lineout_lines = collections.Counter()
            self.rho_grid = []
            self.rho_decision = []
            self.n_out_grid = []
            self.info_keys_seen = set()
            self.maintenance_ever_scheduled = False

        # -- hook every underlying grid2op step -------------------------
        def _wrap_grid_step(self):
            inner = self.env_g2op
            orig = inner.step

            def counting_step(action):
                obs, rew, done, info = orig(action)
                try:
                    self._record_grid_step(obs, info)
                except Exception:
                    pass  # never let instrumentation kill a rollout
                return obs, rew, done, info

            inner.step = counting_step

        def _record_grid_step(self, obs, info):
            if isinstance(info, dict):
                self.info_keys_seen.update(info.keys())
            self.grid_steps += 1

            status = np.asarray(obs.line_status, dtype=bool)
            out_idx = np.where(~status)[0]
            n_out = int(out_idx.size)
            self.n_out_grid.append(n_out)
            if n_out:
                self.grid_steps_lineout += 1
                for li in out_idx:
                    self.lineout_lines[int(li)] += 1

            atk = info.get("opponent_attack_line") if isinstance(info, dict) else None
            if atk is not None:
                arr = np.asarray(atk).reshape(-1)
                if arr.dtype != bool:
                    arr = arr.astype(bool)
                if arr.any():
                    self.grid_steps_attack_flag += 1
                    for li in np.where(arr)[0]:
                        self.attacked_lines[int(li)] += 1

            tnm = getattr(obs, "time_next_maintenance", None)
            if tnm is not None:
                tnm = np.asarray(tnm)
                if np.any(tnm == 0):
                    self.grid_steps_maint += 1
                if np.any(tnm > 0):
                    self.maintenance_ever_scheduled = True

            if obs.rho.size:
                self.rho_grid.append(float(np.max(obs.rho)))

        # -- hook the decision gate -------------------------------------
        def heuristic_actions(self, g2op_obs, reward, done, info):
            acts = super().heuristic_actions(g2op_obs, reward, done, info)
            if not acts:  # empty list => the RL agents are being asked to act
                try:
                    self._record_decision(g2op_obs, info)
                except Exception:
                    pass
            return acts

        def _record_decision(self, obs, info):
            self.decisions += 1
            if not np.asarray(obs.line_status, dtype=bool).all():
                self.decisions_lineout += 1

            atk = info.get("opponent_attack_line") if isinstance(info, dict) else None
            if atk is not None:
                arr = np.asarray(atk).reshape(-1)
                if arr.dtype != bool:
                    arr = arr.astype(bool)
                if arr.any():
                    self.decisions_attack_flag += 1

            tnm = getattr(obs, "time_next_maintenance", None)
            if tnm is not None and np.any(np.asarray(tnm) == 0):
                self.decisions_maint += 1

            if obs.rho.size:
                self.rho_decision.append(float(np.max(obs.rho)))

    return Probe


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
def do_nothing_actions(env, rng=None):
    """Curtail entries have low=0 (1.0 == no curtailment); storage and
    redispatch entries have low<0 (0.0 == no action, since preprocess_act.json
    has add=0 for both).  This rule is derived from the space bounds, so it
    stays correct if the zone partition changes."""
    out = {}
    for a in env.agents:
        low = np.asarray(env.action_space(a).low)
        out[a] = np.where(low < 0, 0.0, 1.0).astype(np.float32)
    return out


def random_actions(env, rng):
    out = {}
    for a in env.agents:
        sp = env.action_space(a)
        out[a] = rng.uniform(np.asarray(sp.low), np.asarray(sp.high)).astype(np.float32)
    return out


POLICIES = {"do_nothing": do_nothing_actions, "random": random_actions}


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------
def run_episode(env, policy_fn, rng, chronic_id=None, max_steps=100_000):
    pinned = False
    if chronic_id is not None:
        try:
            env.env_g2op.set_id(chronic_id)
            pinned = True
        except Exception:
            pinned = False
    opts = {_TIME_SERIE_ID: chronic_id} if pinned else None

    env.reset(options=opts)

    steps = 0
    ret = 0.0
    while steps < max_steps:
        act = policy_fn(env, rng)
        _, rew, done, _trunc, _info = env.step(act)
        ret += float(rew["redispatching_agent"])
        steps += 1
        if any(done.values()):
            break
    return steps, ret, pinned


def run_condition(probe_cls, env_name, extra_g2op, label, policy_name,
                  n_episodes, chronic_ids, seed, max_steps, verbose=True):
    from grid2op.Chronics import Multifolder

    cfg = dict(TRAIN_CFG)
    cfg["env_name"] = env_name
    # Plain Multifolder instead of MultifolderWithCache: identical dynamics,
    # but avoids loading every February chronic into RAM
    # (the diagnostic does not need the throughput, and the cache can be GBs).
    cfg["env_g2op_config"] = dict(extra_g2op)
    cfg["env_g2op_config"].setdefault("chronics_class", Multifolder)

    t0 = time.time()
    env = probe_cls(**cfg)
    if verbose:
        print(f"    [{label}/{policy_name}] env built in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(seed)
    policy_fn = POLICIES[policy_name]

    lengths, returns, pins = [], [], []
    for i in range(n_episodes):
        cid = chronic_ids[i] if chronic_ids is not None else None
        try:
            steps, ret, pinned = run_episode(env, policy_fn, rng, cid, max_steps)
        except Exception as exc:
            print(f"    ! episode {i} failed: {type(exc).__name__}: {exc}")
            continue
        lengths.append(steps)
        returns.append(ret)
        pins.append(pinned)
        if verbose:
            print(f"    ep {i:>3}  chronic={cid}  agent_steps={steps:>5}  "
                  f"return={ret:>10.2f}")

    res = dict(
        label=label,
        policy=policy_name,
        n_episodes=len(lengths),
        chronics_pinned=bool(pins and all(pins)),
        ep_len_mean=float(np.mean(lengths)) if lengths else float("nan"),
        ep_len_std=float(np.std(lengths)) if lengths else float("nan"),
        ep_len_all=[int(x) for x in lengths],
        return_mean=float(np.mean(returns)) if returns else float("nan"),
        return_std=float(np.std(returns)) if returns else float("nan"),
        grid_steps=env.grid_steps,
        grid_steps_lineout=env.grid_steps_lineout,
        grid_steps_attack_flag=env.grid_steps_attack_flag,
        grid_steps_maint=env.grid_steps_maint,
        decisions=env.decisions,
        decisions_lineout=env.decisions_lineout,
        decisions_attack_flag=env.decisions_attack_flag,
        decisions_maint=env.decisions_maint,
        maintenance_ever_scheduled=env.maintenance_ever_scheduled,
        attacked_lines=dict(env.attacked_lines),
        lineout_lines=dict(env.lineout_lines),
        info_keys_seen=sorted(env.info_keys_seen),
        rho_grid_mean=float(np.mean(env.rho_grid)) if env.rho_grid else float("nan"),
        rho_decision_mean=(float(np.mean(env.rho_decision))
                           if env.rho_decision else float("nan")),
        mean_lines_out=float(np.mean(env.n_out_grid)) if env.n_out_grid else 0.0,
    )
    res["frac_grid_degraded"] = _safe_div(res["grid_steps_lineout"], res["grid_steps"])
    res["frac_decisions_degraded"] = _safe_div(res["decisions_lineout"], res["decisions"])
    res["concentration_factor"] = _safe_div(res["frac_decisions_degraded"],
                                            res["frac_grid_degraded"])
    res["grid_steps_per_decision"] = _safe_div(res["grid_steps"], res["decisions"])

    zones = {z: [int(x) for x in env.zones_dict[z]["line_large_idx"]]
             for z in env.zone_names}
    res["_zones"] = zones
    return res, env


def _safe_div(a, b):
    try:
        return float(a) / float(b) if b else float("nan")
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------
# Q3: observability of the disturbances
# --------------------------------------------------------------------------
def observability_report(zones, event_counter):
    total = sum(event_counter.values())
    rows = []
    if not total:
        return rows, total, []

    for zone, large in sorted(zones.items()):
        s = set(large)
        seen = sum(c for li, c in event_counter.items() if li in s)
        rows.append(dict(zone=zone, n_lines_observed=len(s),
                         events_visible=seen, frac_visible=seen / total))

    # how many zones can see each individual event occurrence
    per_event_visibility = []
    for li, c in event_counter.items():
        n_zones = sum(1 for large in zones.values() if li in set(large))
        per_event_visibility.extend([n_zones] * c)
    return rows, total, per_event_visibility


# --------------------------------------------------------------------------
# Static introspection: what does the dataset actually declare?
# --------------------------------------------------------------------------
def introspect_dataset(env_name, env_g2op):
    out = {}
    cfg_path = os.path.join(env_name, "config.py")
    out["config_py_path"] = cfg_path
    out["config_py_exists"] = os.path.exists(cfg_path)
    if out["config_py_exists"]:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                out["config_py"] = f.read()
        except Exception as exc:
            out["config_py"] = f"<unreadable: {exc}>"

    # Probe the LIVE env for opponent/maintenance wiring without assuming
    # attribute names -- report whatever is actually there.
    found = {}
    for attr in dir(env_g2op):
        low = attr.lower()
        if "opponent" in low or "maintenance" in low:
            try:
                v = getattr(env_g2op, attr)
            except Exception:
                continue
            if callable(v):
                continue
            if isinstance(v, (int, float, str, bool, type(None))):
                found[attr] = v
            elif isinstance(v, (list, tuple)) and len(v) < 60:
                found[attr] = list(v)
            else:
                found[attr] = f"<{type(v).__name__}>"
    out["live_opponent_maintenance_attrs"] = found

    try:
        ch = env_g2op.chronics_handler
        out["chronics_handler"] = type(ch).__name__
        out["chronics_real_data"] = type(ch.real_data).__name__
        inner = getattr(ch.real_data, "data", None)
        out["chronics_grid_value_class"] = type(inner).__name__ if inner is not None else None
        out["n_chronics_after_filter"] = int(
            np.asarray(ch.available_chronics()).shape[0])
    except Exception as exc:
        out["chronics_error"] = f"{type(exc).__name__}: {exc}"

    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def pct(x):
    return "n/a" if x != x else f"{100.0 * x:6.2f}%"


def num(x, w=8, p=2):
    return "n/a" if x != x else f"{x:{w}.{p}f}"


def _pick_pair(results):
    """The ON/OFF pair must share a policy for the comparison to mean anything."""
    for pol in ("do_nothing", "random"):
        on = results.get(("opponent_on", pol))
        if on is not None:
            return on, results.get(("opponent_off", pol)), pol
    for k, v in results.items():
        if k[0] == "opponent_on":
            return v, results.get(("opponent_off", k[1])), k[1]
    return None, None, None


def print_report(intro, results, args):
    W = 78
    def rule(c="-"):
        print(c * W)

    print()
    rule("=")
    print("EXOGENOUS NON-STATIONARITY DIAGNOSTIC")
    rule("=")

    # ---- Q1 ----------------------------------------------------------
    print("\nQ1  IS EXOGENOUS NS ACTUALLY RUNNING?")
    rule()
    print(f"  dataset config.py present : {intro['config_py_exists']}  "
          f"({intro['config_py_path']})")
    print(f"  chronics handler          : {intro.get('chronics_handler')} / "
          f"{intro.get('chronics_real_data')}")
    print(f"  grid value class          : {intro.get('chronics_grid_value_class')}")
    print(f"    -> a name containing 'WithMaintenance' means maintenance is ON")
    print(f"  chronics after Feb filter : {intro.get('n_chronics_after_filter')}")

    attrs = intro.get("live_opponent_maintenance_attrs", {})
    if attrs:
        print("\n  live env opponent/maintenance attributes:")
        for k in sorted(attrs):
            print(f"    {k:<42} = {attrs[k]}")
    else:
        print("\n  live env exposed no opponent/maintenance attributes")

    on, off, cmp_pol = _pick_pair(results)

    if on:
        print(f"\n  measured over {on['grid_steps']} grid steps "
              f"(opponent ON, {cmp_pol}):")
        print(f"    steps with >=1 line out       : {on['grid_steps_lineout']:>7}"
              f"  ({pct(on['frac_grid_degraded'])})")
        print(f"    steps with attack flag in info: {on['grid_steps_attack_flag']:>7}")
        print(f"    steps with active maintenance : {on['grid_steps_maint']:>7}")
        print(f"    maintenance ever scheduled    : {on['maintenance_ever_scheduled']}")
        print(f"    distinct lines seen attacked  : {len(on['attacked_lines'])}")
        print(f"    info keys observed            : {', '.join(on['info_keys_seen'][:12])}")

    if on and off:
        d_out = on["frac_grid_degraded"] - off["frac_grid_degraded"]
        print(f"\n  PAIRED COUNTERFACTUAL (same chronics, opponent ON vs OFF):")
        print(f"    degraded grid steps  ON : {pct(on['frac_grid_degraded'])}"
              f"   OFF : {pct(off['frac_grid_degraded'])}")
        print(f"    difference attributable to the opponent : {pct(d_out)}")
        verdict = ("OPPONENT IS ACTIVE" if abs(d_out) > 1e-6
                   or on["grid_steps_attack_flag"] > 0 else
                   "NO OPPONENT EFFECT DETECTED")
        print(f"    -> {verdict}")

    # ---- Q2 ----------------------------------------------------------
    print("\n\nQ2  HOW MUCH OF THE LEARNING PROBLEM DOES IT TOUCH?")
    rule()
    hdr = (f"  {'condition':<22}{'grid deg':>10}{'decis deg':>11}"
           f"{'conc x':>9}{'grid/dec':>10}{'ep len':>9}{'ret sd':>10}")
    print(hdr)
    for (label, pol), r in results.items():
        print(f"  {label + '/' + pol:<22}"
              f"{pct(r['frac_grid_degraded']):>10}"
              f"{pct(r['frac_decisions_degraded']):>11}"
              f"{num(r['concentration_factor'], 8, 2):>9}"
              f"{num(r['grid_steps_per_decision'], 9, 1):>10}"
              f"{num(r['ep_len_mean'], 8, 0):>9}"
              f"{num(r['return_std'], 9, 1):>10}")

    print("\n  'grid deg'  = share of simulated grid steps with a line out")
    print("  'decis deg' = share of AGENT DECISION POINTS with a line out")
    print("                (decision points are what become training frames)")
    print("  'conc x'    = decis deg / grid deg. >1 means the rho>0.9 gate")
    print("                CONCENTRATES outages into the training distribution.")
    print("  'grid/dec'  = grid steps consumed per agent decision (semi-MDP dilation)")

    if on and off:
        print(f"\n  SEVERITY, opponent ON vs OFF ({cmp_pol} policy, paired chronics):")
        print(f"    mean episode length   : {num(on['ep_len_mean'],8,0)}"
              f"  vs {num(off['ep_len_mean'],8,0)}")
        print(f"    return std dev        : {num(on['return_std'],8,1)}"
              f"  vs {num(off['return_std'],8,1)}")
        print(f"    mean rho at decisions : {num(on['rho_decision_mean'],8,3)}"
              f"  vs {num(off['rho_decision_mean'],8,3)}")
        surv = _safe_div(on["ep_len_mean"], off["ep_len_mean"])
        varr = _safe_div(on["return_std"], off["return_std"])
        print(f"\n    survival ratio ON/OFF        : {num(surv,8,3)}"
              "   (<1 => opponent shortens episodes)")
        print(f"    return-variance ratio ON/OFF : {num(varr,8,3)}"
              "   (>1 => extra variance into advantages)")

    # ---- Q3 ----------------------------------------------------------
    print("\n\nQ3  CAN THE AGENTS SEE IT?")
    rule()
    if on and on.get("_obs_rows"):
        rows, total, per_event = on["_obs_rows"], on["_obs_total"], on["_obs_per_event"]
        print(f"  {total} disturbance occurrences observed "
              f"(source: {on['_obs_source']})\n")
        print(f"  {'zone':<10}{'lines in obs':>14}{'visible':>10}{'frac visible':>15}")
        for r in rows:
            print(f"  {r['zone']:<10}{r['n_lines_observed']:>14}"
                  f"{r['events_visible']:>10}{pct(r['frac_visible']):>15}")
        if per_event:
            pe = np.asarray(per_event)
            print(f"\n  zones able to see a given occurrence: "
                  f"mean {pe.mean():.2f}, median {np.median(pe):.0f}, "
                  f"min {pe.min()}, max {pe.max()}")
            print(f"  occurrences visible to NO zone agent : "
                  f"{int((pe == 0).sum())} / {len(pe)}  "
                  f"({pct((pe == 0).mean())})")
            print("\n  Reading: an occurrence visible to 0 zones is a HIDDEN regime")
            print("  switch for every zone agent -- a memoryless MLP policy cannot")
            print("  adapt to it, only average over it. The redispatching agent has")
            print("  global line_status and so sees all of them structurally, though")
            print("  no agent observes remaining outage DURATION in any case.")
    else:
        print("  no disturbance occurrences recorded -- nothing to analyse.")

    # ---- verdict -----------------------------------------------------
    print("\n\nVERDICT")
    rule()
    if on:
        c = on["concentration_factor"]
        fd = on["frac_decisions_degraded"]
        if fd != fd or on["decisions"] == 0:
            print("  Inconclusive: no decision points recorded.")
        else:
            print(f"  {pct(fd)} of the frames an algorithm trains on occur while the")
            print(f"  grid is in a degraded topology.")
            if c == c and c > 1.15:
                print(f"  The heuristic gate CONCENTRATES these by {c:.2f}x relative to")
                print("  their share of simulated time. Exogenous NS is NOT negligible.")
            elif c == c and c < 0.85:
                print(f"  The heuristic gate DILUTES these ({c:.2f}x). Exogenous NS is")
                print("  under-represented in training relative to simulated time.")
            else:
                print("  The heuristic gate neither concentrates nor dilutes them.")
    print()
    rule("=")
    print()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10,
                    help="episodes per condition (default: 10)")
    ap.add_argument("--env", type=str, default="l2rpn_idf_2023",
                    help="grid2op env folder name (default: l2rpn_idf_2023)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=100_000,
                    help="cap on agent decisions per episode")
    ap.add_argument("--policies", type=str, default="do_nothing,random",
                    help="comma list from: do_nothing,random")
    ap.add_argument("--no-counterfactual", action="store_true",
                    help="skip the opponent-OFF condition (faster, weaker)")
    ap.add_argument("--no-pin", action="store_true",
                    help="do not pin chronic ids. Pinning is what makes the "
                         "ON/OFF comparison paired, but PZMAEnvWithHeuristics."
                         "reset() retries the SAME chronic in a `while done:` "
                         "loop, so a chronic that terminates on step 1 would "
                         "hang. Use this if a run stalls at reset.")
    ap.add_argument("--quick", action="store_true",
                    help="2 episodes, do_nothing only, no counterfactual")
    ap.add_argument("--opponent", type=str, default="default",
                    help="opponent preset from ns_opponent.py to measure as the "
                         "'ON' condition: default, frequent, hidden, brutal "
                         "(default: default)")
    ap.add_argument("--out", type=str, default="ns_diagnostic.json")
    args = ap.parse_args()

    if args.quick:
        args.episodes = 2
        args.policies = "do_nothing"
        args.no_counterfactual = True

    grid2op, mod, base_cls = import_env_classes()

    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    print(f"  grid2op version      : {grid2op.__version__}")
    print(f"  grid2op local dir    : {grid2op.get_current_local_dir()}")
    print(f"  GRID2OP_DATA_PATH    : {os.environ.get('GRID2OP_DATA_PATH', '<unset>')}")
    print(f"  env module in use    : {mod.__file__}")
    print("    ^ if this is a site-packages copy, it is the INSTALLED benchmarl,")
    print("      not your working tree. Re-run `pip install .` after edits.")

    env_name = os.path.join(grid2op.get_current_local_dir(), args.env)
    print(f"  env path             : {env_name}")
    if not os.path.isdir(env_name):
        print("\n  !! That path does not exist. GRID2OP_DATA_PATH is probably not")
        print("     honoured by this grid2op build -- nothing in this repo reads it;")
        print("     utils.py calls grid2op.get_current_local_dir(). Use")
        print("     grid2op.change_local_dir(...) or ~/.grid2opconfig.json instead.")
        sys.exit(2)

    probe_cls = make_probe_class(base_cls)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in POLICIES:
            print(f"unknown policy {p!r}; choose from {list(POLICIES)}")
            sys.exit(2)

    chronic_ids = None if args.no_pin else list(range(args.episodes))

    from ns_opponent import get_preset
    on_kwargs = get_preset(args.opponent)
    print(f"  opponent preset      : {args.opponent} -> "
          f"{sorted(on_kwargs.keys()) or 'dataset default'}")
    conditions = [("opponent_on", on_kwargs)]
    if not args.no_counterfactual:
        try:
            conditions.append(("opponent_off", opponent_off_kwargs()))
        except Exception as exc:
            print(f"\n  ! could not build opponent-off kwargs ({exc});"
                  " skipping counterfactual")

    results = {}
    failures = {}
    intro = None
    intro_from = None
    for label, extra in conditions:
        for pol in policies:
            print(f"\n--- condition={label}  policy={pol} "
                  f"({args.episodes} episodes) ---")
            try:
                res, env = run_condition(
                    probe_cls, env_name, extra, label, pol,
                    args.episodes, chronic_ids, args.seed, args.max_steps)
            except Exception as exc:
                tb = traceback.format_exc()
                print(f"  !! condition failed:\n{tb}")
                failures[f"{label}::{pol}"] = f"{type(exc).__name__}: {exc}"
                continue
            results[(label, pol)] = res
            # Introspect the ON env specifically: reading opponent wiring off an
            # opponent-disabled env would report the wrong opponent class.
            if intro is None or (label == "opponent_on" and intro_from != "opponent_on"):
                intro = introspect_dataset(env_name, env.env_g2op)
                intro_from = label
            try:
                env.env_g2op.close()
            except Exception:
                pass

    if failures:
        print("\n" + "!" * 78)
        print("CONDITIONS THAT FAILED TO RUN -- results below are INCOMPLETE")
        for k, v in failures.items():
            print(f"  {k}: {v}")
        if not any(k[0] == "opponent_on" for k in results):
            print("\n  The opponent-ON condition never ran, so this report says")
            print("  NOTHING about the preset you asked for. Fix the error above")
            print("  (try: python ns_opponent.py --smoke-test) and re-run.")
        print("!" * 78)

    if not results:
        print("\nNo condition completed. Nothing to report.")
        sys.exit(1)

    # Observability: prefer real attack flags; fall back to observed outages.
    for key, r in results.items():
        counter = collections.Counter(
            {int(k): v for k, v in r["attacked_lines"].items()})
        source = "info['opponent_attack_line']"
        if not counter:
            counter = collections.Counter(
                {int(k): v for k, v in r["lineout_lines"].items()})
            source = "observed line outages (attack flag unavailable)"
        r["_obs_rows"], r["_obs_total"], r["_obs_per_event"] = \
            observability_report(r["_zones"], counter)
        r["_obs_source"] = source

    print_report(intro or {}, results, args)

    dump = {
        "setup": {
            "grid2op_version": grid2op.__version__,
            "local_dir": str(grid2op.get_current_local_dir()),
            "env_path": env_name,
            "env_module": mod.__file__,
            "train_cfg": {k: v for k, v in TRAIN_CFG.items()},
            "episodes": args.episodes,
            "seed": args.seed,
        },
        "introspection": intro,
        "introspection_from_condition": intro_from,
        "failed_conditions": failures,
        "conditions": {f"{k[0]}::{k[1]}": {kk: vv for kk, vv in v.items()
                                           if not kk.startswith("_")}
                       for k, v in results.items()},
        "observability": {f"{k[0]}::{k[1]}": {
            "source": v.get("_obs_source"),
            "total_occurrences": v.get("_obs_total"),
            "per_zone": v.get("_obs_rows"),
        } for k, v in results.items()},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"Full results written to {os.path.abspath(args.out)}\n")

    if failures:
        sys.exit(3)   # non-zero so a failed treatment cannot pass unnoticed


if __name__ == "__main__":
    main()
