#!/usr/bin/env python
"""
screen_topology.py -- build a set of exogenous substation reconfigurations that
are RECOVERABLE by construction, and prove it before any training runs.

WHY THIS EXISTS
---------------
The DLR-field NS failed the recoverability test: an oracle handed the true field
bought +5% return and +1% survival over blind. Diagnosis: the field attacks
rho = flow/limit by shrinking the limit, so it pushes the system against a HARD
cliff (rho>1 trips the line), and curtailment plus a few storage units cannot
offset a 25% capacity loss on a binding corridor. Knowing the future does not
help when you cannot act far enough.

Design rule that follows: the NS must change WHICH ACTION IS OPTIMAL without
changing HOW BINDING THE CONSTRAINT IS. Same feasible set, different optimum.

A busbar split at a substation redistributes flows across the network -- it
changes the PTDF sensitivities, i.e. the map from every agent's injection to
every line's flow -- WITHOUT removing capacity. So the feasible set is nearly
untouched (gap_B ~ 0) while the optimal action changes completely. A blind agent
keeps applying the old sensitivity and its corrections become counterproductive,
which is a soft efficiency loss rather than a game over.

It is also the most routine operation in real transmission control, and a
NEIGHBOURING operator's reconfiguration is exactly what you do not observe.
obs_attr_to_keep_default carries no topo_vect, so agents already never see it.

WHAT THIS SCRIPT DECIDES
------------------------
For every candidate reconfiguration it measures three things and keeps only the
candidates that pass all three:

  (a) SAFE      no islanding, no divergence, no game-over on application
  (b) BENIGN    do-nothing survival stays within --max_surv_drop of nominal
                -> this is what CONSTRUCTS gap_B ~ 0, rather than hoping for it
  (c) POTENT    the sensitivity actually moves, above --min_ptdf_change
                -> otherwise the NS is invisible AND harmless

(a) and (b) give the recoverability guarantee. (c) guarantees it bites. The
severity dial is then how many screened configurations fire and how often --
turning severity up never touches feasibility, which is precisely what the field
design got wrong.

Writes topology_screen.json. Run it before building anything on top.
"""

import argparse
import itertools
import json
import os
import sys
import time
import traceback

import numpy as np


def load_zones():
    import benchmarl.environments.G2OpPowerGrid.utils as g2u
    return g2u.ZONES_DICT


def make_env(env_name, with_rules=True):
    import grid2op
    from grid2op.Action import PlayableAction
    from grid2op.Chronics import Multifolder
    try:
        from lightsim2grid import LightSimBackend as backend_cls
    except ImportError:
        from grid2op.Backend import PandaPowerBackend as backend_cls
    return grid2op.make(env_name, action_class=PlayableAction,
                        backend=backend_cls(), chronics_class=Multifolder)


# ---------------------------------------------------------------------------
def sub_element_slice(env, sub_id):
    start = int(np.sum(env.sub_info[:sub_id]))
    return start, start + int(env.sub_info[sub_id])


def line_positions_at_sub(env, sub_id):
    """Which topo_vect positions at this substation are LINE ends.

    A bus carrying no line is electrically islanded, so every candidate split
    must leave at least one line on each bus.
    """
    lo, hi = sub_element_slice(env, sub_id)
    types = env.grid_objects_types            # (n_topo, 6): sub, load, gen, or, ex, storage
    out = []
    for p in range(lo, hi):
        row = types[p]
        is_line = (row[3] != -1) or (row[4] != -1)
        out.append(bool(is_line))
    return np.asarray(out, dtype=bool)


def candidate_splits(env, sub_id, max_per_sub, rng):
    """Binary bus assignments with both buses non-empty and each holding a line."""
    n = int(env.sub_info[sub_id])
    if n < 4:
        return []
    is_line = line_positions_at_sub(env, sub_id)
    if is_line.sum() < 2:
        return []

    seen, out = set(), []
    space = 2 ** n
    exhaustive = space <= 256
    pool = (itertools.product([1, 2], repeat=n) if exhaustive else None)
    tries = 0
    while len(out) < max_per_sub and tries < max_per_sub * 40:
        tries += 1
        if exhaustive:
            try:
                cfg = np.array(next(pool), dtype=int)
            except StopIteration:
                break
        else:
            cfg = rng.integers(1, 3, size=n)
        if cfg.min() == cfg.max():
            continue                                   # not a split
        if not (is_line[cfg == 1].any() and is_line[cfg == 2].any()):
            continue                                   # a bus with no line
        key = tuple(cfg)
        alt = tuple(3 - c for c in cfg)                # bus labels are arbitrary
        if key in seen or alt in seen:
            continue
        seen.add(key)
        out.append(cfg)
    return out


def topo_action(env, sub_id, cfg):
    return env.action_space({"set_bus": {"substations_id": [(sub_id, cfg)]}})


# ---------------------------------------------------------------------------
def get_ptdf(env):
    """The declared operator. Missing is reported, never silently proxied."""
    for obj, name in ((getattr(env, "backend", None), "backend"),):
        if obj is None:
            continue
        for meth in ("get_ptdf", "get_PTDF"):
            if hasattr(obj, meth):
                try:
                    return np.asarray(getattr(obj, meth)(), dtype=float), f"{name}.{meth}"
                except Exception:
                    pass
    return None, None


def rollout_do_nothing(env, chronic_id, n_steps, apply_at=None, sub_id=None, cfg=None):
    """Do-nothing rollout; optionally inject a reconfiguration at `apply_at`.

    Returns (steps_survived, max_rho_after, illegal, exception, flows_before,
    flows_after).
    """
    try:
        env.set_id(chronic_id)
    except Exception:
        pass
    env.reset()
    dn = env.action_space({})
    flows_b = flows_a = None
    illegal = False
    exc = None
    steps = 0
    max_rho = 0.0

    for t in range(n_steps):
        act = dn
        if apply_at is not None and t == apply_at and sub_id is not None:
            flows_b = np.abs(env.get_obs().p_or).copy()
            act = topo_action(env, sub_id, cfg)
        obs, _, done, info = env.step(act)
        if apply_at is not None and t == apply_at:
            illegal = bool(info.get("is_illegal", False))
            e = info.get("exception")
            exc = (type(e[0]).__name__ if isinstance(e, (list, tuple)) and e
                   else (type(e).__name__ if e else None))
            flows_a = np.abs(obs.p_or).copy()
        steps += 1
        if apply_at is None or t >= apply_at:
            max_rho = max(max_rho, float(np.max(obs.rho)) if obs.rho.size else 0.0)
        if done:
            break
    return steps, max_rho, illegal, exc, flows_b, flows_a


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=None)
    ap.add_argument("--n_chronics", type=int, default=3,
                    help="chronics per candidate (default: 3)")
    ap.add_argument("--apply_at", type=int, default=40,
                    help="step at which the reconfiguration fires (default: 40)")
    ap.add_argument("--steps", type=int, default=200,
                    help="total do-nothing steps per rollout (default: 200)")
    ap.add_argument("--max_per_sub", type=int, default=6,
                    help="candidate splits per substation (default: 6)")
    ap.add_argument("--max_subs", type=int, default=40,
                    help="substations to consider (default: 40)")
    ap.add_argument("--max_surv_drop", type=float, default=0.10,
                    help="(b) max allowed relative survival loss (default: 0.10)")
    ap.add_argument("--min_ptdf_change", type=float, default=0.02,
                    help="(c) min relative sensitivity change (default: 0.02)")
    ap.add_argument("--max_rho_ratio", type=float, default=1.25,
                    help="""(a) max rho after the switch, as a MULTIPLE of the
                            nominal run's own max rho. Absolute thresholds do not
                            work here: do-nothing already peaks near rho 1.56,
                            because grid2op tolerates several overflow steps
                            before tripping. An absolute rho<=1 gate rejected all
                            90 candidates including ones whose survival exactly
                            equalled nominal. (default: 1.25)""")
    ap.add_argument("--outside_zones_only", action="store_true", default=True,
                    help="only substations outside every agent zone (the "
                         "'neighbouring TSO' story). Default on.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="topology_screen.json")
    args = ap.parse_args()

    import grid2op
    env_name = args.env or os.path.join(grid2op.get_current_local_dir(),
                                        "l2rpn_idf_2023")
    print("=" * 78)
    print("TOPOLOGY SCREEN -- building a recoverable exogenous NS")
    print("=" * 78)
    print(f"  env : {env_name}")

    env = make_env(env_name)
    rng = np.random.default_rng(args.seed)
    print(f"  n_sub={env.n_sub}  n_line={env.n_line}")

    ptdf0, ptdf_src = get_ptdf(env)
    if ptdf0 is None:
        print("\n  PTDF not exposed by this backend. Falling back to a MEASURED")
        print("  proxy: relative flow redistribution at identical injections,")
        print("  ||dp_or|| / ||p_or||. Reported as ptdf_source='flow_proxy' so the")
        print("  distinction is never lost downstream.")
    else:
        print(f"  PTDF via {ptdf_src}, shape {ptdf0.shape}")

    # -- which substations count as exogenous -------------------------------
    zones = load_zones()
    owned = set()
    for z in zones.values():
        owned.update(int(s) for s in z.get("sub_inside_ids", []))
    all_subs = list(range(env.n_sub))
    cands = [s for s in all_subs if s not in owned] if args.outside_zones_only else all_subs
    print(f"  substations inside agent zones : {len(owned)}")
    print(f"  exogenous candidates           : {len(cands)}")
    if len(cands) < 3:
        print("\n  Too few substations outside the zones. Re-run with "
              "--outside_zones_only false; agents never act on topology here, so "
              "any substation is exogenous in practice -- it just weakens the "
              "'neighbouring operator' framing.")
    rng.shuffle(cands)
    cands = cands[:args.max_subs]

    # -- nominal baseline ---------------------------------------------------
    print("\n--- nominal (no reconfiguration) ---")
    base, base_rho = [], []
    for c in range(args.n_chronics):
        s, mr, *_ = rollout_do_nothing(env, c, args.steps)
        base.append(s); base_rho.append(mr)
        print(f"  chronic {c}: do-nothing survived {s} steps (max rho {mr:.3f})")
    base_mean = float(np.mean(base))
    base_max_rho = float(np.max(base_rho))
    rho_thr = base_max_rho * args.max_rho_ratio
    print(f"  nominal mean survival = {base_mean:.1f} steps")
    print(f"  nominal max rho       = {base_max_rho:.3f}  -> safety threshold "
          f"{rho_thr:.3f} ({args.max_rho_ratio}x nominal)")

    # -- screen -------------------------------------------------------------
    rows = []
    t0 = time.time()
    n_eval = 0
    for sub in cands:
        for cfg in candidate_splits(env, sub, args.max_per_sub, rng):
            n_eval += 1
            surv, rhos, illegal, excs, dflow = [], [], False, [], []
            for c in range(args.n_chronics):
                try:
                    s, mr, il, ex, fb, fa = rollout_do_nothing(
                        env, c, args.steps, args.apply_at, sub, cfg)
                except Exception as e:
                    s, mr, il, ex, fb, fa = 0, float("inf"), True, type(e).__name__, None, None
                surv.append(s); rhos.append(mr)
                illegal = illegal or il
                if ex: excs.append(ex)
                if fb is not None and fa is not None:
                    nb = np.linalg.norm(fb)
                    if nb > 1e-6:
                        dflow.append(float(np.linalg.norm(fa - fb) / nb))

            ptdf_change = float(np.mean(dflow)) if dflow else float("nan")
            if ptdf0 is not None:
                try:
                    p1, _ = get_ptdf(env)
                    if p1 is not None and p1.shape == ptdf0.shape:
                        d = np.linalg.norm(p1 - ptdf0) / max(1e-9, np.linalg.norm(ptdf0))
                        ptdf_change = float(d)
                except Exception:
                    pass

            sm = float(np.mean(surv))
            drop = (base_mean - sm) / base_mean if base_mean > 0 else float("nan")
            rows.append(dict(
                sub_id=int(sub), config=[int(x) for x in cfg],
                survival=sm, survival_drop=drop, max_rho=float(np.max(rhos)),
                illegal=bool(illegal), exceptions=sorted(set(excs)),
                sensitivity_change=ptdf_change,
                ptdf_source=("ptdf" if ptdf0 is not None else "flow_proxy"),
            ))

    dt = time.time() - t0
    print(f"\n  evaluated {n_eval} candidates in {dt:.0f}s")

    # -- verdict ------------------------------------------------------------
    def passes(r):
        safe = (not r["illegal"]) and (not r["exceptions"]) and r["max_rho"] <= rho_thr
        benign = r["survival_drop"] == r["survival_drop"] and r["survival_drop"] <= args.max_surv_drop
        potent = r["sensitivity_change"] == r["sensitivity_change"] and \
            r["sensitivity_change"] >= args.min_ptdf_change
        return safe, benign, potent

    keep = []
    n_safe = n_benign = n_potent = 0
    for r in rows:
        s, b, p = passes(r)
        r["safe"], r["benign"], r["potent"] = s, b, p
        n_safe += s; n_benign += b; n_potent += p
        if s and b and p:
            keep.append(r)

    print("\n" + "=" * 78)
    print("SCREEN RESULT")
    print("=" * 78)
    print(f"  candidates evaluated        : {len(rows)}")
    print(f"  (a) SAFE    no trip/island  : {n_safe}")
    print(f"  (b) BENIGN  survival kept   : {n_benign}   "
          f"(drop <= {100*args.max_surv_drop:.0f}%)")
    print(f"  (c) POTENT  sensitivity moved: {n_potent}   "
          f"(>= {100*args.min_ptdf_change:.0f}%)")
    print(f"  -> PASSING ALL THREE        : {len(keep)}")

    if keep:
        keep.sort(key=lambda r: -r["sensitivity_change"])
        print(f"\n  {'sub':>5}{'surv':>8}{'drop':>8}{'maxrho':>8}{'sens':>8}  config")
        for r in keep[:15]:
            print(f"  {r['sub_id']:>5}{r['survival']:>8.1f}"
                  f"{100*r['survival_drop']:>7.1f}%{r['max_rho']:>8.3f}"
                  f"{100*r['sensitivity_change']:>7.1f}%  {r['config']}")
        sens = [r["sensitivity_change"] for r in keep]
        drops = [r["survival_drop"] for r in keep]
        print(f"\n  sensitivity change: median {100*np.median(sens):.1f}% "
              f"max {100*np.max(sens):.1f}%")
        print(f"  survival drop     : median {100*np.median(drops):.1f}% "
              f"max {100*np.max(drops):.1f}%")
        print("\n  These reconfigurations move the sensitivities while leaving")
        print("  do-nothing survival essentially intact -- gap_B is small BY")
        print("  CONSTRUCTION, and this table is the evidence for it.")
    else:
        print("\n  NOTHING PASSED. Do not build on this design yet. Read the")
        print("  per-criterion counts above:")
        print("   - few SAFE   -> splits are islanding; require more lines per bus")
        print("   - few BENIGN -> reconfiguration IS costing capacity, which is the")
        print("                   same failure the DLR field had. Loosen nothing;")
        print("                   pick gentler substations.")
        print("   - few POTENT -> changes are invisible; the NS would be harmless.")

    out = dict(
        env=env_name,
        nominal_survival=base_mean,
        nominal_max_rho=base_max_rho,
        rho_threshold=rho_thr,
        criteria=dict(max_surv_drop=args.max_surv_drop,
                      min_ptdf_change=args.min_ptdf_change,
                      max_rho_ratio=args.max_rho_ratio),
        ptdf_source=("ptdf" if ptdf0 is not None else "flow_proxy"),
        n_evaluated=len(rows), n_passing=len(keep),
        screened=keep, all_candidates=rows,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  written to {os.path.abspath(args.out)}")
    try:
        env.close()
    except Exception:
        pass
    return 0 if keep else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
