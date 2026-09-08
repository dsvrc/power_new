#!/usr/bin/env python
"""
screen_signflip.py -- screen reconfigurations for the property that actually
makes a non-stationarity recoverable: they must flip the SIGN of each agent's
control authority WITHOUT displacing the binding constraint.

THE MISTAKE THIS CORRECTS
-------------------------
Three NS designs were built and all three came back with a recoverable fraction
near zero. authority.py explains why with one number:

    actuator reach   delta ~ 0.024 - 0.032 rho-units, CONSTANT across every NS
                           (it measures the agents, not the disturbance)
    field A=0.25     Delta = 0.276  -> kappa 0.17  -> oracle measured  +1%
    reconfig std     Delta = 0.106  -> kappa 0.02  -> oracle measured  -8%

Every one of those designs DISPLACED THE CONSTRAINT. Since delta is fixed by the
task, any Delta above ~0.03 rho-units is outside the reachable set, and knowing
the latent changes which infeasible action you take rather than whether you are
feasible. Information cannot pay. The old topology screen selected on
`sensitivity_change` = flow-redistribution magnitude, which is CORRELATED with
Delta -- it was actively selecting for the thing that destroys recoverability.

THE CORRECTED TARGET
--------------------
Do not move the constraint. Move the actuator's EFFECT.

    Delta ~ 0                      the grid is no harder than before
    sign(d rho_bind / d a_i) flips your lever now works backwards

Then:
    no-NS   agents learn the sign, act correctly, reach P
    oracle  knows the CURRENT sign, acts correctly, also reaches P
            -> gap_B = 0 BY CONSTRUCTION, not by screening luck
    blind   applies the stale sign, so its corrections LOAD the binding line
            instead of relieving it -- actively worse than doing nothing

Recoverable fraction is 100% by definition, because the oracle's achievable set
is identical to no-NS. And the blind agent does not merely lose its actuator, it
gets a negatively useful one, which is what makes MAPPO fall hard.

A busbar split genuinely flips PTDF signs -- a generator that relieved a corridor
can load it once the bus splits -- so this is the same physical mechanism and the
same "neighbouring operator" story, screened on the right quantity.

WHAT IS MEASURED, per candidate configuration
---------------------------------------------
  Delta       max rho under the config minus max rho under do-nothing.
              WANT ~ 0. This is the quantity kappa says must stay under ~0.03.
  flip_rate   fraction of zone agents whose d(max rho)/d(own curtailment)
              CHANGES SIGN between nominal and reconfigured topology.
              WANT LARGE. This is the information the blind agent lacks.

Reads topology_screen.json for the candidate list and its survival numbers, so
the earlier screening work is reused rather than repeated.
"""

import argparse
import json
import os
import sys
import traceback
from collections import Counter

import numpy as np


def make_env(env_name):
    import grid2op
    from grid2op.Action import PlayableAction
    from grid2op.Chronics import Multifolder
    try:
        from lightsim2grid import LightSimBackend as backend_cls
    except ImportError:
        from grid2op.Backend import PandaPowerBackend as backend_cls
    return grid2op.make(env_name, action_class=PlayableAction,
                        backend=backend_cls(), chronics_class=Multifolder)


def _max_rho(sim):
    try:
        r = np.asarray(sim.rho, dtype=float)
        r = r[np.isfinite(r)]
        return float(np.max(r)) if r.size else float("nan")
    except Exception:
        return float("nan")


def zone_curtail_gens(env):
    """Renewable generators each zone agent actually controls.

    Recomputed from the env exactly as utils.get_obs_act_attr_and_kwargs does
    (intersect the zone's gens with env.gen_renewable) rather than trusting a
    json key, so it cannot drift from what the agents really control.
    """
    import benchmarl.environments.G2OpPowerGrid.utils as g2u
    ren = np.where(env.gen_renewable)[0]
    out = {}
    for zname, z in sorted(g2u.ZONES_DICT.items()):
        inside = np.asarray(z.get("gen_inside_idx", []), dtype=int)
        g = np.intersect1d(inside, ren)
        if g.size:
            out[zname] = g
    return out


def _simulate(obs, act):
    try:
        sim, _, done, _ = obs.simulate(act)
        if done:
            return float("nan")
        return _max_rho(sim)
    except Exception:
        return float("nan")


def measure_config(env, obs, sub, cfg, zgens, curtail_level, margin, noise_floor):
    """(Delta, flip_rate, n_agents_measured) at this state, or None."""
    dn = env.action_space({})
    topo = env.action_space(
        {"set_bus": {"substations_id": [(int(sub), np.asarray(cfg, int))]}})

    base0 = _simulate(obs, dn)          # nominal topology, no control
    base1 = _simulate(obs, topo)        # reconfigured,     no control
    if not (np.isfinite(base0) and np.isfinite(base1)):
        return None
    Delta = base1 - base0

    flips = n = 0
    for zname, gens in zgens.items():
        cur = env.action_space(
            {"curtail": [(int(g), float(curtail_level)) for g in gens]})
        try:
            cur.limit_curtail_storage(obs, margin=margin)
        except Exception:
            pass
        s0 = _simulate(obs, cur) - base0            # nominal sensitivity
        s1 = _simulate(obs, topo + cur) - base1     # reconfigured sensitivity
        if not (np.isfinite(s0) and np.isfinite(s1)):
            continue
        if abs(s0) < noise_floor or abs(s1) < noise_floor:
            continue                                 # no measurable authority
        n += 1
        if np.sign(s0) != np.sign(s1):
            flips += 1
    if n == 0:
        return None
    return Delta, flips / n, n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=None)
    ap.add_argument("--screen", default="topology_screen.json")
    ap.add_argument("--n_chronics", type=int, default=2)
    ap.add_argument("--states", type=int, default=8,
                    help="states sampled per configuration (default: 8)")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--curtail_level", type=float, default=0.8,
                    help="curtailment probe level; 0.8 = a 20%% cut (default: 0.8)")
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--noise_floor", type=float, default=1e-4,
                    help="min |d rho| for a sensitivity to count as measurable")
    ap.add_argument("--max_delta", type=float, default=0.03,
                    help="(a) max constraint displacement; the measured actuator "
                         "reach is ~0.03 rho-units (default: 0.03)")
    ap.add_argument("--min_flip", type=float, default=0.25,
                    help="(b) min fraction of agents whose sign flips (default: 0.25)")
    ap.add_argument("--max_surv_drop", type=float, default=0.02)
    ap.add_argument("--out", default="signflip_screen.json")
    args = ap.parse_args()

    import grid2op
    env_name = args.env or os.path.join(grid2op.get_current_local_dir(),
                                        "l2rpn_idf_2023")
    print("=" * 78)
    print("SIGN-FLIP SCREEN -- recoverable by construction")
    print("=" * 78)
    print(f"  env    : {env_name}")
    if not os.path.exists(args.screen):
        print(f"  {args.screen} not found. Run screen_topology.py first.")
        return 2
    with open(args.screen, "r", encoding="utf-8") as f:
        prev = json.load(f)

    cands = [c for c in prev.get("all_candidates", [])
             if not c.get("illegal") and not c.get("exceptions")
             and c.get("survival_drop", 1.0) <= args.max_surv_drop]
    print(f"  candidates from the survival screen (drop <= "
          f"{100*args.max_surv_drop:.0f}%) : {len(cands)}")
    if not cands:
        print("  nothing to test.")
        return 2

    env = make_env(env_name)
    zgens = zone_curtail_gens(env)
    print(f"  zone agents with curtailable generation : {len(zgens)}")
    print(f"  probe: curtail to {args.curtail_level} on the agent's own renewables\n")

    # -- collect states once, reuse for every candidate ---------------------
    states = []
    dn = env.action_space({})
    for c in range(args.n_chronics):
        try:
            env.set_id(c)
        except Exception:
            pass
        obs = env.reset()
        for t in range(args.states * args.every):
            if t % args.every == 0:
                states.append(obs.copy())
                if len(states) >= args.states * args.n_chronics:
                    break
            obs, _, done, _ = env.step(dn)
            if done:
                break
    print(f"  sampled {len(states)} states\n")

    rows, reasons = [], Counter()
    for i, c in enumerate(cands):
        D, F, N = [], [], []
        for obs in states:
            m = measure_config(env, obs, c["sub_id"], c["config"], zgens,
                               args.curtail_level, args.margin, args.noise_floor)
            if m is None:
                reasons["unusable_state"] += 1
                continue
            D.append(m[0]); F.append(m[1]); N.append(m[2])
        if not D:
            reasons["no_states"] += 1
            continue
        rows.append(dict(sub_id=c["sub_id"], config=c["config"],
                         survival_drop=c.get("survival_drop"),
                         delta_rho=float(np.mean(np.abs(D))),
                         delta_rho_signed=float(np.mean(D)),
                         flip_rate=float(np.mean(F)),
                         agents_measured=float(np.mean(N)),
                         n_states=len(D)))
        print(f"  [{i+1}/{len(cands)}] sub {c['sub_id']:>3}  "
              f"|Delta|={np.mean(np.abs(D)):.4f}  flip={100*np.mean(F):5.1f}%  "
              f"agents={np.mean(N):.1f}")

    keep = [r for r in rows
            if r["delta_rho"] <= args.max_delta and r["flip_rate"] >= args.min_flip]
    keep.sort(key=lambda r: (-r["flip_rate"], r["delta_rho"]))

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  measured                       : {len(rows)}")
    print(f"  (a) |Delta| <= {args.max_delta:<6}          : "
          f"{sum(1 for r in rows if r['delta_rho'] <= args.max_delta)}")
    print(f"  (b) flip_rate >= {100*args.min_flip:.0f}%          : "
          f"{sum(1 for r in rows if r['flip_rate'] >= args.min_flip)}")
    print(f"  -> PASSING BOTH                : {len(keep)}")

    if keep:
        print(f"\n  {'sub':>5}{'|Delta|':>10}{'flip':>8}{'surv drop':>11}  config")
        for r in keep[:20]:
            print(f"  {r['sub_id']:>5}{r['delta_rho']:>10.4f}"
                  f"{100*r['flip_rate']:>7.1f}%"
                  f"{100*(r['survival_drop'] or 0):>10.1f}%  {r['config']}")
        print(f"\n  |Delta| median {np.median([r['delta_rho'] for r in keep]):.4f} "
              f"rho-units, against an actuator reach of ~0.03 -- so kappa >= 1 and")
        print("  the disturbance is INSIDE the reachable set.")
        print(f"  flip rate median "
              f"{100*np.median([r['flip_rate'] for r in keep]):.0f}% of agents.")
        print("\n  An oracle that knows the current signs can do everything the")
        print("  no-NS agent can. A blind agent applies stale signs and pushes the")
        print("  binding line the wrong way.")
    else:
        print("\n  NOTHING PASSED. Read the two counts above:")
        print("   - few with small |Delta| -> every reconfiguration here displaces")
        print("     the constraint, so this task cannot host a recoverable NS by")
        print("     this mechanism. Raising actuator authority (topology actions")
        print("     for the zone agents) is then the only route.")
        print("   - few with sign flips -> reconfigurations do not reverse control")
        print("     authority at this grid's operating points; the NS would be")
        print("     invisible AND harmless.")

    out = dict(env=env_name, source_screen=args.screen,
               criteria=dict(max_delta=args.max_delta, min_flip=args.min_flip,
                             curtail_level=args.curtail_level,
                             noise_floor=args.noise_floor,
                             max_surv_drop=args.max_surv_drop),
               n_measured=len(rows), n_passing=len(keep),
               screened=keep, all_measured=rows)
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
