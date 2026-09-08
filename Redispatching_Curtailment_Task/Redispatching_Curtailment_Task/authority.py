#!/usr/bin/env python
"""
authority.py -- measure the ACTUATOR AUTHORITY RATIO kappa, which predicts how
much of a non-stationarity is recoverable by information BEFORE you train.

THE OBSERVATION THIS FORMALISES
-------------------------------
Three NS designs were built on this task and all three came back with a
recoverable fraction near zero:

    hidden opponent   do-nothing lost 53% survival -- and do-nothing uses NO
                      information, so that loss is pure capacity
    DLR field         oracle handed the true 5-parameter field: +1% survival,
                      +5% return over blind, at 1M frames
    reconfiguration   oracle handed the true bus vectors: LEADS for the first
                      third, then crosses over and finishes 8% BELOW blind

Different mechanisms, same outcome. The common cause is not the NS design and
not the algorithm. It is that performance here is set by staying under a hard
overload constraint, and the agents' actuators -- curtailment plus a handful of
storage units -- cannot move that constraint as far as the disturbance does.

THE CLAIM
---------
Let the binding constraint be rho(x, a, z) <= 1 with z the latent NS parameter.
Write, at a state x:

    Delta(x) = displacement of the binding constraint caused by the NS,
               holding the action fixed
    delta(x) = the most the FULL joint action set can move the binding
               constraint back, at that state
    kappa(x) = delta(x) / Delta(x)

If kappa < 1 the disturbance is outside the reachable set: no action, chosen
with any amount of knowledge, restores feasibility. Knowing z changes WHICH
infeasible action you take, not WHETHER you are feasible. So the value of
information collapses:

    kappa < 1   =>   gap_A -> 0  and  gap_B -> the whole penalty
    kappa > 1   =>   information is actionable; gap_A can be recovered

This makes "is my benchmark's non-stationarity recoverable?" a question you
answer with a few hundred power flows instead of three 16-hour training runs --
and it is a DESIGN RULE: an NS benchmark that wants to test adaptation must be
built inside the actuators' reach, or it tests capacity instead.

kappa is a per-state quantity; what matters is its distribution over the states
where the agents are actually consulted (rho > safe_max_rho), because those are
the only states that become training frames.

WHAT THIS SCRIPT DOES
---------------------
Rolls out do-nothing, and at sampled states measures:
  delta  : max |rho| reduction achievable by the strongest joint action
  Delta  : max |rho| increase caused by the NS with the action held fixed
  kappa  : their ratio, reported as a distribution and as P(kappa > 1)

for each NS family. No training, no learning, no fitting.
"""

import argparse
import json
import os
import sys
import traceback

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


def candidate_actions(env, obs, curtail_margin=30):
    """A SEARCH over the action set, not one guess at the best action.

    The first version used a single "maximum relief" action -- every renewable
    curtailed to zero, every storage at full discharge -- and it failed twice:

      * it measured delta < 0, i.e. it made loading WORSE. Curtailment only
        relieves a line if the zone is exporting through it; in an importing
        zone it raises imports and loads the line harder. Without a PTDF there
        is no way to know the sign a priori, so the direction has to be
        SEARCHED rather than assumed.
      * being that extreme, it diverged the simulated power flow, so `done` came
        back True and 117 of 120 sampled states were discarded.

    Every candidate is passed through limit_curtail_storage exactly as
    PZMAEnvRecoDNLimit.fix_action does, so the reachable set measured here is
    the one training actually executes.
    """
    ren = np.where(env.gen_renewable)[0]
    out = [("do_nothing", env.action_space({}))]
    for lvl in (0.0, 0.25, 0.5, 0.75):
        if ren.size:
            out.append((f"curtail@{lvl}",
                        env.action_space({"curtail": [(int(g), lvl) for g in ren]})))
    if env.n_storage:
        pmax = env.storage_max_p_prod
        for name, sign in (("stor_discharge", -1.0), ("stor_charge", +1.0)):
            out.append((name, env.action_space(
                {"set_storage": [(int(s), sign * float(pmax[s]))
                                 for s in range(env.n_storage)]})))
    fixed = []
    for name, a in out:
        try:
            a.limit_curtail_storage(obs, margin=curtail_margin)
        except Exception:
            pass
        fixed.append((name, a))
    return fixed


def _max_rho(sim_obs):
    try:
        r = np.asarray(sim_obs.rho, dtype=float)
        r = r[np.isfinite(r)]
        return float(np.max(r)) if r.size else float("nan")
    except Exception:
        return float("nan")


def measure_state(env, obs, ns_kind, ns_arg, safe_max_rho, reasons):
    """Return (rho0, delta, Delta, best) at this state, or None if unusable.

    Both quantities are ACTION EFFECTS measured at the same one-step horizon
    against the same do-nothing baseline, so neither is contaminated by the
    grid's own evolution:

        baseline = max rho after do-nothing
        delta    = baseline - min over candidate actions of max rho   (>= 0,
                   because do_nothing is itself a candidate)
        Delta    = max rho under the NS - baseline
    """
    rho0 = _max_rho(obs)
    if not np.isfinite(rho0):
        reasons["bad_rho"] += 1
        return None

    dn = env.action_space({})
    try:
        sim0, _, d0, _ = obs.simulate(dn)
        if d0:
            reasons["baseline_done"] += 1
            return None
        baseline = _max_rho(sim0)
    except Exception:
        reasons["baseline_raised"] += 1
        return None
    if not np.isfinite(baseline):
        reasons["bad_rho"] += 1
        return None

    # -- delta: search the action set; a failing candidate is SKIPPED, never
    #    a reason to discard the state (that is what threw away 117 of 120).
    best_rho, best_name, n_ok = baseline, "do_nothing", 0
    for name, a in candidate_actions(env, obs):
        try:
            sim, _, done, _ = obs.simulate(a)
            if done:
                reasons["cand_done"] += 1
                continue
            r = _max_rho(sim)
            if not np.isfinite(r):
                continue
            n_ok += 1
            if r < best_rho:
                best_rho, best_name = r, name
        except Exception:
            reasons["cand_raised"] += 1
            continue
    if n_ok == 0:
        reasons["no_candidate"] += 1
        return None
    delta = baseline - best_rho

    # -- Delta: how far the NS pushes the constraint, action held fixed ------
    if ns_kind == "field":
        # Ratings scale by f, so rho scales by 1/f; the trough is f = 1 - A.
        f = 1.0 - float(ns_arg)
        Delta = baseline * (1.0 / max(f, 1e-3) - 1.0)
    elif ns_kind == "reconfig":
        sub, cfg = ns_arg
        try:
            act = env.action_space(
                {"set_bus": {"substations_id": [(int(sub), np.asarray(cfg, int))]}})
            sim1, _, d1, _ = obs.simulate(act)
            if d1:
                reasons["ns_done"] += 1
                return None
            Delta = _max_rho(sim1) - baseline
        except Exception:
            reasons["ns_raised"] += 1
            return None
    else:
        return None

    if not (np.isfinite(delta) and np.isfinite(Delta)):
        reasons["bad_rho"] += 1
        return None
    return rho0, float(delta), float(Delta), best_name


def run(env, ns_kind, ns_args, n_chronics, steps, every, safe_max_rho, label):
    import collections
    rows, reasons = [], collections.Counter()
    best_counts = collections.Counter()
    dn = env.action_space({})
    attempted = 0
    for c in range(n_chronics):
        try:
            env.set_id(c)
        except Exception:
            pass
        obs = env.reset()
        for t in range(steps):
            if t % every == 0:
                attempted += 1
                arg = (ns_args if ns_kind == "field"
                       else ns_args[np.random.randint(len(ns_args))])
                m = measure_state(env, obs, ns_kind, arg, safe_max_rho, reasons)
                if m is not None:
                    rho0, d, D, best = m
                    best_counts[best] += 1
                    rows.append(dict(rho0=rho0, delta=d, Delta=D,
                                     consulted=bool(rho0 > safe_max_rho)))
            obs, _, done, _ = env.step(dn)
            if done:
                break
    print(f"\n  [{label}] {len(rows)}/{attempted} states usable"
          + (f"  rejected: {dict(reasons)}" if reasons else ""))
    if best_counts:
        top = ", ".join(f"{k}:{v}" for k, v in best_counts.most_common(4))
        print(f"  best relief action was  {top}")
    return summarise(rows, label)


def summarise(rows, label):
    out = {"label": label, "n": len(rows)}
    if not rows:
        print(f"\n{label}: no usable states")
        return out
    D = np.array([r["Delta"] for r in rows])
    d = np.array([r["delta"] for r in rows])
    con = np.array([r["consulted"] for r in rows])
    # Guard the ratio with NaN, never an epsilon: where the disturbance is
    # genuinely ~0 the ratio is meaningless, and a 1e-12 floor manufactures
    # enormous values that poison the average.
    k = np.where(np.abs(D) > 1e-3, d / np.where(np.abs(D) > 1e-3, D, np.nan), np.nan)
    fin = np.isfinite(k)

    def stats(mask, name):
        kk = k[mask & fin]
        if kk.size == 0:
            print(f"  {name:<22} no valid states")
            return {}
        p = float(np.mean(kk > 1.0))
        s = dict(n=int(kk.size), median=float(np.median(kk)),
                 q25=float(np.percentile(kk, 25)), q75=float(np.percentile(kk, 75)),
                 p_kappa_gt_1=p)
        print(f"  {name:<22} n={s['n']:<5} kappa median {s['median']:>7.3f}  "
              f"IQR [{s['q25']:.3f}, {s['q75']:.3f}]   P(kappa>1) = {100*p:.0f}%")
        return s

    print(f"\n{label}")
    print(f"  mean actuator reach   delta = {np.mean(d):.4f} rho-units")
    print(f"  mean NS displacement  Delta = {np.mean(D):.4f} rho-units")
    out["all"] = stats(np.ones(len(rows), bool), "all sampled states")
    out["consulted"] = stats(con, "states agents SEE")
    out["mean_delta"] = float(np.mean(d))
    out["mean_Delta"] = float(np.mean(D))
    return out


def verdict(res):
    print("\n" + "=" * 78)
    print("VERDICT -- predicted recoverability, before any training")
    print("=" * 78)
    for r in res:
        c = r.get("consulted") or r.get("all") or {}
        if not c:
            print(f"  {r['label']:<28} inconclusive")
            continue
        p, m = c["p_kappa_gt_1"], c["median"]
        if m >= 1.5:
            v = "RECOVERABLE -- actuators outrun the disturbance"
        elif m >= 1.0:
            v = "MARGINAL -- information helps only sometimes"
        else:
            v = "NOT RECOVERABLE -- disturbance exceeds actuator reach"
        print(f"  {r['label']:<28} kappa={m:6.3f}  P(k>1)={100*p:3.0f}%   {v}")
    print("\n  kappa < 1 means no action, however well informed, restores the")
    print("  binding constraint. The oracle arm then cannot beat blind -- which")
    print("  is what was measured three times, at roughly 16 h per run.")
    print("  Use this to screen an NS design in minutes instead.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=None)
    ap.add_argument("--n_chronics", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=5,
                    help="sample a state every N steps (default: 5)")
    ap.add_argument("--safe_max_rho", type=float, default=0.9,
                    help="the heuristic's consultation threshold; only states "
                         "above it become training frames (default: 0.9)")
    ap.add_argument("--field_amplitudes", type=float, nargs="+",
                    default=[0.25, 0.10, 0.05])
    ap.add_argument("--screen", default="topology_screen.json")
    ap.add_argument("--out", default="authority.json")
    args = ap.parse_args()

    import grid2op
    env_name = args.env or os.path.join(grid2op.get_current_local_dir(),
                                        "l2rpn_idf_2023")
    print("=" * 78)
    print("ACTUATOR AUTHORITY RATIO")
    print("=" * 78)
    print(f"  env: {env_name}")
    env = make_env(env_name)
    print(f"  n_line={env.n_line}  n_storage={env.n_storage}  "
          f"n_renewable={int(np.sum(env.gen_renewable))}")

    res = []
    for A in args.field_amplitudes:
        res.append(run(env, "field", A, args.n_chronics, args.steps,
                       args.every, args.safe_max_rho, f"field amplitude={A}"))

    if os.path.exists(args.screen):
        with open(args.screen, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfgs = [(c["sub_id"], c["config"]) for c in d.get("all_candidates", [])
                if not c.get("illegal") and not c.get("exceptions")
                and c.get("survival_drop", 1) <= 0.02
                and c.get("sensitivity_change", 0) >= 0.05]
        if cfgs:
            res.append(run(env, "reconfig", cfgs, args.n_chronics, args.steps,
                           args.every, args.safe_max_rho,
                           f"reconfig standard ({len(cfgs)} cfgs)"))
        else:
            print(f"\n  no standard-tier configs in {args.screen}")
    else:
        print(f"\n  {args.screen} not found -- skipping the reconfiguration NS")

    verdict(res)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(env=env_name, results=res), f, indent=2)
    print(f"\n  written to {os.path.abspath(args.out)}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
