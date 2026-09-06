#!/usr/bin/env python
"""
ns_verdict.py -- read NSDiagnosticsCallback CSVs and decide, early, whether
exogenous non-stationarity is breaking MAPPO.

The point is to avoid spending 24h per condition.  Run a SHORT training under
two opponent presets, then compare.  The critic's explained variance separates
long before returns do, because ev measures whether the learning problem is
well-posed at all, while return measures whether the policy has had time to
exploit it.

    # ~25 iterations each, three seeds, two conditions
    python main.py --n_frames 150_000 --opponent off    --seeds 0 1 2 --save_experiment
    python main.py --n_frames 150_000 --opponent hidden --seeds 0 1 2 --save_experiment
    python ns_verdict.py saved_models/ns_diagnostics --baseline off --treatment hidden

Decision rules (applied to the last third of iterations, averaged over seeds):

  BROKEN     critic_ev < 0.10                     value function cannot fit returns
  DEGRADED   critic_ev drop vs baseline > 0.15    NS specifically hurts the critic
  COLLAPSED  policy_scale fell > 50% while return did not improve
  STALLED    return slope <= 0 over the window
  UNSTABLE   |drift_loss_objective| growing       30 inner epochs too many

Anything flagged BROKEN or DEGRADED answers the question without a full run.
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def col(rows, name):
    return [r[name] for r in rows
            if name in r and isinstance(r[name], float) and not math.isnan(r[name])]


def slope(ys):
    n = len(ys)
    if n < 3:
        return float("nan")
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def tail(xs, frac=0.34):
    if not xs:
        return []
    k = max(1, int(len(xs) * frac))
    return xs[-k:]


def summarise(paths, label):
    """Aggregate one condition across seeds."""
    per_seed = []
    for p in paths:
        rows = read_csv(p)
        if len(rows) < 3:
            print(f"  ! {os.path.basename(p)}: only {len(rows)} iterations, skipping")
            continue
        ev = col(rows, "critic_ev_mean")
        ret = col(rows, "mean_return")
        scale = col(rows, "policy_scale_mean")
        eplen = col(rows, "approx_ep_len")
        deg = col(rows, "frac_degraded")
        drift = [abs(x) for x in col(rows, "drift_loss_objective_mean")]
        ess = col(rows, "ESS_mean")

        per_seed.append(dict(
            file=os.path.basename(p),
            n_iters=len(rows),
            ev_tail=mean(tail(ev)),
            ev_first=mean(ev[:max(1, len(ev) // 5)]),
            ret_tail=mean(tail(ret)),
            ret_slope=slope(ret),
            scale_first=mean(scale[:max(1, len(scale) // 5)]) if scale else float("nan"),
            scale_tail=mean(tail(scale)) if scale else float("nan"),
            eplen_tail=mean(tail(eplen)) if eplen else float("nan"),
            eplen_slope=slope(eplen) if eplen else float("nan"),
            deg=mean(deg) if deg else float("nan"),
            drift_tail=mean(tail(drift)) if drift else float("nan"),
            ess_tail=mean(tail(ess)) if ess else float("nan"),
        ))

    if not per_seed:
        return None

    agg = {"label": label, "n_seeds": len(per_seed), "seeds": per_seed}
    for k in ("n_iters", "ev_tail", "ev_first", "ret_tail", "ret_slope",
              "scale_first", "scale_tail", "eplen_tail", "eplen_slope",
              "deg", "drift_tail", "ess_tail"):
        vals = [s[k] for s in per_seed if not (isinstance(s[k], float) and math.isnan(s[k]))]
        agg[k] = mean(vals) if vals else float("nan")
    return agg


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:.{p}f}"


def verdict(treat, base):
    flags = []

    ev = treat["ev_tail"]
    if not math.isnan(ev) and ev < 0.10:
        flags.append(("BROKEN",
                      f"critic_ev={fmt(ev)} < 0.10 -- the value function cannot "
                      f"explain returns, so advantages are noise"))

    if base is not None:
        d = base["ev_tail"] - treat["ev_tail"]
        if not math.isnan(d) and d > 0.15:
            flags.append(("DEGRADED",
                          f"critic_ev drops {fmt(d)} vs {base['label']} "
                          f"({fmt(base['ev_tail'])} -> {fmt(treat['ev_tail'])}) -- "
                          f"attributable to the opponent, not to the task"))

    sf, st = treat["scale_first"], treat["scale_tail"]
    if not math.isnan(sf) and not math.isnan(st) and sf > 0:
        drop = 1.0 - st / sf
        if drop > 0.5 and not (treat["ret_slope"] > 0):
            flags.append(("COLLAPSED",
                          f"policy sigma fell {100*drop:.0f}% with no return "
                          f"improvement -- converged onto the regime-marginal "
                          f"compromise policy"))

    if not math.isnan(treat["ret_slope"]) and treat["ret_slope"] <= 0:
        flags.append(("STALLED",
                      f"return slope {fmt(treat['ret_slope'], 4)} <= 0 over the run"))

    if base is not None:
        for k, name in (("ret_tail", "return"), ("eplen_tail", "episode length")):
            a, b = treat[k], base[k]
            if not math.isnan(a) and not math.isnan(b) and b != 0:
                r = a / b
                if r < 0.85:
                    flags.append(("IMPACT",
                                  f"{name} is {100*(1-r):.0f}% below {base['label']} "
                                  f"({fmt(a,1)} vs {fmt(b,1)})"))

    if not math.isnan(treat["ess_tail"]) and treat["ess_tail"] < 0.5:
        flags.append(("UNSTABLE",
                      f"ESS={fmt(treat['ess_tail'])} -- the 30 inner epochs are "
                      f"training on data the current joint policy no longer produces"))

    return flags


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_dir", help="directory of NSDiagnosticsCallback CSVs")
    ap.add_argument("--baseline", default="off",
                    help="opponent preset treated as control (default: off)")
    ap.add_argument("--treatment", default="hidden",
                    help="opponent preset under test (default: hidden)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not paths:
        print(f"No CSVs in {args.csv_dir}")
        return

    groups = defaultdict(list)
    for p in paths:
        m = re.match(r".*?_(\w+?)_seed\d+\.csv$", os.path.basename(p))
        groups[m.group(1) if m else "unknown"].append(p)

    print("=" * 78)
    print("NS EARLY VERDICT")
    print("=" * 78)
    print(f"  {args.csv_dir}")
    for k, v in sorted(groups.items()):
        print(f"    {k:<12} {len(v)} seed(s)")

    summaries = {}
    for k, v in sorted(groups.items()):
        print(f"\n--- {k} ---")
        s = summarise(v, k)
        if s:
            summaries[k] = s

    if not summaries:
        print("\nNothing summarisable yet -- need >=3 iterations per run.")
        return

    hdr = (f"\n  {'condition':<12}{'iters':>6}{'critic_ev':>11}{'return':>10}"
           f"{'ret slope':>11}{'ep len':>9}{'sigma':>9}{'degraded':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for k, s in sorted(summaries.items()):
        print(f"  {k:<12}{s['n_iters']:>6.0f}{fmt(s['ev_tail']):>11}"
              f"{fmt(s['ret_tail'],1):>10}{fmt(s['ret_slope'],4):>11}"
              f"{fmt(s['eplen_tail'],1):>9}{fmt(s['scale_tail']):>9}"
              f"{fmt(s['deg']):>10}")
    print("\n  critic_ev / return / ep len / sigma are means over the last third")
    print("  of iterations, averaged across seeds.")

    base = summaries.get(args.baseline)
    treat = summaries.get(args.treatment)

    print("\n" + "=" * 78)
    if treat is None:
        print(f"VERDICT: no runs found for treatment {args.treatment!r}.")
        print("=" * 78)
        return
    if base is None:
        print(f"note: no baseline {args.baseline!r} found -- absolute checks only")

    flags = verdict(treat, base)
    print(f"VERDICT for {args.treatment!r}"
          + (f" vs {args.baseline!r}" if base else ""))
    print("=" * 78)
    if not flags:
        print("\n  No failure signature detected.")
        print("  MAPPO is coping with this level of exogenous NS. Escalate the")
        print("  preset (hidden -> brutal) or extend the run before concluding")
        print("  that it is robust.")
    else:
        for tag, msg in flags:
            print(f"\n  [{tag}] {msg}")
        names = {t for t, _ in flags}
        print("\n  " + "-" * 74)
        if {"BROKEN", "DEGRADED"} & names:
            print("  Conclusive: exogenous NS is damaging the learning problem")
            print("  itself, not merely the achieved score. A full-length run will")
            print("  not recover this -- the critic is missing information that is")
            print("  absent from its input, which more gradient steps cannot supply.")
        elif "IMPACT" in names:
            print("  Outcome effect present but the critic still fits. Worth a")
            print("  full-length run to see whether it is a slowdown or a ceiling.")
        else:
            print("  Weak signature. Extend the run or escalate the preset.")
    print()


if __name__ == "__main__":
    main()
