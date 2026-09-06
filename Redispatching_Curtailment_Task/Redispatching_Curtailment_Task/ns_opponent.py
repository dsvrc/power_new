"""
ns_opponent.py -- opponent presets for the redispatching task.

WHY THESE PRESETS EXIST
-----------------------
The dataset default (l2rpn_idf_2023/config.py) runs GeometricOpponentMultiArea
with attack_every_xxx_hour=32 and average_attack_duration_hour=2 over 22 lines
split into 3 areas.  Measured with diagnose_ns.py that yields ~11% of grid steps
under attack -- real, but it changes NEITHER survival (ON/OFF ratio 1.00 under a
random policy) NOR return variance.  Episodes under untrained policies end in
27-102 agent steps, while attacks start roughly once per 128 grid steps, so an
episode barely contains one attack window.  The signal sits under the noise.

WHICH ATTACK HURTS MAPPO MOST
-----------------------------
Not simply "more attacks".  The binding weakness measured in the diagnostic is
OBSERVABILITY: each attack is visible to a mean of 1.48 of the 11 zone agents,
because a zone agent only observes line_status over its own line_large_idx.

Cross-referencing the 22 attackable lines against zones_definitions.json gives
each line's visibility (how many of the 11 zone agents can see it at all):

    4 zones : 117, 131, 136, 141
    2 zones : 62, 68, 88, 93
    1 zone  : 33, 37, 43, 61, 81, 106, 110, 121, 125, 126, 154, 160, 162, 180

The 14 one-zone lines are the maximally-hidden set.  Attacking only those means
10 of 11 zone agents receive NO observable signal for any attack, while their
power flows still move -- a pure hidden regime switch.  A memoryless MLP policy
cannot adapt to that; it can only learn the regime-marginal compromise.  That is
an information limit, not an optimisation one, which is what makes it a clean
result rather than a tuning artefact.

So the "hidden" preset = maximal unobservability x high intensity.  That is the
configuration expected to break MAPPO, and the one to compare against "off".

BUDGET
------
The dataset budget (init 1000, +0.51/ts) supports ~2000 attack-timesteps per
week-long chronic.  The aggressive presets would exhaust that and silently
revert to a lower attack rate, so they raise the budget until it is
non-binding.  The geometric timer then remains the single interpretable knob.
"""

# ---------------------------------------------------------------------------
# Line names, verbatim from l2rpn_idf_2023/config.py.
# grid2op names are "<sub_a>_<sub_b>_<line_id>", so the trailing integer is the
# line index used by zones_definitions.json / obs.line_status.
# ---------------------------------------------------------------------------
AREA_1 = ["26_31_106", "21_22_93", "17_18_88", "4_10_162", "12_14_68", "29_37_117"]
AREA_2 = ["62_58_180", "62_63_160", "48_50_136", "48_53_141", "41_48_131",
          "39_41_121", "43_44_125", "44_45_126", "34_35_110", "54_58_154"]
AREA_3 = ["74_117_81", "93_95_43", "88_91_33", "91_92_37", "99_105_62", "102_104_61"]

ALL_LINES = [AREA_1, AREA_2, AREA_3]

# The 14 lines each visible to exactly ONE zone agent, kept in their original
# areas so GeometricOpponentMultiArea still has three independent adversaries.
# Derivation is in the module docstring; verify_hidden_set() re-checks it.
HIDDEN_1 = ["26_31_106", "4_10_162"]                       # lines 106, 162
HIDDEN_2 = ["34_35_110", "39_41_121", "43_44_125",         # 110, 121, 125
            "44_45_126", "54_58_154", "62_63_160",         # 126, 154, 160
            "62_58_180"]                                   # 180
HIDDEN_3 = ["88_91_33", "91_92_37", "93_95_43",            # 33, 37, 43
            "102_104_61", "74_117_81"]                     # 61, 81

HIDDEN_LINES = [HIDDEN_1, HIDDEN_2, HIDDEN_3]

BIG_BUDGET = dict(opponent_init_budget=1_000_000.0, opponent_budget_per_ts=100.0)


def _check_invariants(every_hour, avg_duration_hour, min_duration_hour,
                      max_duration_steps):
    """A geometric attack process is only coherent if the mean gap between
    attack STARTS exceeds the mean attack DURATION -- otherwise an attack is
    still running when the next is due and the process is ill-posed.  grid2op
    rejects such a configuration at env construction time.

    My first version of these presets violated this (every 3h, lasting 4h) and
    every aggressive preset failed to build.
    """
    if not (min_duration_hour <= avg_duration_hour):
        raise ValueError(
            f"minimum_attack_duration_hour ({min_duration_hour}) must be <= "
            f"average_attack_duration_hour ({avg_duration_hour})")
    if not (avg_duration_hour < every_hour):
        raise ValueError(
            f"average_attack_duration_hour ({avg_duration_hour}) must be < "
            f"attack_every_xxx_hour ({every_hour}); attacks would otherwise "
            f"overlap themselves. Raise the interval or shorten the duration.")
    if max_duration_steps < avg_duration_hour * 12:
        raise ValueError(
            f"opponent_attack_duration ({max_duration_steps} steps = "
            f"{max_duration_steps/12:.1f}h) caps the mean duration "
            f"({avg_duration_hour}h); raise the cap.")


def _geometric(lines, every_hour, avg_duration_hour, max_duration_steps,
               min_duration_hour=1, pmax_pmin_ratio=4):
    from grid2op.Action import PowerlineSetAction
    from grid2op.Opponent import GeometricOpponentMultiArea, BaseActionBudget

    _check_invariants(every_hour, avg_duration_hour, min_duration_hour,
                      max_duration_steps)

    cfg = dict(
        opponent_class=GeometricOpponentMultiArea,
        opponent_action_class=PowerlineSetAction,
        opponent_budget_class=BaseActionBudget,
        opponent_attack_cooldown=0,
        opponent_attack_duration=max_duration_steps,
        kwargs_opponent=dict(
            lines_attacked=lines,
            attack_every_xxx_hour=every_hour,
            average_attack_duration_hour=avg_duration_hour,
            minimum_attack_duration_hour=min_duration_hour,
            pmax_pmin_ratio=pmax_pmin_ratio,
        ),
    )
    cfg.update(BIG_BUDGET)
    return cfg


def opponent_off():
    """grid2op's documented recipe for a genuinely inert opponent."""
    from grid2op.Action import DontAct
    from grid2op.Opponent import BaseOpponent, NeverAttackBudget

    return dict(
        opponent_attack_cooldown=999_999,
        opponent_attack_duration=0,
        opponent_budget_per_ts=0.0,
        opponent_init_budget=0.0,
        opponent_action_class=DontAct,
        opponent_class=BaseOpponent,
        opponent_budget_class=NeverAttackBudget,
    )


def get_preset(name):
    """Return env_g2op_config kwargs for a named opponent preset."""
    name = (name or "default").lower()

    if name == "default":
        return {}                       # dataset config.py, untouched
    if name == "off":
        return opponent_off()

    # Per-area duty cycle d ~ avg_duration / interval; with 3 independent areas
    # the share of steps with >=1 line attacked is ~ 1 - (1-d)^3.

    if name == "frequent":
        # Pure intensity: same 22 lines, d=0.25. Companion to 'hidden'.
        return _geometric(ALL_LINES, every_hour=12, avg_duration_hour=3,
                          max_duration_steps=96)

    if name == "hidden":
        # RECOMMENDED, and MEASURED (diagnose_ns.py, 12 episodes, do-nothing):
        #   51.6% of grid steps under attack   (vs 11.3% for 'default')
        #   60.1% of agent decision points degraded (vs 27.9% opponent-off)
        #   survival 0.47x vs opponent-off, shortened in 10 of 12 episodes
        #   mean 1.000 of 11 zone agents can see any given attack -- exactly
        #     one, by construction, so 10 of 11 are blind to every attack
        #   Zone1, Zone3 and Zone6 observed ZERO attacks across the whole run
        return _geometric(HIDDEN_LINES, every_hour=12, avg_duration_hour=3,
                          max_duration_steps=96)

    if name == "brutal":
        # Upper bound, d=0.67. 'hidden' already halves survival, so this is
        # unlikely to be needed and risks leaving no learning signal at all.
        return _geometric(HIDDEN_LINES, every_hour=6, avg_duration_hour=4,
                          max_duration_steps=96)

    raise ValueError(f"unknown opponent preset {name!r}; "
                     f"choose from {sorted(PRESETS)}")


PRESETS = {"off", "default", "frequent", "hidden", "brutal"}


# ---------------------------------------------------------------------------
def line_id(name):
    """'62_58_180' -> 180."""
    return int(name.rsplit("_", 1)[1])


def verify_hidden_set(zones_json_path=None, verbose=True):
    """Re-derive the one-zone-visible set from zones_definitions.json and check
    it matches HIDDEN_LINES.  Run this if the zone partition ever changes."""
    import json
    import os

    if zones_json_path is None:
        import benchmarl.environments.G2OpPowerGrid.utils as g2u
        zones_json_path = os.path.join(os.path.dirname(g2u.__file__),
                                       "zones_definitions.json")

    Z = json.load(open(zones_json_path, "r", encoding="utf-8"))
    zones = {z: set(int(x) for x in Z[z]["line_large_idx"]) for z in Z}

    flat = [n for area in ALL_LINES for n in area]
    vis = {n: sum(1 for s in zones.values() if line_id(n) in s) for n in flat}
    derived = {n for n, v in vis.items() if v == 1}
    declared = {n for area in HIDDEN_LINES for n in area}

    ok = derived == declared
    if verbose:
        print(f"zones file: {zones_json_path}")
        print(f"{'line':<14}{'id':>5}{'zones that see it':>20}")
        for n in sorted(flat, key=lambda n: (vis[n], line_id(n))):
            mark = " <- hidden" if vis[n] == 1 else ""
            print(f"  {n:<12}{line_id(n):>5}{vis[n]:>18}{mark}")
        print(f"\nderived one-zone set ({len(derived)}) == declared "
              f"HIDDEN_LINES ({len(declared)}): {ok}")
        if not ok:
            print(f"  only in derived : {sorted(derived - declared)}")
            print(f"  only in declared: {sorted(declared - derived)}")
    return ok


def smoke_test(env_name=None, steps=5):
    """Actually construct a grid2op env for every preset and report the real
    exception if one fails.  Settles construction errors in ~1 minute instead of
    discovering them halfway through a diagnostic run."""
    import os
    import traceback

    import grid2op
    from grid2op.Action import PlayableAction
    from grid2op.Chronics import Multifolder

    try:
        from lightsim2grid import LightSimBackend as backend_cls
    except ImportError:
        from grid2op.Backend import PandaPowerBackend as backend_cls

    if env_name is None:
        env_name = os.path.join(grid2op.get_current_local_dir(), "l2rpn_idf_2023")

    print(f"env: {env_name}\n")
    results = {}
    for preset in sorted(PRESETS):
        print(f"--- {preset} ---")
        try:
            kw = get_preset(preset)
        except Exception as exc:
            print(f"  PRESET INVALID: {type(exc).__name__}: {exc}\n")
            results[preset] = f"preset error: {exc}"
            continue

        env = None
        try:
            env = grid2op.make(env_name, action_class=PlayableAction,
                               backend=backend_cls(), chronics_class=Multifolder,
                               **kw)
            env.reset()
            for _ in range(steps):
                _, _, done, _ = env.step(env.action_space({}))
                if done:
                    env.reset()
            opp = type(getattr(env, "_opponent", None)).__name__
            print(f"  OK   opponent={opp}")
            results[preset] = f"ok ({opp})"
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            results[preset] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
        print()

    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:<10} {v}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Opponent presets: verify and smoke-test.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="build a real grid2op env for every preset and report "
                         "the actual exception on failure")
    ap.add_argument("--env", default=None, help="path to the grid2op env folder")
    a = ap.parse_args()

    verify_hidden_set()
    if a.smoke_test:
        print()
        smoke_test(a.env)
