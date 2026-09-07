
import os
import yaml
import argparse
import gc
from benchmarl.algorithms import MappoConfig, MasacConfig
from benchmarl.environments import G2OpPowerGridTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
import grid2op
from utils import ROOT_DIR, G2OP_ENV_DIR, IS_LINUX
from BMMAAgent import BMMAAgent
from evaluate import evaluate
from ns_opponent import get_preset, PRESETS
from ns_callback import NSDiagnosticsCallback


def _require_benchmarl_features(*needed):
    """Fail loudly if `benchmarl` resolves to a stale tree.

    `python -m ...AmbientField` run from inside BenchMARL/ works because the cwd
    shadows the installed package -- while main.py, run from this directory,
    imports whatever is actually installed. If those are different trees you
    train the wrong environment and nothing looks wrong.
    """
    import benchmarl
    from benchmarl.environments.G2OpPowerGrid import common as _g2c
    have = set(getattr(_g2c, "G2OP_FEATURES", ()))
    missing = set(needed) - have
    if missing:
        raise SystemExit(
            "\n" + "=" * 74 +
            f"\nSTALE BENCHMARL INSTALL -- missing {sorted(missing)}\n" + "=" * 74 +
            f"\n  benchmarl imported from : {benchmarl.__file__}"
            f"\n  G2OpPowerGrid/common.py : {_g2c.__file__}"
            f"\n  this script             : {os.path.abspath(__file__)}"
            "\n\n  That common.py is not the one you edited. Reinstall from the tree"
            "\n  you are actually working in:"
            "\n\n      cd ../BenchMARL && pip install . --constraint ../requirements.txt"
            "\n\n  (or `pip install -e .` from there for an editable install, so future"
            "\n  edits take effect without reinstalling).\n")
    return benchmarl


_require_benchmarl_features("ambient_field", "state_fix")
from benchmarl.environments.G2OpPowerGrid.AmbientField import get_field_preset

def cli():
    parser = argparse.ArgumentParser(description="Train some agents.")
    parser.add_argument('--n_frames', type=int, default=6_000,
                        help="Total number of frames to collect for training. (default: 6_000)")
    parser.add_argument('--lr', type=float, default=3e-5,
                        help="Learning rate. (default: 3e-5)")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Gamma. (default: 0.99)")
    parser.add_argument('--frames_per_batch', type=int, default=6000,
                        help="Frames per batch. (default: 6000)")
    parser.add_argument('--MAPPO_n_episode', type=int, default=30,
                        help="Number of episodes when training with MAPPO. (default: 15)")
    parser.add_argument('--MASAC_n_optimizer_steps', type=int, default=1000,
                        help="""Number of times MASAC_train_batch_size will be sampled from 
                        the buffer and trained over when training with MASAC. (default: 1000)""")
    parser.add_argument('--MASAC_train_batch_size', type=int, default=256,
                        help="""Number of frames used for each optimizer's step when training 
                                with MASAC. (default: 128)""")
    parser.add_argument('--seeds', type=int, default=[0, 1, 2], nargs='+',
                        help="Random seeds (default: [0, 1, 2])")
    parser.add_argument('--alg', type=str, default="MAPPO", choices=["MAPPO", "MASAC"],
                        help="Whether to train with MAPPO or MASAC. (default: MAPPO)")
    parser.add_argument('--save_experiment', action='store_true', 
                        help="""Whether or not to save the experiment. Note that a MASAC checkpoint
                                can be heavy because the buffer is also saved in it. (default: False)""")
    parser.add_argument('--evaluate_agents', action='store_true',
                        help="Whether or not to evaluate the trained agents. (default: False)")
    parser.add_argument('--opponent', type=str, default="default",
                        choices=sorted(PRESETS),
                        help="""Exogenous non-stationarity level. 'default' uses the
                                dataset's own opponent (measured: ~11%% of grid steps
                                attacked, no effect on survival). 'off' disables it.
                                'hidden' attacks only the 14 lines visible to a single
                                zone agent, at ~4x the duty cycle -- the configuration
                                expected to break MAPPO. 'frequent' is the same
                                intensity over all 22 lines, so hidden-vs-frequent
                                isolates observability from severity. Verify with
                                'python ns_opponent.py --smoke-test'. (default: default)""")
    parser.add_argument('--field', type=str, default="off",
                        help="""Recoverable exogenous NS: an advecting weather front
                                modulating line thermal ratings (Dynamic Line Rating).
                                Mean-zero in space and time, so average capacity is
                                UNCHANGED and the undisturbed baseline stays reachable
                                -- unlike --opponent, which removes capacity. Choices:
                                off, mild, standard, strong, fast. (default: off)""")
    parser.add_argument('--field_oracle', type=str, default="none",
                        choices=["none", "local", "full"],
                        help="""Append the true field to every agent's observation.
                                'full' is the CEILING arm: gap_B = baseline - oracle is
                                irreducible, gap_A = oracle - blind is what any method
                                could win. Run it before building a method.
                                (default: none)""")
    parser.add_argument('--field_controller', type=str, default="none",
                        choices=["none", "oracle", "consensus", "local"],
                        help="""Anticipatory curtailment controller, applied in the
                                env wrapper so the host RL is untouched.
                                'oracle' uses the true field -- the CEILING TEST: if
                                this does not beat blind, the field is not actionable
                                and no estimator will rescue it.
                                'consensus' estimates the field by pooling every
                                agent's line-rating measurements.
                                'local' estimates it from each agent's own zone only
                                -- the ablation that says whether pooling is needed.
                                (default: none)""")
    parser.add_argument('--ctrl_gain', type=float, default=0.5,
                        help="Curtailment correction per unit anticipated rho excess. "
                             "Negative flips the sign, which identifies the coupling "
                             "direction if the positive gain hurts. (default: 0.5)")
    parser.add_argument('--ctrl_horizon', type=int, default=12,
                        help="Steps ahead to anticipate; 12 = 1 h. (default: 12)")
    parser.add_argument('--ctrl_target', type=float, default=0.90,
                        help="rho above which anticipated loading counts as excess. "
                             "(default: 0.90, matching safe_max_rho)")
    parser.add_argument('--ns_csv', type=str, default=None,
                        help="""Directory for per-iteration NS diagnostics CSVs, one
                                per seed. Lets you judge whether MAPPO is failing after
                                ~20 iterations instead of a full run. Set to '' to
                                disable. (default: <save_folder>/ns_diagnostics)""")
    return parser.parse_args()

def train_algo(task, algorithm_config, model_config, critic_model_config, experiment_config, seed, evaluate_agent, callbacks=None):
        print("Creating experiment...")
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed,
            config=experiment_config,
            callbacks=callbacks if callbacks else None,
        )
        print("Starting training...")
        experiment.run()
        experiment.close()

        # Evaluation
        if evaluate_agent:
            print("Starting evaluation...")

            env = grid2op.make(os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023_test_new"))
            # Load the agent
            grid2op_agent = BMMAAgent(env.action_space, nn_kwargs={})
            grid2op_agent.load(experiment=experiment)
            print("Agent loaded.")

            # Evaluate the agent
            algo = algorithm_config.associated_class().__name__.upper()
            evaluate(grid2op_agent, agent_name=f"{algo}_{seed}", results_path_agents=os.path.join(ROOT_DIR, "agents_results"))

        return experiment


if __name__ == "__main__":

    if IS_LINUX:
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)

    args = cli()
    # Explicit attribute access rather than positional unpacking of
    # vars(args).values(), so adding a CLI flag cannot silently shift bindings.
    n_frames = args.n_frames
    lr = args.lr
    gamma = args.gamma
    frames_per_batch = args.frames_per_batch
    MAPPO_n_episode = args.MAPPO_n_episode
    MASAC_n_optimizer_steps = args.MASAC_n_optimizer_steps
    MASAC_train_batch_size = args.MASAC_train_batch_size
    seeds = args.seeds
    alg = args.alg
    save_experiment = args.save_experiment
    evaluate_agents = args.evaluate_agents

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml() # 

    # Loads from "benchmarl/conf/task/Grid2OpPowerGrid/my_power_grid.yaml"
    task = G2OpPowerGridTask.MY_POWER_GRID.get_from_yaml()
    task.config["env_name"] = os.path.join(G2OP_ENV_DIR, "l2rpn_idf_2023")

    if alg == "MAPPO":
        # Loads from "benchmarl/conf/algorithm/mappo.yaml"
        algorithm_config = MappoConfig.get_from_yaml()
    elif alg == "MASAC":
        # Loads from "benchmarl/conf/algorithm/masac.yaml"
        algorithm_config = MasacConfig.get_from_yaml()
    else:
        raise ValueError(f"Unknown algorithm: {alg}")
    
    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    with open("configs/expes_config.yaml", "r") as f:
        updates = yaml.safe_load(f)

    for config_type, config in zip(["experiment", "task", "algorithm", "model", "critic_model"],
                                    [experiment_config, task, algorithm_config, model_config, critic_model_config]):
        new_hps = updates[config_type]
        for hp in new_hps:
            if config_type == "task" and hp == "config": # In this case, we want to update the dict, not replace it
                config.config.update(new_hps[hp])
            else:
                setattr(config, hp, new_hps[hp])


    # Exogenous non-stationarity level. Must come after the expes_config loop so
    # it is not overwritten; env_g2op_config is a straight pass-through to
    # grid2op.make, and a YAML cannot carry the opponent classes.
    task.config["env_g2op_config"] = get_preset(args.opponent)
    print(f"Opponent preset: {args.opponent} -> "
          f"{sorted(task.config['env_g2op_config'].keys()) or 'dataset default'}")

    # Recoverable exogenous NS (advecting DLR field). Seeded per run so the
    # field realisation is reproducible and identical across arms.
    field_cfg = get_field_preset(
        args.field, oracle=args.field_oracle, controller=args.field_controller,
        ctrl_gain=args.ctrl_gain, ctrl_horizon=args.ctrl_horizon,
        ctrl_target=args.ctrl_target)
    task.config["ambient_field"] = field_cfg
    print(f"Ambient field: {args.field} (oracle={args.field_oracle}) -> "
          f"{'disabled' if field_cfg is None else field_cfg}")

    experiment_config.save_folder = os.path.join(ROOT_DIR, "saved_models")
    os.makedirs(experiment_config.save_folder, exist_ok=True)
    experiment_config.checkpoint_at_end = save_experiment # A MASAC checkpoint is 67G

    experiment_config.max_n_frames = n_frames

    experiment_config.parallel_collection = IS_LINUX

    if alg == "MAPPO":
        experiment_config.on_policy_n_envs_per_worker = 10 if IS_LINUX else 1
        experiment_config.on_policy_collected_frames_per_batch = frames_per_batch
        experiment_config.on_policy_minibatch_size = experiment_config.on_policy_collected_frames_per_batch
        experiment_config.on_policy_n_minibatch_iters = MAPPO_n_episode
    elif alg == "MASAC":
        experiment_config.off_policy_n_envs_per_worker = 10 if IS_LINUX else 1
        experiment_config.off_policy_collected_frames_per_batch = frames_per_batch
        experiment_config.off_policy_train_batch_size = MASAC_train_batch_size
        experiment_config.off_policy_n_optimizer_steps = MASAC_n_optimizer_steps
        experiment_config.off_policy_memory_size = 500_000
    else:
        raise ValueError(f"Unknown algorithm: {alg}, possible values are 'MAPPO' or 'MASAC'")



    experiment_config.lr = lr
    experiment_config.gamma = gamma

    ns_csv_dir = args.ns_csv
    if ns_csv_dir is None:
        ns_csv_dir = os.path.join(experiment_config.save_folder, "ns_diagnostics")

    for i, seed in enumerate(seeds):
        print(f"Running experiment {i + 1}/{len(seeds)}.")
        callbacks = None
        if ns_csv_dir:
            os.makedirs(ns_csv_dir, exist_ok=True)
            # Arm tag drives the CSV filename, which ns_verdict.py groups on.
            ftag = "off" if args.field == "off" else args.field
            if args.field_oracle != "none":
                ftag += f"-oracle-{args.field_oracle}"
            if args.field_controller != "none":
                ftag += f"-ctrl-{args.field_controller}"
            tag = f"{alg}_field-{ftag}_seed{seed}"
            callbacks = [NSDiagnosticsCallback(
                csv_path=os.path.join(ns_csv_dir, f"{tag}.csv"), run_tag=tag)]
        train_algo(task, algorithm_config, model_config, critic_model_config,
                   experiment_config, seed, evaluate_agents, callbacks=callbacks)
        gc.collect()