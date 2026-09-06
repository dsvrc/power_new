"""
ns_cpu.py -- size the collection pool to the machine instead of hardcoding 10.

WHERE THE PARALLELISM IS
------------------------
BenchMARL builds `ParallelEnv(on_policy_n_envs_per_worker, env_func)` when
`parallel_collection` is true (experiment.py:490-493), so that setting IS the
number of OS processes stepping grid2op.  main.py pinned it at 10, which on a
~48-core box is the observed ~21% utilisation.

The policy runs in the MAIN process (the collector steps the batched env and
infers centrally); the workers only run grid2op + LightSim.  So worker processes
do no torch work, and the right split is:

    * one env process per core        -- parallelism ACROSS environments
    * one BLAS/OMP thread per process -- never within, or 48 workers x 48 OMP
      threads oversubscribe the box into thrashing
    * torch threads only in the main process, for the training phase

MEMORY IS USUALLY THE REAL CAP, NOT CORES
-----------------------------------------
PZMultiAgentEnv forces `MultifolderWithCache` whenever `regex_filter_chronics`
is set, so every worker holds ALL matching chronics in RAM.  For this task that
is 64 February chronics, each with 12 forecast horizons -- on the order of
GBs per worker.  Multiplying workers multiplies that, and the box will OOM long
before it runs out of cores.

Passing `chronics_class=Multifolder` disables the cache: one chronic resident at
a time, roughly a tenth of the memory, at the cost of a re-read per episode
reset.  Episodes here take minutes, so that read is free.  Same data, same
filter, same sampling -- `diagnose_ns.py` has been running this way throughout.

Use --no_chronics_cache when scaling past a handful of workers.
"""

import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# Rough per-worker resident cost of a grid2op l2rpn_idf_2023 env.
GB_PER_ENV_CACHED = 3.0     # MultifolderWithCache: all filtered chronics + forecasts
GB_PER_ENV_NOCACHE = 0.45   # plain Multifolder: one chronic at a time


def configure_threads(n=1):
    """Pin BLAS/OMP to `n` threads per process.

    MUST run before numpy/torch are imported -- these are read at library load.
    Uses setdefault, so an explicit OMP_NUM_THREADS in the environment wins.
    """
    already = [m for m in ("torch", "numpy") if m in sys.modules]
    for v in _THREAD_VARS:
        os.environ.setdefault(v, str(n))
    if already:
        print(f"[cpu] WARNING: {already} already imported before configure_threads(); "
              f"thread limits may not apply. Call it at the top of main.py.")
    return {v: os.environ[v] for v in _THREAD_VARS}


def usable_cores():
    """Cores this process may actually run on -- affinity, cgroup and SLURM aware.

    os.cpu_count() reports the machine, not the allocation. Under taskset, a
    container, or a scheduler, it overcounts and the pool oversubscribes.
    """
    cands = []
    try:
        cands.append(len(os.sched_getaffinity(0)))       # Linux: taskset/cpuset
    except (AttributeError, OSError):
        pass
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var, "")
        if v.isdigit():
            cands.append(int(v))
    try:                                                  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            cands.append(max(1, int(int(quota) / int(period))))
    except Exception:
        pass
    try:                                                  # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            cands.append(max(1, q // p))
    except Exception:
        pass
    cands.append(os.cpu_count() or 1)
    return max(1, min(cands))


def available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return None


def _largest_divisor_leq(n, k):
    """torchrl splits frames_per_batch across envs; a divisor avoids remainders."""
    for d in range(int(k), 0, -1):
        if n % d == 0:
            return d
    return 1


def plan_workers(frames_per_batch, cpu_frac=0.9, reserve=2, requested=None,
                 gb_per_env=None, chronics_cached=True, mem_frac=0.8,
                 hard_cap=None, verbose=True):
    """Choose the number of collection processes.

    Returns (n_envs, info). `requested` overrides the core-based target but is
    still capped by memory and snapped to a divisor of frames_per_batch.
    """
    cores = usable_cores()
    if gb_per_env is None:
        gb_per_env = GB_PER_ENV_CACHED if chronics_cached else GB_PER_ENV_NOCACHE

    by_cores = max(1, int(cores * cpu_frac) - reserve)
    target = int(requested) if requested else by_cores

    avail = available_gb()
    by_mem = None
    if avail is not None and gb_per_env > 0:
        by_mem = max(1, int((avail * mem_frac) / gb_per_env))
        target = min(target, by_mem)

    if hard_cap:
        target = min(target, int(hard_cap))
    target = max(1, min(target, frames_per_batch))

    n_envs = _largest_divisor_leq(frames_per_batch, target)

    info = dict(cores=cores, cpu_frac=cpu_frac, reserve=reserve,
                by_cores=by_cores, requested=requested,
                available_gb=avail, gb_per_env=gb_per_env,
                chronics_cached=chronics_cached, by_mem=by_mem,
                target=target, n_envs=n_envs,
                frames_per_env=frames_per_batch // n_envs,
                binding=("request" if requested else
                         "memory" if (by_mem is not None and by_mem <= by_cores)
                         else "cores"))

    if verbose:
        print("[cpu] " + "-" * 66)
        print(f"[cpu] usable cores        : {cores}"
              f"   (affinity/cgroup/SLURM aware)")
        print(f"[cpu] available memory    : "
              f"{'unknown' if avail is None else f'{avail:.1f} GB'}")
        print(f"[cpu] chronics cache      : "
              f"{'ON  (~%.1f GB/env)' % gb_per_env if chronics_cached else 'OFF (~%.2f GB/env)' % gb_per_env}")
        print(f"[cpu] cap by cores        : {by_cores}")
        print(f"[cpu] cap by memory       : {by_mem if by_mem is not None else 'n/a'}")
        print(f"[cpu] -> env processes    : {n_envs}"
              f"   ({info['frames_per_env']} frames each, binding cap: {info['binding']})")
        if n_envs < target:
            print(f"[cpu]    snapped down from {target} to divide "
                  f"frames_per_batch={frames_per_batch} evenly")
        if chronics_cached and by_mem is not None and by_mem <= by_cores:
            nc = max(1, int((avail * mem_frac) / GB_PER_ENV_NOCACHE))
            print(f"[cpu] NOTE: memory is the binding cap. --no_chronics_cache would "
                  f"allow ~{min(nc, by_cores)} workers instead of {n_envs}.")
        print("[cpu] " + "-" * 66)
    return n_envs, info


def train_threads(cores, cap=16):
    """Threads for the training phase, in the MAIN process only.

    Safe to exceed 1: collection workers run grid2op, never torch, so they do
    not contend. Collection dominates wall-clock anyway (~353 s/iter against
    ~tens of seconds of optimisation), so this is a second-order knob.
    """
    return max(1, min(cap, cores // 2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Show the CPU plan for this machine.")
    ap.add_argument("--frames_per_batch", type=int, default=6000)
    ap.add_argument("--cpu_frac", type=float, default=0.9)
    ap.add_argument("--no_chronics_cache", action="store_true")
    ap.add_argument("--n_envs", type=int, default=None)
    a = ap.parse_args()
    n, info = plan_workers(a.frames_per_batch, cpu_frac=a.cpu_frac,
                           requested=a.n_envs,
                           chronics_cached=not a.no_chronics_cache)
    print(f"\ntorch threads for training: {train_threads(info['cores'])}")
