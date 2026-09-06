"""
ns_callback.py -- per-iteration health metrics written to CSV, so you can tell
whether MAPPO is failing after ~20 iterations instead of after 24 hours.

WHAT TO WATCH, AND WHY
----------------------
critic_ev  (explained variance of the value function)
    ev = 1 - Var(value_target - state_value) / Var(value_target)
    This is THE non-stationarity diagnostic.  MAPPO's critic is V(s), not
    Q(s,a): it never conditions on teammates' actions, and under the "hidden"
    opponent preset it also cannot see the attacked line for 10 of 11 zone
    agents.  When the returns are driven by things absent from s, no critic can
    fit them and ev collapses toward 0 or negative.  Advantages then become
    noise, and the policy gradient is estimating nothing.  ev separates between
    conditions within ~15-25 iterations, long before returns do.

policy_scale  (mean Gaussian sigma of the actor)
    Entropy proxy.  entropy_coef is 0.0 in mappo.yaml, so loss_entropy carries
    no signal -- sigma does.  Collapsing sigma with flat return means premature
    convergence onto the regime-marginal compromise policy: exactly the failure
    predicted for a memoryless policy under unobserved regime switching.

ESS  (effective sample size, if torchrl reports it)
    How off-policy the batch has become across the 30 inner epochs.  Low ESS
    means MAPPO_n_episode=30 with full-batch minibatches is training on data its
    own policy no longer generates -- and nothing corrects for the *teammates'*
    drift at all.

drift_loss_objective  (last inner epoch minus first)
    Same story, measured directly.

frac_degraded
    Share of collected frames where at least one line is out.  Ties the training
    curves back to the exogenous process.  Best-effort: needs line_status to be
    locatable inside the state vector; NaN if not.

Usage:  Experiment(..., callbacks=[NSDiagnosticsCallback("run.csv")])
        then compare runs with ns_verdict.py
"""

import csv
import os
import time

import numpy as np
import torch

from benchmarl.experiment.callback import Callback


# Keys whose within-iteration drift (last inner epoch - first) is informative.
_DRIFT_KEYS = ("loss_objective", "loss_critic", "ESS")


def _locate_attr(space, attr):
    """Candidate index range of `attr` inside a grid2op BoxGymnasiumObsSpace.

    Only a CANDIDATE: the first version of this trusted the answer and produced
    frac_degraded == 1.0 for every frame of every run, because the slice landed
    on a normalised integer attribute whose values are all < 0.5.  The caller
    must validate against real data via _validate_slice.
    """
    if hasattr(space, "get_indexes"):
        try:
            idx = np.asarray(space.get_indexes(attr))
            if idx.size:
                return int(idx.min()), int(idx.max()) + 1
        except Exception:
            pass
    attrs = list(getattr(space, "_attr_to_keep", []) or [])
    dims = getattr(space, "_dims", None)
    if attrs and dims is not None and attr in attrs:
        try:
            d = np.asarray(dims, dtype=int)
            # _dims may be cumulative offsets or per-attribute sizes.
            cum = d if np.all(np.diff(d) >= 0) and d[0] != d.sum() else np.cumsum(d)
            i = attrs.index(attr)
            lo = 0 if i == 0 else int(cum[i - 1])
            return lo, int(cum[i])
        except Exception:
            pass
    return None


def _validate_slice(state, sl, width):
    """A line_status slice must be `width` wide and binary-valued."""
    if sl is None or (sl[1] - sl[0]) != width:
        return False
    try:
        v = state[..., sl[0]:sl[1]].reshape(-1)
        u = torch.unique(v)
        return bool(u.numel() <= 2 and torch.all((u == 0) | (u == 1)))
    except Exception:
        return False


def _scan_for_binary_window(state, width):
    """Last resort: find the one `width`-wide window that is binary-valued."""
    try:
        flat = state.reshape(-1, state.shape[-1])
        n = flat.shape[-1]
        hits = []
        for lo in range(0, n - width + 1):
            v = flat[..., lo:lo + width]
            u = torch.unique(v)
            if u.numel() <= 2 and torch.all((u == 0) | (u == 1)):
                hits.append((lo, lo + width))
        return hits[0] if len(hits) == 1 else None
    except Exception:
        return None


def _f(x):
    """Tensor/array -> python float, NaN on anything unusable."""
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().float()
            if x.numel() == 0:
                return float("nan")
            return float(x.mean().item())
        a = np.asarray(x, dtype=float)
        return float(a.mean()) if a.size else float("nan")
    except Exception:
        return float("nan")


def _explained_variance(target, pred):
    try:
        t = target.detach().float().reshape(-1)
        p = pred.detach().float().reshape(-1)
        n = min(t.numel(), p.numel())
        if n < 2:
            return float("nan")
        t, p = t[:n], p[:n]
        vt = torch.var(t, unbiased=False)
        if not torch.isfinite(vt) or vt.item() <= 1e-12:
            return float("nan")
        return float((1.0 - torch.var(t - p, unbiased=False) / vt).item())
    except Exception:
        return float("nan")


def _get(td, group, name):
    """Fetch (group, name) with a flat-key fallback."""
    for key in ((group, name), name):
        try:
            v = td.get(key, None)
            if v is not None:
                return v
        except Exception:
            continue
    return None


class NSDiagnosticsCallback(Callback):
    def __init__(self, csv_path, run_tag="", flush_every=1):
        super().__init__()
        self.csv_path = csv_path
        self.run_tag = run_tag
        self.flush_every = max(1, int(flush_every))
        self._row = {}
        self._groups_done = set()
        self._writer = None
        self._fh = None
        self._header = None
        self._t0 = time.time()
        self._state_slice = None
        self._state_slice_tried = False
        self._state_slice_ok = None
        self._n_line = None
        self._rows_written = 0

    # -- setup ---------------------------------------------------------
    def on_setup(self):
        d = os.path.dirname(os.path.abspath(self.csv_path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._resolve_state_slice()
        print(f"[ns] diagnostics -> {os.path.abspath(self.csv_path)}"
              f"  (line_status slice: {self._state_slice})")

    def _resolve_state_slice(self):
        """Candidate only; confirmed against real data in _confirm_state_slice."""
        if self._state_slice_tried:
            return
        self._state_slice_tried = True
        # Path proven to work in this repo by BMMAAgent.load().
        for attrpath in ("test_env", "rollout_env"):
            try:
                env = getattr(self.experiment, attrpath)
                pz = env.base_env._env
                self._n_line = int(pz.env_g2op.n_line)
                self._state_slice = _locate_attr(pz._aux_state_space, "line_status")
                if self._state_slice:
                    return
            except Exception:
                continue

    def _confirm_state_slice(self, batch):
        """Validate (or repair) the slice once, against a real state tensor."""
        if self._state_slice_ok is not None:
            return
        try:
            state = batch.get("state")
        except Exception:
            self._state_slice_ok = False
            return
        width = getattr(self, "_n_line", None)
        if state is None or not width:
            self._state_slice_ok = False
            return

        if _validate_slice(state, self._state_slice, width):
            self._state_slice_ok = True
        else:
            found = _scan_for_binary_window(state, width)
            if found is not None:
                print(f"[ns] line_status slice {self._state_slice} failed "
                      f"validation; using scanned window {found}")
                self._state_slice, self._state_slice_ok = found, True
            else:
                print("[ns] could not locate a binary line_status window in the "
                      "state vector -- frac_degraded disabled (reported as NaN)")
                self._state_slice_ok = False

    # -- collection ----------------------------------------------------
    def on_batch_collected(self, batch):
        self._resolve_state_slice()
        exp = self.experiment
        row = {
            "run_tag": self.run_tag,
            "iter": int(getattr(exp, "n_iters_performed", -1)),
            "total_frames": int(getattr(exp, "total_frames", -1)),
            "wallclock_s": round(time.time() - self._t0, 1),
        }

        mr = getattr(exp, "mean_return", None)
        row["mean_return"] = _f(mr) if mr is not None else float("nan")

        # Episode turnover: frames per termination is a cheap survival proxy.
        try:
            done = batch.get(("next", "done"))
            n_done = float(done.sum().item())
            row["n_dones"] = n_done
            row["approx_ep_len"] = (float(done.numel()) / n_done
                                    if n_done > 0 else float("nan"))
        except Exception:
            row["n_dones"] = float("nan")
            row["approx_ep_len"] = float("nan")

        # Exogenous exposure of the collected frames.
        self._confirm_state_slice(batch)
        row["frac_degraded"] = float("nan")
        if self._state_slice_ok:
            try:
                lo, hi = self._state_slice
                ls = batch.get("state")[..., lo:hi]
                row["frac_degraded"] = float((ls < 0.5).any(-1).float().mean().item())
            except Exception:
                pass

        self._row = row
        self._groups_done = set()

    # -- per optimizer step --------------------------------------------
    def on_train_step(self, batch, group):
        # Called once per inner epoch; record only the first, whose
        # state_value/value_target come from process_batch (start of iteration).
        key = f"critic_ev__{group}"
        if key in self._row:
            return None

        vt = _get(batch, group, "value_target")
        sv = _get(batch, group, "state_value")

        # The cached "state_value" came back identically zero in the first run,
        # which makes EV exactly 0 for every agent and iteration (Var(t-0) ==
        # Var(t)) and looks like a finding. Recompute from the critic itself.
        try:
            loss = self.experiment.losses[group]
            critic = getattr(loss, "critic_network", None)
            if critic is not None:
                with torch.no_grad():
                    td = batch.clone(False)
                    critic(td)
                    sv2 = _get(td, group, "state_value")
                if sv2 is not None and float(sv2.float().std()) > 0:
                    sv = sv2
                    self._row[f"critic_sv_source__{group}"] = "recomputed"
        except Exception:
            pass

        # Log the components too, so a degenerate probe is visible next time
        # rather than being mistaken for a collapsed value function.
        if vt is not None:
            self._row[f"value_target_std__{group}"] = float(vt.float().std())
        if sv is not None:
            self._row[f"state_value_std__{group}"] = float(sv.float().std())
        if vt is not None and sv is not None:
            self._row[key] = _explained_variance(vt, sv)
            # Scale-free companion: independent of GAE's internal conventions.
            try:
                a = vt.detach().float().reshape(-1)
                b = sv.detach().float().reshape(-1)
                n = min(a.numel(), b.numel())
                a, b = a[:n], b[:n]
                if a.std() > 0 and b.std() > 0:
                    self._row[f"critic_corr__{group}"] = float(
                        ((a - a.mean()) * (b - b.mean())).mean()
                        / (a.std() * b.std()))
            except Exception:
                pass

        adv = _get(batch, group, "advantage")
        if adv is not None:
            try:
                self._row[f"adv_std__{group}"] = float(
                    adv.detach().float().std().item())
            except Exception:
                pass

        scale = _get(batch, group, "scale")
        if scale is not None:
            self._row[f"policy_scale__{group}"] = _f(scale)

        lp = _get(batch, group, "log_prob")
        if lp is not None:
            self._row[f"log_prob__{group}"] = _f(lp)

        return None

    # -- per group, end of its inner epochs -----------------------------
    def on_train_end(self, training_td, group):
        try:
            keys = list(training_td.keys())
        except Exception:
            keys = []
        for k in keys:
            if not isinstance(k, str):
                continue
            try:
                v = training_td.get(k)
            except Exception:
                continue
            self._row[f"{k}__{group}"] = _f(v)
            if k in _DRIFT_KEYS:
                try:
                    flat = v.detach().float().reshape(v.shape[0], -1).mean(-1)
                    if flat.numel() >= 2:
                        self._row[f"drift_{k}__{group}"] = float(
                            (flat[-1] - flat[0]).item())
                except Exception:
                    pass

        self._groups_done.add(group)
        try:
            n_groups = len(self.experiment.train_group_map)
        except Exception:
            n_groups = len(self._groups_done)
        if len(self._groups_done) >= n_groups:
            self._finalise_and_write()

    # -- aggregation + IO ----------------------------------------------
    def _finalise_and_write(self):
        row = self._row
        for prefix in ("critic_ev", "critic_corr", "policy_scale", "adv_std",
                       "log_prob", "loss_critic", "loss_objective", "ESS",
                       "value_target_std", "state_value_std",
                       "entropy", "clip_fraction", "kl_approx",
                       "drift_loss_objective", "drift_ESS"):
            vals = [v for k, v in row.items()
                    if k.startswith(prefix + "__") and v == v]
            if vals:
                row[f"{prefix}_mean"] = float(np.mean(vals))
                if prefix in ("critic_ev", "policy_scale"):
                    row[f"{prefix}_min"] = float(np.min(vals))
                    row[f"{prefix}_max"] = float(np.max(vals))

        if self._writer is None:
            self._header = list(row.keys())
            new = not os.path.exists(self.csv_path) or \
                os.path.getsize(self.csv_path) == 0
            self._fh = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=self._header,
                                          extrasaction="ignore")
            if new:
                self._writer.writeheader()

        self._writer.writerow({k: row.get(k, "") for k in self._header})
        self._rows_written += 1
        if self._rows_written % self.flush_every == 0:
            self._fh.flush()

        ev = row.get("critic_ev_mean", float("nan"))
        print(f"[ns] iter {row.get('iter')}  return={row.get('mean_return'):.3g}  "
              f"critic_ev={ev:.3f}  ep_len~{row.get('approx_ep_len', float('nan')):.1f}  "
              f"degraded={row.get('frac_degraded', float('nan')):.3f}")

    def __del__(self):
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
