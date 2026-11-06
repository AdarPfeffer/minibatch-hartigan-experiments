"""
Mini-Batch Hartigan clustering (Phase 3, solution 4).

Each iteration reassigns a random fraction of the points by the Hartigan
transfer cost and recomputes every centroid from the full assignment. Sampling a
fresh subset each step breaks the deterministic, everyone-moves-at-once
oscillation that makes plain batch Hartigan cycle, so exact loops essentially
never occur below proportion ~= 1.0.

Structure mirrors batch_hartigan.py: compare_mini_batch is one reassignment pass
(cf. compare_batch_hart) and mini_batch_hartigan is the driver loop (cf.
batch_hartigan). By convention every function lists its return values
alphabetically.
"""

import sys

import math
import time

import numpy as np

from pathlib import Path

# batch_hartigan.py lives in the sibling study: <repo>/batch_hartigan_tests/experiments/
batch_hartigan_dir = Path(__file__).resolve().parents[2] / "batch_hartigan_tests" / "experiments"
sys.path.insert(0, str(batch_hartigan_dir))

from batch_hartigan import compare_init, create_rng, generate_ground_truth_data
from numba import jit
from streamed_hartigan import (
    _full_eval,
    _weighted_cost,
    assign_labels_lloyd_all,
    compute_centroids_np,
)


def mb_batch_params(n, amount_of_clusters, proportion):
    """(batch_size, patience) under the LEAVE-ONE-PER-CLUSTER constraint.

    The batch may never contain every point of a cluster, so at least `amount_of_clusters`
    points are held out and batch_size <= n - K. `patience` derives from the EFFECTIVE
    proportion (batch_size / n), which is the sampling rate actually used.
    """
    b_max = max(1, n - amount_of_clusters)
    batch_size = min(max(1, math.ceil(proportion * n)), b_max)   # round UP, then cap
    p_eff = batch_size / n
    patience = math.ceil(1.0 / p_eff ** 2) if p_eff > 0.33 else 10
    return batch_size, patience


def sample_batch_holdout(rng, assigns, amount_of_clusters, batch_size):
    """Sample `batch_size` point indices, PROPHYLACTICALLY never taking every member of a
    cluster: one uniformly-random member of each non-empty cluster is held out first, then the
    batch is drawn from the remaining points.

    A held-out point is not reassigned this pass, so its cluster keeps at least one member —
    hence NO cluster can ever be emptied by a mini-batch sweep. This replaces the broken
    reliance on the singleton pin: the sweep scales costs by FIXED start-of-pass populations
    (it is order-independent, which is what keeps CPU and GPU bit-identical), so the pin only
    fires for a cluster that already had one member and cannot stop several members of a small
    cluster from leaving simultaneously.

    The hold-out is re-drawn every pass, so no point is permanently frozen. Deterministic given
    `rng` and `assigns`, so runs stay reproducible from the per-cell mb seed.
    """
    n = len(assigns)
    held = []
    for k in range(amount_of_clusters):
        members = np.flatnonzero(assigns == k)
        if members.size:
            held.append(members[rng.integers(members.size)])
    mask = np.ones(n, dtype=bool)
    if held:
        mask[np.asarray(held, dtype=np.int64)] = False
    pool = np.flatnonzero(mask)
    if batch_size >= pool.size:           # cannot exceed the pool (batch_size <= n-K already)
        return pool
    return rng.choice(pool, size=batch_size, replace=False)


class CycleTracker:
    """OBSERVATIONAL cycle detector for mini-batch Hartigan. Records, per run, whether the
    assignment sequence revisits states and how large the recurrent set is. It NEVER stops the
    run: it consumes no rng, changes no decision, and leaves results BIT-IDENTICAL — it is
    pure instrumentation.

    Why not a proof like batch Hartigan's loop_checker: there, state == assignment and the map
    is deterministic, so a repeat proves eternal periodicity. In mini-batch the state is
    (assignment, rng position); PCG64's period is 2**128 and advances monotonically, so the full
    state never recurs and no cycle certificate can exist. A repeated ASSIGNMENT still shows the
    trajectory is confined to a small recurrent set, which is the reportable phenomenon: a
    measured example visited only 50 distinct states across 7,401 passes before terminating.

    `cycle_pass` = pass index of the FIRST revisit (-1 if none). `n_states` = distinct
    assignment states seen. `capped` = the run exhausted its iteration budget.
    """

    __slots__ = ("seen", "repeats", "cycle_pass", "first_lag")

    def __init__(self):
        self.seen = {}
        self.repeats = 0
        self.cycle_pass = -1
        self.first_lag = -1

    def observe(self, assigns, counter):
        """Record this pass's assignment. A repeat counts as a CYCLE only at lag >= 2.

        A lag-1 repeat means the assignment did not change from the previous pass — a quiet
        pass (moved == 0), which is the signal that TRIGGERS the convergence certificate, not
        an oscillation. Counting it would flag every healthy run: measured smoke runs of 5-34
        passes all contain quiet passes. batch_hartigan.py avoids this by testing
        np.array_equal(assigns, old_assigns) BEFORE calling loop_checker, so lag-1 never
        reaches its memo and every loop it reports has loop_size >= 2; this reproduces that
        semantics for mini-batch, where a quiet pass does NOT imply convergence (the next pass
        draws a different batch and may move points again).

        The lag is measured against the MOST RECENT prior occurrence, so it is the minimal
        cycle period (same notion as loop_size in the batch study).
        """
        h = hash(assigns.tobytes())
        prev = self.seen.get(h)
        self.seen[h] = counter                       # most recent occurrence
        if prev is None:
            return
        lag = counter - prev
        if lag < 2:                                  # unchanged partition, not an oscillation
            return
        self.repeats += 1
        if self.cycle_pass < 0:
            self.cycle_pass = counter
            self.first_lag = lag

    def write(self, diag, counter, max_iter):
        diag["passes"] = int(counter)
        diag["n_states"] = len(self.seen)
        diag["repeats"] = int(self.repeats)
        diag["cycle_pass"] = int(self.cycle_pass)
        diag["cycle_lag"] = int(self.first_lag)
        diag["capped"] = bool(counter >= max_iter)
        # CONFINEMENT: the fraction of passes that revisit an already-seen state. This is the
        # measure to report, NOT a boolean "cycled" flag: a lag>=2 repeat alone still fires on
        # ~15% of healthy runs (mini-batch loss is not monotone, so benign revisits happen),
        # whereas confinement separates cleanly and orders by severity. Measured scale:
        #   healthy grid proportions        <= 0.46  (median 0.12)
        #   p_max, ordinary (28-298 passes)  0.55 - 0.75
        #   p_max, pathological (7,401)      0.9932   <- 50 distinct states over 7,401 passes
        diag["confinement"] = (1.0 - len(self.seen) / counter) if counter > 0 else 0.0


def compare_mini_batch(data, centroids, assigns, amount_of_clusters, batch_size, rng, zero):
    """One mini-batch Hartigan reassignment pass.

    Sample `batch_size` point indices without replacement and reassign ONLY those
    points by the Hartigan transfer cost -- scaled by the FULL-dataset cluster
    sizes (n_k/(n_k-1) to stay, n_k/(n_k+1) to move), identical to
    compare_batch_hart -- then recompute every centroid from the full assignment.
    Sampling a fresh subset each call is what breaks the batch-Hartigan
    oscillation.

    Args:
        data:               (n, dim) data matrix
        centroids:          (amount_of_clusters, dim) current centers (mutated)
        assigns:            (n,) current assignments (sampled entries mutated)
        amount_of_clusters: number of clusters
        batch_size:         number of points to reassign this pass
        rng:                numpy Generator
        zero:               bool flag -- True if an empty cluster was already seen

    Returns (alphabetical):
        assigns: (n,) updated assignments
        moved:   int -- number of sampled points that changed cluster
        zero:    bool -- True if an empty cluster was detected
    """
    n = len(data)
    populations = np.bincount(assigns, minlength=amount_of_clusters)
    if (populations == 0).any():
        zero = True

    batch_idx = sample_batch_holdout(rng, assigns, amount_of_clusters, batch_size)
    before = assigns[batch_idx].copy()

    # Hartigan-scaled squared distances for the sampled points only; scaling uses
    # the FULL-dataset cluster sizes, identical to compare_batch_hart.
    sq = np.linalg.norm(
        data[batch_idx][:, None, :] - centroids[None, :, :], axis=-1) ** 2
    pop = populations.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        move_factor = pop / (pop + 1.0)                           # n_k / (n_k + 1)
        stay_factor = np.where(pop == 1, 0.0, pop / (pop - 1.0))  # n_k / (n_k - 1)
    scale = np.broadcast_to(move_factor, (batch_size, amount_of_clusters)).copy()
    scale[np.arange(batch_size), before] = stay_factor[before]
    assigns[batch_idx] = (sq * scale).argmin(axis=1)

    # Recompute centroids from the full (mostly unchanged) assignment.
    for k in range(amount_of_clusters):
        centroids[k] = np.mean(data[assigns == k], axis=0)

    moved = int(np.sum(before != assigns[batch_idx]))   # count of sampled points that changed
    return assigns, moved, zero


def mini_batch_hartigan(data, init_centers, centroids, amount_of_clusters, rng,
                        proportion=0.5, max_iter=500, tolerance_fraction=0.01):
    """Run mini-batch Hartigan until convergence, empty cluster, or the budget.

    Convergence: `patience` CONSECUTIVE quiet passes, where a pass is quiet when it
    reassigns at most tolerance = floor(tolerance_fraction * n) sampled points (the
    quiet counter resets the moment a pass moves more than that; tolerance_fraction=0.0
    recovers the classic zero-move rule). `patience` is derived from `proportion` --
    ceil(1 / proportion**2) above 0.33, else 10 -- so a smaller subset (less coverage
    per pass) demands a longer quiet streak.

    Empty cluster: the per-pass `zero` flag (set from the incoming populations)
    returns immediately, exactly as in batch_hartigan. Init never produces empties
    (compare_init gives every distinct data-point center its own point), so no
    special init guard is needed.

    Args:
        data:               (n, dim) data matrix
        init_centers:       (amount_of_clusters, dim) initial centers (record)
        centroids:          (amount_of_clusters, dim) mutable working centers
        amount_of_clusters: number of clusters
        rng:                numpy Generator (the trial's rng -- the one that built
                            `data`/`init_centers`; also drives the sampling)
        proportion:         fraction of points reassigned per pass (0 < p <= 1)
        max_iter:           per-trial iteration budget
        tolerance_fraction: a pass is quiet when it moves at most
                            floor(tolerance_fraction * n) points (0.0 = zero-move rule)

    Returns (alphabetical):
        assigns:   (n,) final assignments
        centroids: (amount_of_clusters, dim) final centers
        counter:   iterations performed (counter < max_iter and not zero => converged)
        zero:      True if an empty cluster was encountered
    """
    n = len(data)
    # leave-one-per-cluster constraint: batch_size <= n - K (see sample_batch_holdout)
    batch_size, patience = mb_batch_params(n, amount_of_clusters, proportion)
    tolerance = math.floor(tolerance_fraction * n)   # 1% of data size by default

    # Initial assignment: each point to its nearest initial center.
    assigns = compare_init(data, init_centers)
    for k in range(amount_of_clusters):
        centroids[k] = np.mean(data[assigns == k], axis=0)

    zero = False
    counter = 0
    quiet = 0
    while counter < max_iter:
        assigns, moved, zero = compare_mini_batch(
            data, centroids, assigns, amount_of_clusters, batch_size, rng, zero)
        counter += 1
        if zero:
            return assigns, centroids, counter, zero
        if moved <= tolerance:
            quiet += 1
            if quiet >= patience:
                return assigns, centroids, counter, zero
        else:
            quiet = 0

    return assigns, centroids, counter, zero


# --- Numba-accelerated variant (used by the runtime diagnostic vs streamed Hartigan) --
# Verified to produce results identical to the numpy mini_batch_hartigan above: same rng
# seed -> identical (assigns, counter, zero) across well-separated, high-dim, and
# overlapping regimes. Reuses the jit'd helpers from streamed_hartigan so online and
# mini-batch share the exact per-point Hartigan cost and centroid recompute.


@jit
def _mini_batch_sweep(data, centroids, cluster_populations, batch_idx, assigns, amount_of_clusters):
    """Reassign the sampled points by the Hartigan transfer cost, as a numba double loop
    over the sampled batch (Hartigan scaling, FIXED centroids/populations). Updates and
    RETURNS `assigns`."""
    for b in range(batch_idx.shape[0]):
        j = batch_idx[b]
        a = assigns[j]
        distances = np.zeros(amount_of_clusters)
        for k in range(amount_of_clusters):
            d = np.sum((data[j] - centroids[k]) ** 2)
            if k == a:
                if cluster_populations[k] <= 1:
                    d = -1.0                                  # pin a lone point (no empties)
                else:
                    d *= cluster_populations[k] / (cluster_populations[k] - 1)
            else:
                d *= cluster_populations[k] / (cluster_populations[k] + 1)
            distances[k] = d
        assigns[j] = np.argmin(distances)
    return assigns


def mini_batch_hartigan_nb(data, init_centers, centroids, amount_of_clusters, rng,
                           proportion=0.5, max_iter=500, tolerance_fraction=0.01,
                           diag=None):
    """Numba-accelerated mini-batch Hartigan. Same signature/semantics as
    mini_batch_hartigan; the hot compute (init via assign_labels_lloyd_all, the per-batch
    sweep, and the centroid recompute via compute_centroids_np) is jit'd, while the driver
    stays Python so `rng` is a numpy Generator used for sampling (never created inside).
    Convergence counts consecutive quiet batches toward `patience`, where a batch is quiet
    when it moves at most tolerance = floor(tolerance_fraction * n) points; the snapshot
    `old_labels` refreshes EVERY pass so the count is per-pass movement, not drift against
    a stale snapshot (at tolerance 0 this is behavior-identical to the classic rule).

    Returns (alphabetical):
        assigns:   (n,) final assignments
        centroids: (amount_of_clusters, dim) final centers
        counter:   iterations performed
        zero:      True if an empty cluster was encountered
    """
    n = len(data)
    # leave-one-per-cluster constraint: batch_size <= n - K (see sample_batch_holdout)
    batch_size, patience = mb_batch_params(n, amount_of_clusters, proportion)
    tolerance = math.floor(tolerance_fraction * n)   # 1% of data size by default

    assigns = assign_labels_lloyd_all(init_centers, data)
    centroids = compute_centroids_np(data, assigns, centroids)
    old_labels = assigns.copy()

    counter = 0
    quiet = 0
    zero = False
    _trk = CycleTracker()                    # observational only: never stops the run
    if diag is not None:
        _trk.write(diag, counter, max_iter)
    while counter < max_iter:
        populations = np.bincount(assigns, minlength=amount_of_clusters)
        if (populations == 0).any():
            zero = True
            return assigns, centroids, counter, zero
        if diag is not None:                 # skipped entirely in production -> zero cost
            _trk.observe(assigns, counter)
        batch_idx = sample_batch_holdout(rng, assigns, amount_of_clusters, batch_size)
        assigns = _mini_batch_sweep(
            data, centroids, populations, batch_idx, assigns, amount_of_clusters)
        centroids = compute_centroids_np(data, assigns, centroids)
        counter += 1
        if diag is not None:                 # refreshed every pass -> valid on every return path
            _trk.write(diag, counter, max_iter)
        moved_count = int(np.sum(assigns != old_labels))
        old_labels = assigns.copy()          # snapshot EVERY pass -> per-pass movement
        if moved_count <= tolerance:
            quiet += 1
            if quiet >= patience:
                return assigns, centroids, counter, zero
        else:
            quiet = 0
    return assigns, centroids, counter, zero


# --- Certified mini-batch: live-set stamps + TRUE-convergence certificate -------------
# The patience/tolerance heuristic above stops on a streak of quiet passes, which proves
# nothing about unsampled points. mini_batch_hartigan_nb_certified keeps the identical
# mini-batch dynamics but (a) removes tolerance entirely (quiet means moved == 0), and
# (b) when the quiet streak fires, runs a fixed-point CHECK over all n points (applying
# nothing): only a pass of zero would-movers counts as convergence — the same fixed-point
# condition the online algorithm converges to. AS 136-style stamps make repeated checks
# cheap: a point is re-evaluated against a cluster only if that cluster changed since the
# point's last stay-decision.
#
# Stamps are PASS-granular here (every decision in a pass sees the pass-start state):
#   last_update[l] = pass at whose END cluster l's membership changed (gained/lost >= 1
#                    point; a swap keeps the count but moves the recomputed mean)
#   last_eval[i]   = pass at whose START state point i was evaluated and stayed
#   cache for l is valid iff last_update[l] < last_eval[i]   (STRICT <)
# Unchanged membership => bit-identical recomputed centroid (deterministic recompute), so
# "not stamped" really does mean "same bits".


@jit
def _decide_ls(data, centroids, cluster_populations, assigns, j,
               last_update, last_eval, d_own, eval_stamp):
    """Would point j move, given the current (pass-start) centroids + populations?
    Returns its argmin label; refreshes the point's cache when the answer is 'stay'.
    Mutates nothing else — callers decide whether to apply the move."""
    a = assigns[j]
    if last_update[a] >= last_eval[j]:
        # own cluster changed since the cached decision (or never evaluated):
        # the cached own-cost is stale -> full evaluation over all K
        new_label, w_own = _full_eval(centroids, cluster_populations, data[j], a)
        if new_label == a:
            d_own[j] = w_own
            last_eval[j] = eval_stamp
        return new_label
    # own cluster unchanged: cached own-cost is exact; only clusters stamped since the
    # cached decision can beat it. Tie rule = np.argmin first-minimum (see _reduced_eval
    # in streamed_hartigan.py for the argument).
    new_label = a
    best_w = d_own[j]
    for l in range(centroids.shape[0]):
        if l == a or last_update[l] < last_eval[j]:
            continue
        w = _weighted_cost(centroids, cluster_populations, data[j], l, False)
        if w < best_w or (w == best_w and l < new_label):
            best_w = w
            new_label = l
    if new_label == a:
        last_eval[j] = eval_stamp
    return new_label


@jit
def _mini_batch_sweep_ls(data, centroids, cluster_populations, batch_idx, assigns,
                         last_update, last_eval, d_own, t, changed):
    """One mini-batch pass over the sampled points (decision-identical to
    _mini_batch_sweep; stamp-accelerated). Applies moves, marks changed clusters,
    returns the move count. t = 1-based pass index = the state every decision sees."""
    moved = 0
    for b in range(batch_idx.shape[0]):
        j = batch_idx[b]
        new_label = _decide_ls(data, centroids, cluster_populations, assigns, j,
                               last_update, last_eval, d_own, t)
        if new_label != assigns[j]:
            changed[assigns[j]] = True
            changed[new_label] = True
            assigns[j] = new_label
            moved += 1
    return moved


@jit
def _is_fixed_point(data, centroids, cluster_populations, assigns,
                    last_update, last_eval, d_own, eval_stamp):
    """The convergence certificate: True iff NO point in the dataset would move under
    the current centroids + populations. Applies nothing; EARLY-EXITS on the first
    would-mover (movers are never applied, so counting them is waste). Stayers examined
    before the exit keep their cache refreshes, so a repeated check resumes cheaply."""
    for j in range(data.shape[0]):
        if _decide_ls(data, centroids, cluster_populations, assigns, j,
                      last_update, last_eval, d_own, eval_stamp) != assigns[j]:
            return False
    return True


def mini_batch_hartigan_nb_certified(data, init_centers, centroids, amount_of_clusters,
                                     rng, proportion=0.5, max_iter=500, certify=True,
                                     diag=None):
    """Mini-batch Hartigan with a TRUE-convergence certificate (no tolerance parameter
    — deliberately: tolerance is a heuristic and this function's contract is exactness).

    Identical mini-batch dynamics to mini_batch_hartigan_nb with tolerance_fraction=0.0
    (same rng stream, same passes — enforced bitwise by test_live_sets.py). `patience`
    consecutive zero-move passes now merely TRIGGER the certificate; convergence is
    declared only when a fixed-point check over all n points finds no would-mover. On a
    failed check the movers are NOT applied (applying them all at once is batch
    Hartigan, the documented oscillation mode) — mini-batch passes simply continue.
    The check consumes no rng, so a certified run is the certify=False run plus extra
    passes. certify=False reproduces the heuristic stop (for equivalence testing).

    Returns (alphabetical): assigns, centroids, certified, counter, zero.
      certified: True only with a proven fixed point (False on budget/zero exhaustion).
      counter counts mini-batch passes only (certificate checks are not passes)."""
    n = len(data)
    # leave-one-per-cluster constraint: batch_size <= n - K (see sample_batch_holdout)
    batch_size, patience = mb_batch_params(n, amount_of_clusters, proportion)

    assigns = assign_labels_lloyd_all(init_centers, data)
    centroids = compute_centroids_np(data, assigns, centroids)

    K = amount_of_clusters
    last_update = np.zeros(K, dtype=np.int64)
    last_eval = np.full(n, -1, dtype=np.int64)   # -1: first evaluation is always full
    d_own = np.zeros(n, dtype=np.float64)
    changed = np.zeros(K, dtype=np.bool_)

    counter = 0
    quiet = 0
    zero = False
    certified = False
    _trk = CycleTracker()                       # observational only: never stops the run
    if diag is not None:
        _trk.write(diag, counter, max_iter)
    while counter < max_iter:
        populations = np.bincount(assigns, minlength=K)
        if (populations == 0).any():
            zero = True
            return assigns, centroids, certified, counter, zero
        _trk.observe(assigns, counter)           # instrumentation only (no rng, no decision)
        batch_idx = sample_batch_holdout(rng, assigns, amount_of_clusters, batch_size)
        t = counter + 1                      # 1-based pass index (state = start of pass t)
        changed[:] = False
        moved = _mini_batch_sweep_ls(data, centroids, populations, batch_idx, assigns,
                                     last_update, last_eval, d_own, t, changed)
        centroids = compute_centroids_np(data, assigns, centroids)
        last_update[changed] = t             # stamped at pass END, after the recompute
        counter += 1
        if diag is not None:                 # refreshed every pass -> valid on every return path
            _trk.write(diag, counter, max_iter)
        if moved == 0:                       # no tolerance: quiet means QUIET
            quiet += 1
            if quiet >= patience:
                if not certify:
                    return assigns, centroids, certified, counter, zero
                if batch_size == n:
                    # a full-batch zero-move pass evaluated every point against the
                    # fixed pass-start state and moved none: it IS the certificate
                    certified = True
                    return assigns, centroids, certified, counter, zero
                populations = np.bincount(assigns, minlength=K)
                # the check sees the state after pass `counter`'s recompute, i.e. the
                # start-of-pass-(counter+1) state — hence the eval stamp counter + 1
                if _is_fixed_point(data, centroids, populations, assigns,
                                   last_update, last_eval, d_own, counter + 1):
                    certified = True
                    return assigns, centroids, certified, counter, zero
                quiet = 0                    # movers exist somewhere: keep mini-batching
        else:
            quiet = 0
    return assigns, centroids, certified, counter, zero


def mini_batch_hartigan_nb_certified_timed(data, init_centers, centroids,
                                           amount_of_clusters, rng, proportion=0.5,
                                           max_iter=500, certify=True, diag=None,
                                           init_labels=None):
    """mini_batch_hartigan_nb_certified with post-init timing — the ONLY difference is
    perf_counter() reads around the pass loop. The Lloyd init (assign_labels_lloyd_all
    + compute_centroids_np) and the stamp-array allocations happen BEFORE the clock
    starts, so postinit_seconds is the wall time of the mini-batch passes plus whatever
    certificate checks they trigger — the marginal per-pass cost is postinit_seconds /
    counter. perf_counter touches no rng state, so outputs are BIT-IDENTICAL to the
    untimed entry point (same rng stream, same passes — asserted at benchmark warmup).
    Keep the loop body verbatim in sync with mini_batch_hartigan_nb_certified.

    Returns (alphabetical, timing appended):
        assigns, centroids, certified, counter, zero, postinit_seconds.

    `init_labels` (n,) int64, when given, IS the starting partition: the unweighted
    nearest-centroid bootstrap below is SKIPPED and `init_centers` goes unused (`centroids`
    is already a separate (K, d) buffer). That bootstrap exists only to manufacture a grouping
    from centroids, since Hartigan's criterion weights by cluster size and needs one; a
    partition-defined scheme supplies the grouping and its sizes directly, so running it
    anyway discards them. See init_schemes.ASSIGNMENT_SCHEMES.
    Default None is the original path, byte-for-byte."""
    n = len(data)
    # leave-one-per-cluster constraint: batch_size <= n - K (see sample_batch_holdout)
    batch_size, patience = mb_batch_params(n, amount_of_clusters, proportion)

    if init_labels is None:
        assigns = assign_labels_lloyd_all(init_centers, data)
    else:
        assigns = np.asarray(init_labels, dtype=np.int64)
    centroids = compute_centroids_np(data, assigns, centroids)

    K = amount_of_clusters
    last_update = np.zeros(K, dtype=np.int64)
    last_eval = np.full(n, -1, dtype=np.int64)   # -1: first evaluation is always full
    d_own = np.zeros(n, dtype=np.float64)
    changed = np.zeros(K, dtype=np.bool_)

    counter = 0
    quiet = 0
    zero = False
    certified = False
    _trk = CycleTracker()                # observational only: never stops the run
    _inst = 0.0                          # accumulated instrumentation time, SUBTRACTED below so
    if diag is not None:                 # the post-init clock is unaffected by diagnostics
        _trk.write(diag, counter, max_iter)
    t_start = time.perf_counter()        # init excluded: the clock sees the pass loop only
    while counter < max_iter:
        populations = np.bincount(assigns, minlength=K)
        if (populations == 0).any():
            zero = True
            return (assigns, centroids, certified, counter, zero,
                    time.perf_counter() - t_start - _inst)
        if diag is not None:             # skipped entirely in production (diag=None) -> zero cost
            _i0 = time.perf_counter()
            _trk.observe(assigns, counter)   # instrumentation only (no rng, no decision)
            _inst += time.perf_counter() - _i0
        batch_idx = sample_batch_holdout(rng, assigns, amount_of_clusters, batch_size)
        t = counter + 1                      # 1-based pass index (state = start of pass t)
        changed[:] = False
        moved = _mini_batch_sweep_ls(data, centroids, populations, batch_idx, assigns,
                                     last_update, last_eval, d_own, t, changed)
        centroids = compute_centroids_np(data, assigns, centroids)
        last_update[changed] = t             # stamped at pass END, after the recompute
        counter += 1
        if diag is not None:                 # refreshed every pass -> valid on every return path
            _trk.write(diag, counter, max_iter)
        if moved == 0:                       # no tolerance: quiet means QUIET
            quiet += 1
            if quiet >= patience:
                if not certify:
                    return (assigns, centroids, certified, counter, zero,
                            time.perf_counter() - t_start - _inst)
                if batch_size == n:
                    # a full-batch zero-move pass evaluated every point against the
                    # fixed pass-start state and moved none: it IS the certificate
                    certified = True
                    return (assigns, centroids, certified, counter, zero,
                            time.perf_counter() - t_start - _inst)
                populations = np.bincount(assigns, minlength=K)
                # the check sees the state after pass `counter`'s recompute, i.e. the
                # start-of-pass-(counter+1) state — hence the eval stamp counter + 1
                if _is_fixed_point(data, centroids, populations, assigns,
                                   last_update, last_eval, d_own, counter + 1):
                    certified = True
                    return (assigns, centroids, certified, counter, zero,
                            time.perf_counter() - t_start - _inst)
                quiet = 0                    # movers exist somewhere: keep mini-batching
        else:
            quiet = 0
    return (assigns, centroids, certified, counter, zero,
            time.perf_counter() - t_start - _inst)


rng, seed = create_rng()
centroids, data, init_centers, labels = generate_ground_truth_data(
    rng, cluster_size=100, amount_of_clusters=5, dim=2,
    intracluster_variance=1.0, intercluster_variance=1.0)
assigns, centroids, counter, zero = mini_batch_hartigan(
    data, init_centers, centroids, 5, rng, proportion=0.5, max_iter=500)
