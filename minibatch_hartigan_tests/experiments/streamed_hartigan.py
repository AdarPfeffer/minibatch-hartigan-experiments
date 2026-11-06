"""
Online (streamed) Hartigan k-means, numba-JIT'd.

Extracted verbatim from the compute-time benchmark
(ICML_numerical_results/compute_time_benchmark/benchmark.ipynb, cells 4 + 6):
sequential per-point reassignment with immediate incremental centroid updates,
singleton protection (distance -1 pins a lone point so a cluster can't empty), and
convergence when labels are unchanged or a 2-epoch cycle is detected.

The helpers (compute_mean, compute_centroids_np, assign_labels_lloyd_all,
_assign_label_hartigan_np) are shared with the numba mini-batch in mini_batch.py.
"""
import time

import numpy as np
from numba import jit


@jit
def _compute_loss_np(data, centroids, labels):
    return np.sum(np.abs(data - centroids[labels]) ** 2)


@jit
def compute_mean(data, mask):
    n_elements = np.sum(mask)
    if n_elements > 0:
        mean = np.sum(data[mask], axis=0)
        mean /= n_elements
    else:
        mean = np.zeros_like(data[0])
    return mean


@jit
def compute_centroids_np(data, labels, centroids):
    for i in range(centroids.shape[0]):
        idx_mask = labels == i
        centroids[i] = compute_mean(data, idx_mask)
    return centroids


@jit
def assign_labels_lloyd_all(centroids, data):
    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for n in range(data.shape[0]):
        for k in range(centroids.shape[0]):
            distances[n, k] = np.sum((data[n] - centroids[k]) ** 2)
    return np.argmin(distances, axis=1)


@jit
def _assign_label_hartigan_np(centroids, cluster_populations, data_point, label_point):
    distances = np.sum((data_point[None, ...] - centroids) ** 2, axis=-1)

    for i in range(centroids.shape[0]):
        if label_point == i:
            if cluster_populations[i] <= 1:
                distances[i] = -1.0  # prevent empty clusters
            else:
                scale_factor = cluster_populations[i] / (cluster_populations[i] - 1)
                distances[i] *= scale_factor
        else:
            scale_factor = cluster_populations[i] / (cluster_populations[i] + 1)
            distances[i] *= scale_factor
    return np.argmin(distances)


@jit
def _weighted_cost(centroids, cluster_populations, data_point, l, is_own):
    """One Hartigan-weighted cost, shared by EVERY evaluation path of the live-set port
    (full and reduced) so their arithmetic is bit-identical by construction.

    Matches _assign_label_hartigan_np / _mini_batch_sweep to the ulp: squared distance
    accumulated in ascending coordinate order (numba lowers their np.sum the same way —
    verified by the bitwise micro-test in test_live_sets.py), then the population ratio
    computed FIRST and applied with ONE multiply (their exact float op order), with the
    -1.0 singleton pin for a lone point's own cluster."""
    acc = 0.0
    for jj in range(data_point.shape[0]):
        diff = data_point[jj] - centroids[l, jj]
        acc += diff * diff
    if is_own:
        if cluster_populations[l] <= 1:
            return -1.0
        return acc * (cluster_populations[l] / (cluster_populations[l] - 1))
    return acc * (cluster_populations[l] / (cluster_populations[l] + 1))


@jit
def _full_eval(centroids, cluster_populations, data_point, label_point):
    """All-K Hartigan evaluation — decision-identical to _assign_label_hartigan_np
    (strict < keeps the FIRST minimum, i.e. np.argmin's tie-break) — additionally
    returning the own-cluster cost so callers can cache it."""
    best = 0
    best_w = 0.0
    w_own = 0.0
    for l in range(centroids.shape[0]):
        w = _weighted_cost(centroids, cluster_populations, data_point, l,
                           l == label_point)
        if l == 0 or w < best_w:
            best_w = w
            best = l
        if l == label_point:
            w_own = w
    return best, w_own


@jit
def _reduced_eval(centroids, cluster_populations, data_point, a, d_own_cached,
                  last_update, point_stamp):
    """Reduced evaluation for a point whose OWN cluster is unchanged since its last
    stay-decision (cached cost d_own_cached still exact): only clusters stamped AFTER
    point_stamp ("live") can possibly beat it — every other cluster's cost is bitwise
    unchanged and already lost. Tie rule replicates np.argmin's first-minimum: a live
    candidate with index below `a` wins an exact tie (argmin would have returned it
    first); one above `a` needs to be strictly better. Iterating ascending keeps the
    first winner among live candidates."""
    best = a
    best_w = d_own_cached
    for l in range(centroids.shape[0]):
        if l == a or last_update[l] <= point_stamp:
            continue
        w = _weighted_cost(centroids, cluster_populations, data_point, l, False)
        if w < best_w or (w == best_w and l < best):
            best_w = w
            best = l
    return best


@jit
def run_hartigan_kmeans_ls(data, init_centroids, max_iters, checks_convergence=True):
    """Online Hartigan with AS 136-style live sets (stamp-based, exact).

    Produces BIT-IDENTICAL results to run_hartigan_kmeans (labels, centroids, loss,
    n_iters — enforced by test_live_sets.py) while skipping provably-unchanged
    evaluations: `step` counts transfers; last_update[k] = step at which cluster k last
    changed (a transfer changes centroid and population together); last_eval[i] = step
    of point i's last stay-decision; d_own[i] = its cached own-cluster cost. A point
    whose own cluster changed since its last check re-evaluates everything (the cached
    comparisons are stale); otherwise only clusters stamped since then are examined.
    Every point pays a uniform, index-independent floor of K integer compares per pass
    (no whole-point skip by design — see the plan: predictable runtime over luck).
    The convergence rule, iteration accounting, and move arithmetic are copied verbatim
    from run_hartigan_kmeans."""
    labels = assign_labels_lloyd_all(init_centroids, data)
    centroids = compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    n = data.shape[0]
    last_update = np.zeros(init_centroids.shape[0], dtype=np.int64)
    last_eval = np.full(n, -1, dtype=np.int64)   # -1: first pass fully evaluates all
    d_own = np.zeros(n, dtype=np.float64)
    step = np.int64(0)                           # int64: bounded by n * max_iters

    old_labels = labels.copy()
    old_old_labels = labels.copy()
    n_iters = 1  # one iteration already done
    for _ in range(max_iters):
        for j in range(n):
            a = labels[j]
            if last_update[a] > last_eval[j]:
                # own cluster changed since this point's last decision: full evaluation
                new_label, w_own = _full_eval(centroids, cluster_populations,
                                              data[j], a)
                if new_label == a:
                    d_own[j] = w_own
                    last_eval[j] = step
                    continue
            else:
                # own cluster unchanged: cached own-cost exact; check live clusters only
                new_label = _reduced_eval(centroids, cluster_populations, data[j], a,
                                          d_own[j], last_update, last_eval[j])
                if new_label == a:
                    last_eval[j] = step
                    continue

            # transfer — verbatim from run_hartigan_kmeans
            n_clust = cluster_populations[a]
            centroids[a] = (centroids[a] * n_clust - data[j]) / (n_clust - 1.0)
            n_clust = cluster_populations[new_label]
            centroids[new_label] = (centroids[new_label] * n_clust + data[j]) / (
                n_clust + 1.0
            )
            cluster_populations[a] -= 1
            cluster_populations[new_label] += 1
            labels[j] = new_label
            # stamp AFTER incrementing: "changed since my eval" is then exactly
            # last_update[l] > last_eval[i], with no same-step ambiguity. The mover
            # itself needs no cache work — its new own cluster carries the newest
            # stamp, forcing a full evaluation on its next visit.
            step += 1
            last_update[a] = step
            last_update[new_label] = step

        break_condition = np.array_equal(labels, old_labels) or (
            np.array_equal(labels, old_old_labels) and n_iters > 0
        )

        if checks_convergence and break_condition:
            break
        else:
            old_old_labels = old_labels.copy()
            old_labels = labels.copy()
            n_iters += 1
    loss = _compute_loss_np(data, centroids, labels)
    return centroids, labels, loss, n_iters


@jit
def _online_epoch_ls(data, centroids, labels, cluster_populations, last_update,
                     last_eval, d_own, step):
    """EXACTLY ONE epoch of run_hartigan_kmeans_ls: a VERBATIM copy of its inner
    `for j in range(n)` sweep — same _full_eval / _reduced_eval decisions, same
    transfer arithmetic, same stamping order. Mutates centroids / labels /
    cluster_populations / last_update / last_eval / d_own in place and returns the
    updated transfer counter `step` (int64). NO convergence logic here — the caller
    owns the epoch loop. Exists so a python driver (run_hartigan_kmeans_ls_timed)
    can clock the post-init epoch loop without touching the frozen
    run_hartigan_kmeans_ls (bit-identity of the pair is enforced at benchmark
    warmup)."""
    n = data.shape[0]
    for j in range(n):
        a = labels[j]
        if last_update[a] > last_eval[j]:
            # own cluster changed since this point's last decision: full evaluation
            new_label, w_own = _full_eval(centroids, cluster_populations,
                                          data[j], a)
            if new_label == a:
                d_own[j] = w_own
                last_eval[j] = step
                continue
        else:
            # own cluster unchanged: cached own-cost exact; check live clusters only
            new_label = _reduced_eval(centroids, cluster_populations, data[j], a,
                                      d_own[j], last_update, last_eval[j])
            if new_label == a:
                last_eval[j] = step
                continue

        # transfer — verbatim from run_hartigan_kmeans
        n_clust = cluster_populations[a]
        centroids[a] = (centroids[a] * n_clust - data[j]) / (n_clust - 1.0)
        n_clust = cluster_populations[new_label]
        centroids[new_label] = (centroids[new_label] * n_clust + data[j]) / (
            n_clust + 1.0
        )
        cluster_populations[a] -= 1
        cluster_populations[new_label] += 1
        labels[j] = new_label
        # stamp AFTER incrementing: "changed since my eval" is then exactly
        # last_update[l] > last_eval[i], with no same-step ambiguity. The mover
        # itself needs no cache work — its new own cluster carries the newest
        # stamp, forcing a full evaluation on its next visit.
        step += 1
        last_update[a] = step
        last_update[new_label] = step
    return step


def run_hartigan_kmeans_ls_timed(data, init_centroids, max_iters,
                                 checks_convergence=True, init_labels=None):
    """PYTHON (non-jit) driver over _online_epoch_ls, BIT-IDENTICAL to
    run_hartigan_kmeans_ls in labels, centroids, loss and n_iters, additionally
    returning the post-init wall time: the init (Lloyd assignment + centroids +
    populations + stamp allocation, the exact lines of run_hartigan_kmeans_ls) runs
    UNTIMED through the same jit helpers; the clock covers ONLY the epoch loop
    (convergence checks included) and stops BEFORE the final loss. The convergence
    rule and n_iters accounting are copied verbatim from run_hartigan_kmeans_ls.
    Returns (centroids, labels, loss, n_iters, postinit_seconds).

    `init_labels` (n,) int64, when given, IS the starting partition: the unweighted
    nearest-centroid bootstrap below is SKIPPED, and `init_centroids` is then used only as a
    (K, d) buffer for compute_centroids_np (its contents are overwritten, so zeros are fine).
    That bootstrap exists solely to manufacture a grouping when only centroids are supplied —
    Hartigan's criterion weights by cluster size and cannot be evaluated before one exists.
    A partition-defined scheme already has the grouping and its sizes, so running it anyway
    discards the specified partition. See init_schemes.ASSIGNMENT_SCHEMES.
    Default None is the original path, byte-for-byte."""
    if init_labels is None:
        labels = assign_labels_lloyd_all(init_centroids, data)
    else:
        labels = np.asarray(init_labels, dtype=np.int64)
    centroids = compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    n = data.shape[0]
    last_update = np.zeros(init_centroids.shape[0], dtype=np.int64)
    last_eval = np.full(n, -1, dtype=np.int64)   # -1: first pass fully evaluates all
    d_own = np.zeros(n, dtype=np.float64)
    step = np.int64(0)                           # int64: bounded by n * max_iters

    old_labels = labels.copy()
    old_old_labels = labels.copy()
    n_iters = 1  # one iteration already done
    t0 = time.perf_counter()
    for _ in range(max_iters):
        step = _online_epoch_ls(data, centroids, labels, cluster_populations,
                                last_update, last_eval, d_own, step)

        break_condition = np.array_equal(labels, old_labels) or (
            np.array_equal(labels, old_old_labels) and n_iters > 0
        )

        if checks_convergence and break_condition:
            break
        else:
            old_old_labels = old_labels.copy()
            old_labels = labels.copy()
            n_iters += 1
    postinit_seconds = time.perf_counter() - t0
    loss = _compute_loss_np(data, centroids, labels)
    return centroids, labels, loss, n_iters, postinit_seconds


@jit
def run_hartigan_kmeans(data, init_centroids, max_iters, checks_convergence=True):
    # Initial quantities
    labels = assign_labels_lloyd_all(init_centroids, data)
    centroids = compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    # Variables to update
    old_labels = labels.copy()
    old_old_labels = labels.copy()
    n_iters = 1  # one iteration already done
    for _ in range(max_iters):
        for j in range(data.shape[0]):

            new_label = _assign_label_hartigan_np(
                centroids, cluster_populations, data[j], labels[j]
            )
            if new_label != labels[j]:
                n_clust = cluster_populations[labels[j]]
                centroids[labels[j]] = (centroids[labels[j]] * n_clust - data[j]) / (
                    n_clust - 1.0
                )

                n_clust = cluster_populations[new_label]
                centroids[new_label] = (centroids[new_label] * n_clust + data[j]) / (
                    n_clust + 1.0
                )

                cluster_populations[labels[j]] -= 1
                cluster_populations[new_label] += 1
                labels[j] = new_label

        break_condition = np.array_equal(labels, old_labels) or (
            np.array_equal(labels, old_old_labels) and n_iters > 0
        )

        if checks_convergence and break_condition:
            break
        else:
            old_old_labels = old_labels.copy()
            old_labels = labels.copy()
            n_iters += 1
    loss = _compute_loss_np(data, centroids, labels)
    return centroids, labels, loss, n_iters
