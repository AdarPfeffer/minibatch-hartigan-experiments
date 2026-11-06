import math
import time
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray
from numba import jit

from _common_functions_for_minibatch import assign_clusters, compute_centroids, compute_loss


@jit
def _compute_loss_np(data, centroids, labels):
    return np.sum(np.abs(data - centroids[labels]) ** 2)


@jit
def _assign_labels_lloyd_np(centroids, data):
    distances = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=1)


@jit
def _compute_centroids_np(data, labels, centroids):
    for i in range(centroids.shape[0]):
        idx_mask = labels == i
        n_elements = np.sum(idx_mask)
        if n_elements > 0:
            centroids[i] = np.sum(data[idx_mask], axis=0) / n_elements
        else:
            centroids[i] = np.zeros_like(centroids[i])
    return centroids


@jit
def _assign_label_hartigan_np(centroids, cluster_populations, data_point, label_point):
    distances = np.sum((data_point[None, ...] - centroids) ** 2, axis=-1)

    for i in range(centroids.shape[0]):
        if label_point == i:
            if cluster_populations[i] <= 1:
                distances[i] = -1.0  # always assign
            else:
                scale_factor = cluster_populations[i] / (cluster_populations[i] - 1)
                distances[i] *= scale_factor

        else:
            scale_factor = cluster_populations[i] / (cluster_populations[i] + 1)
            distances[i] *= scale_factor
    return np.argmin(distances)


@jit
def _run_hartigan_numpy(data, init_centroids, max_iters):
    max_iters = 100

    # Initial quantities
    labels = _assign_labels_lloyd_np(init_centroids, data)
    centroids = _compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    # Variables to update
    old_labels = labels.copy()
    n_iters = 0
    for n_iters in range(max_iters):
        for j in range(data.shape[0]):
            new_label = _assign_label_hartigan_np(
                centroids, cluster_populations, data[j], labels[j]
            )
            if new_label != labels[j]:
                # centroids = compute_centroids(data, labels, centroids)
                n_clust1 = cluster_populations[labels[j]]
                centroids[labels[j]] = (centroids[labels[j]] * n_clust1 - data[j]) / (
                    n_clust1 - 1.0
                )

                n_clust2 = cluster_populations[new_label]
                centroids[new_label] = (centroids[new_label] * n_clust2 + data[j]) / (
                    n_clust2 + 1.0
                )

                cluster_populations[labels[j]] -= 1
                cluster_populations[new_label] += 1
                labels[j] = new_label

        if np.array_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()

    loss = _compute_loss_np(data, centroids, labels)
    return centroids, labels, loss, n_iters


def run_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    """
    Run k-means clustering using online Hartigan's algorithm. Unlike other algorithms
    in the library, this is implemented to run on Numba, and thus has no GPU support.

    **Arguments**:
        data: A numpy-like array of shape (n, d) containing the data points.
        init_centroids: A numpy-like array of shape (K, d)
                        containing the initial centroids.
        max_iters: maximum number of iterations.

    **Returns**:
        A tuple containing:
            - centroids: A numpy-like array of shape (K, d) containing the final centroids
            - labels: A numpy-like array of shape (n,) containing the final cluster labels
            - loss: A float representing the final k-means loss.
            - n_iters: An integer representing the number of iterations performed.
    """
    return _run_hartigan_numpy(
        np.asanyarray(data), np.asanyarray(init_centroids), max_iters
    )


############################ Batched Hartigan ############################
def weight_distance(assignment, cluster_id, cluster_weight, distance):
    # branchless equivalent of the original nested lax.cond (bit-identical values under the
    # double vmap in weight_distances, which lowered the conds to select anyway):
    #   own cluster : -1.0 if weight <= 1 else distance * weight / (weight - 1)
    #   other       : distance * weight / (weight + 1)
    stay = jnp.where(
        cluster_weight <= 1, -1.0, distance * cluster_weight / (cluster_weight - 1)
    )
    move = distance * cluster_weight / (cluster_weight + 1)
    return jnp.where(assignment == cluster_id, stay, move)


def weight_distances(labels, cluster_ids, cluster_populations, distances):
    return jax.vmap(
        jax.vmap(weight_distance, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0)
    )(labels, cluster_ids, cluster_populations, distances)


def assign_dp_to_cluster_batched_hartigan(centroids, labels, cluster_populations, data):
    # Squared distances via ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2, computed as a GEMM at full
    # float32 precision (precision="highest" avoids TF32) instead of the (batch, K, d) broadcast
    # tensor. Keep the ||x||^2 term: distances are multiplied by per-cluster population ratios
    # in weight_distances before the argmin, so the per-point constant is NOT droppable here.
    # Clamp the rare float32 cancellation (true distance ~ 0) to 0.
    xn = jnp.sum(data * data, axis=1)[:, None]                       # (batch, 1) ||x||^2
    cn = jnp.sum(centroids * centroids, axis=1)[None, :]             # (1, K)     ||c||^2
    distances = jnp.maximum(
        xn + cn - 2.0 * jnp.matmul(data, centroids.T, precision="highest"), 0.0
    )
    distances = weight_distances(
        labels, jnp.arange(centroids.shape[0]), cluster_populations, distances
    )
    return jnp.argmin(distances, axis=-1)


def _batched_hartigan_step(carry, data):
    centroids, old_labels, old_old_labels, _, loss, counter = carry

    cluster_populations = jnp.bincount(old_labels, length=centroids.shape[0])
    labels = assign_dp_to_cluster_batched_hartigan(
        centroids, old_labels, cluster_populations, data
    )
    # jax.debug.print(
    #     "Step {c}, Assignments {a}, cluster populations {p}",
    #     c=counter,
    #     a=old_labels,
    #     p=cluster_populations,
    # )
    centroids = compute_centroids(data, labels, centroids.shape[0])

    # loss carried unchanged; computed once after the loop (run_batched_hartigan_kmeans).
    return (
        centroids,
        labels,
        old_labels,
        old_old_labels,
        loss,
        counter + 1,
    )


def _batched_hartigan_stop_condition(carry, max_steps):
    (
        centroids,
        labels,
        old_labels,
        old_old_labels,
        loss,
        counter,
    ) = carry

    cond1 = jnp.any(labels != old_labels)
    cond2 = jnp.any(labels != old_old_labels)
    cond3 = counter <= max_steps
    return cond1 & cond2 & cond3


@partial(jax.jit, static_argnums=(2,))
def _run_batched_hartigan_inner(data, init_centroids, max_iters):
    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])

    # makes sure the initial assignment does not trigger the stop condition
    carry = (
        init_centroids,
        init_labels,
        init_labels - 1,
        init_labels - 2,
        0.0,
        0,
    )
    return jax.lax.while_loop(
        cond_fun=lambda c: _batched_hartigan_stop_condition(c, max_steps=max_iters),
        body_fun=lambda c: _batched_hartigan_step(c, data),
        init_val=carry,
    )


def run_batched_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    # `data` is a TRACED argument to the jitted inner (not a closed-over constant), so the
    # while_loop compiles once per (shape, max_iters) and is reused across calls.
    centroids, labels, _, _, _loss, counter = _run_batched_hartigan_inner(
        data, init_centroids, max_iters
    )
    loss = compute_loss(data, centroids, labels)   # computed once from the final state
    return centroids, labels, loss, counter


########################## Mini-batch Hartigan ##########################
def _sample_batch_holdout(key, old_labels, K, n, batch_size):
    """Sample `batch_size` indices, PROPHYLACTICALLY never taking every member of a cluster:
    one uniformly-random member of each non-empty cluster is held out, then the batch is drawn
    from the rest. The held-out point is not reassigned, so its cluster keeps >= 1 member and
    can never be emptied. (The singleton pin cannot do this job: the sweep scales costs by FIXED
    start-of-pass populations, so several members of a small cluster leave simultaneously.)

    Fully vectorized — no lax.scan over points, so the kernel's parallelism (which the runtime
    study measures) is untouched. Mirrors mini_batch.sample_batch_holdout.
    """
    k_hold, k_pick = jax.random.split(key)
    # per-cluster random member: segmented argmin of a random key over each cluster's points.
    # dtype PINNED to float32: without it the draw follows the ambient default, so enabling
    # jax_enable_x64 would return float64, consume a different number of Threefry words per
    # element, and SELECT DIFFERENT POINTS — silently confounding any float32-vs-float64
    # comparison with a change of sampling. Measured no-op at float32 (12/12 fixtures
    # byte-identical, digest 19f067d7928d), so previously collected float32 results stay
    # bit-reproducible; test_fp64_precision_isolation.py pins that digest as a regression check.
    # jnp.inf below stays a weakly-typed Python scalar and does not upcast the f32 sort key.
    r = jax.random.uniform(k_hold, (n,), dtype=jnp.float32)
    masked = jnp.where(old_labels[:, None] == jnp.arange(K)[None, :], r[:, None], jnp.inf)
    held = jnp.argmin(masked, axis=0)                       # (K,) chosen point per cluster
    pops = jnp.bincount(old_labels, length=K)
    # scatter-MAX (order-independent) so an empty cluster's bogus argmin=0 cannot unset a real
    # hold-out that happens to be point 0
    held_mask = jnp.zeros(n, dtype=jnp.int32).at[held].max(
        (pops > 0).astype(jnp.int32)) > 0
    # non-held points first, in random order -> take the first batch_size (all non-held, since
    # batch_size <= n - K <= #non-held)
    order = jnp.argsort(jnp.where(held_mask, jnp.inf,
                                  jax.random.uniform(k_pick, (n,), dtype=jnp.float32)))
    return order[:batch_size]


def _minibatch_hartigan_step(carry, data, batch_size, tolerance):
    centroids, old_labels, loss, counter, quiet, key = carry

    key, subkey = jax.random.split(key)

    cluster_populations = jnp.bincount(old_labels, length=centroids.shape[0])
    subset_idx = _sample_batch_holdout(
        subkey, old_labels, centroids.shape[0], data.shape[0], batch_size)
    labels = assign_dp_to_cluster_batched_hartigan(
        centroids, old_labels[subset_idx], cluster_populations, data[subset_idx]
    )
    labels = old_labels.at[subset_idx].set(labels)

    centroids = compute_centroids(data, labels, centroids.shape[0])

    # loss carried unchanged; computed once after the loop (run_minibatch_hartigan_kmeans),
    # not every pass — the stop condition only reads counter/quiet.
    moved_count = jnp.sum(labels != old_labels)     # points that changed cluster this pass
    quiet = jnp.where(moved_count > tolerance, 0, quiet + 1)   # tolerate small residual churn
    return (centroids, labels, loss, counter + 1, quiet, key)


def _minibatch_hartigan_stop_condition(carry, patience, max_steps):
    (
        _,
        _,
        _,
        counter,
        quiet,
        _,
    ) = carry

    cond1 = quiet < patience          # keep going until the quiet streak reaches patience
    cond2 = counter <= max_steps

    return cond1 & cond2


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _run_minibatch_hartigan_inner(data, init_centroids, key, batch_size, patience, max_iters,
                                  tolerance):
    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])

    # counter and quiet both start at 0; a quiet streak of `patience` triggers convergence.
    carry = (
        init_centroids,
        init_labels,
        jnp.asarray(0.0),
        jnp.asarray(0),
        jnp.asarray(0),
        key,
    )
    return jax.lax.while_loop(
        cond_fun=lambda c: _minibatch_hartigan_stop_condition(c, patience, max_steps=max_iters),
        body_fun=lambda c: _minibatch_hartigan_step(c, data, batch_size, tolerance),
        init_val=carry,
    )


def run_minibatch_hartigan_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    key: PRNGKeyArray,
    proportion: Float,
    max_iters: Int = 1000,
    tolerance_fraction: Float = 0.01,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    """
    Run k-means clustering using mini-batch Hartigan's algorithm.

    Each pass reassigns a random ``batch_size = ceil(proportion * n)`` subset of the points
    by the Hartigan transfer cost and recomputes every centroid from the full assignment.
    Convergence mirrors the numpy/numba reference (``mini_batch.py``): the run stops after
    ``patience`` CONSECUTIVE quiet passes, where a pass is quiet when it moves at most
    ``tolerance = floor(tolerance_fraction * n)`` sampled points (the quiet counter resets
    the moment a pass moves more than that), and ``patience = ceil(1 / proportion**2)`` for
    ``proportion > 0.33`` and ``10`` otherwise.

    **Arguments**:
        data: A numpy-like array of shape (n, d) containing the data points.
        init_centroids: A numpy-like array of shape (K, d)
                        containing the initial centroids.
        key: A JAX PRNG key driving the per-pass subset sampling.
        proportion: Fraction of points reassigned per pass (0 < proportion <= 1).
        max_iters: maximum number of iterations.
        tolerance_fraction: A pass counts toward the quiet streak when it moves at most
            floor(tolerance_fraction * n) points. Default 0.01 (1% of the data size);
            0.0 recovers the classic zero-move rule.

    **Returns**:
        A tuple containing:
            - centroids: A numpy-like array of shape (K, d) containing the final centroids
            - labels: A numpy-like array of shape (n,) containing the final cluster labels
            - loss: A float representing the final k-means loss.
            - counter: An integer representing the number of iterations performed.
    """
    n = data.shape[0]
    # leave-one-per-cluster constraint: the batch may never contain every point of a cluster
    # (see _sample_batch_holdout), so batch_size <= n - K; patience uses the EFFECTIVE rate.
    K = init_centroids.shape[0]
    batch_size = min(max(1, math.ceil(proportion * n)), max(1, n - K))
    _p_eff = batch_size / n
    patience = math.ceil(1.0 / _p_eff ** 2) if _p_eff > 0.33 else 10
    tolerance = math.floor(tolerance_fraction * n)          # 1% of data size by default

    # data/init/key are TRACED args to the jitted inner (batch_size/patience/max_iters/
    # tolerance are static), so the while_loop compiles once per (shape, batch_size,
    # patience, max_iters, tolerance) and is reused across calls — no per-call
    # recompilation on new data.
    centroids, labels, _loss, counter, _quiet, _key = _run_minibatch_hartigan_inner(
        data, init_centroids, key, batch_size, patience, max_iters, tolerance
    )
    loss = compute_loss(data, centroids, labels)   # computed once from the final state
    return centroids, labels, loss, counter


##################### Certified mini-batch (true-convergence) #####################
# The patience/tolerance stop above is a heuristic: a streak of quiet passes over random
# batches proves nothing about the unsampled points. The certified wrapper below keeps
# the identical mini-batch dynamics but (a) removes tolerance entirely (a pass counts as
# quiet only when it moves NOTHING) and (b) converts the quiet streak into a trigger for
# an exact fixed-point CHECK over all n points; only that check can declare convergence.
# The policy mirrors mini_batch.mini_batch_hartigan_nb_certified (numba) and the
# *_certified entry points in cuda/ — keep the three in sync.


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _resume_minibatch_inner(carry, data, batch_size, patience, max_iters, tolerance):
    """The same while_loop as _run_minibatch_hartigan_inner, but taking the loop carry
    as an argument instead of building it — so the host can pause at a quiet streak,
    run the certificate, and re-enter with the SAME compiled executable (jit compiles
    once per data shape + statics; re-entry costs nothing new)."""
    return jax.lax.while_loop(
        cond_fun=lambda c: _minibatch_hartigan_stop_condition(c, patience,
                                                              max_steps=max_iters),
        body_fun=lambda c: _minibatch_hartigan_step(c, data, batch_size, tolerance),
        init_val=carry,
    )


@jax.jit
def _would_any_move(data, centroids, labels):
    """The convergence certificate as one vectorized op: recompute the populations, ask
    EVERY point where the Hartigan-weighted argmin sends it, and report whether any
    disagree with their current label. Reuses the exact assignment function the
    mini-batch step uses, so the criterion (including the singleton pin) is identical
    by construction. Applies nothing. (No early exit: on an accelerator all n points
    are evaluated in parallel — there is nothing sequential to save.)"""
    populations = jnp.bincount(labels, length=centroids.shape[0])
    new_labels = assign_dp_to_cluster_batched_hartigan(centroids, labels, populations,
                                                       data)
    return jnp.any(new_labels != labels)


def run_minibatch_hartigan_kmeans_certified(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    key: PRNGKeyArray,
    proportion: Float,
    max_iters: Int = 1000,
    init_labels=None,
    *,
    return_postinit: bool = False,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int, bool]:
    """
    Mini-batch Hartigan with a TRUE-convergence certificate. Note there is NO
    tolerance parameter — deliberately: tolerance is a heuristic, and this function's
    contract is exactness. A pass is quiet only when it moves zero points; `patience`
    consecutive quiet passes merely TRIGGER a fixed-point check over all n points
    (~the cost of one proportion=1.0 pass), and convergence is declared only when
    that check finds no would-mover. On a failed check the movers are NOT applied
    (applying them all simultaneously is batch Hartigan, the documented oscillation
    mode) — the mini-batch passes simply resume where they left off, with the same
    PRNG stream (the check consumes no randomness).

    **Returns**: (centroids, labels, loss, counter, certified) — the first four as in
    run_minibatch_hartigan_kmeans; `certified` is True only with a PROVEN fixed point
    (False when the max_iters budget ran out first). `counter` counts mini-batch
    passes only; certificate checks are not passes.

    `return_postinit=True` appends a sixth value: the wall time from the moment the initial
    state is ready (initialization excluded -- the nearest-centroid assignment, or the
    supplied partition's block means) to the end of the last pass, including the certificate
    checks those passes trigger. postinit / counter is then the mean time per mini-batch
    iteration, the same definition numba's mini_batch_hartigan_nb_certified_timed uses. Timing
    reads only: the outputs are identical either way.
    """
    n = data.shape[0]
    # leave-one-per-cluster constraint: the batch may never contain every point of a cluster
    # (see _sample_batch_holdout), so batch_size <= n - K; patience uses the EFFECTIVE rate.
    K = init_centroids.shape[0]
    batch_size = min(max(1, math.ceil(proportion * n)), max(1, n - K))
    _p_eff = batch_size / n
    patience = math.ceil(1.0 / _p_eff ** 2) if _p_eff > 0.33 else 10

    # the same starting state run_minibatch_hartigan_kmeans builds (same jitted
    # helpers => same numbers), as an explicit carry tuple:
    #   (centroids, labels, loss, counter, quiet, key)
    #
    # `init_labels`, when supplied, IS the starting partition: the unweighted nearest-centroid
    # bootstrap is SKIPPED and `init_centroids` serves only to carry K. That bootstrap exists
    # to manufacture a grouping from centroids, because Hartigan's criterion weights by cluster
    # size and cannot be evaluated before one exists; a partition-defined scheme supplies the
    # grouping directly, so running it anyway discards it. See init_schemes.ASSIGNMENT_SCHEMES.
    if init_labels is None:
        init_labels = assign_clusters(init_centroids, data)
    else:
        init_labels = jnp.asarray(init_labels)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])
    carry = (
        init_centroids,
        init_labels,
        jnp.asarray(0.0),
        jnp.asarray(0),
        jnp.asarray(0),
        key,
    )
    if return_postinit:
        jax.block_until_ready(carry)       # initialization finished before the clock starts
        t_post = time.perf_counter()

    certified = False
    while True:
        # run mini-batch passes until a quiet streak or the budget (tolerance=0:
        # quiet passes are zero-move passes)
        carry = _resume_minibatch_inner(carry, data, batch_size, patience, max_iters, 0)
        centroids, labels, loss, counter, quiet, key = carry
        if int(quiet) < patience:
            break                          # budget exhausted without a quiet streak
        if batch_size == n:
            # a zero-move full-batch pass evaluated every point against the fixed
            # pass-start state and moved none: it IS the certificate
            certified = True
            break
        if not bool(_would_any_move(data, centroids, labels)):
            certified = True
            break
        # movers exist somewhere: reset ONLY the quiet counter and resume — the key
        # is part of the carry, so the pass/sample sequence continues exactly where
        # it left off (a certified run is the heuristic run plus extra passes)
        carry = (centroids, labels, loss, counter, jnp.asarray(0), key)

    if return_postinit:
        jax.block_until_ready(carry)
        postinit = time.perf_counter() - t_post
    centroids, labels, _loss, counter, _quiet, _key = carry
    loss = compute_loss(data, centroids, labels)   # same final-state loss convention
    if return_postinit:
        return centroids, labels, loss, counter, certified, postinit
    return centroids, labels, loss, counter, certified
