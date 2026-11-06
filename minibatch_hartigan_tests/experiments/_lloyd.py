from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from _common_functions_for_minibatch import (
    assign_clusters,
    compute_centroids,
    compute_loss,
)


@jax.jit
def _kmeans_step(
    carry: Tuple[
        Float[Array, "K d"],
        Int[Array, " n"],
        Int[Array, " n"],
        Float[Array, " max_steps"],
        Int,
    ],
    data: Float[Array, "n d"],
) -> Tuple[
    Float[Array, "K d"],
    Int[Array, " n"],
    Int[Array, " n"],
    Float[Array, " max_steps"],
    Int,
]:
    centroids, old_labels, _, loss, counter = carry

    labels = assign_clusters(centroids, data)
    centroids = compute_centroids(data, labels, centroids.shape[0])
    # loss carried unchanged; the stop condition never reads it, so it is computed once after
    # the loop (run_lloyd_kmeans) instead of every pass. Callers that want a per-step loss
    # should compute it from the returned (centroids, labels).
    return (centroids, labels, old_labels, loss, counter + 1)


def _kmeans_stop_condition(
    carry: Tuple[
        Float[Array, "K d"],
        Int[Array, " n"],
        Int[Array, " n"],
        Float[Array, " max_steps"],
        Int,
    ],
    max_steps: Int,
) -> Bool:
    _, labels, old_labels, loss, counter = carry

    cond1 = jnp.any(labels != old_labels)
    cond2 = counter <= max_steps

    return cond1 & cond2


@partial(jax.jit, static_argnums=(2,))
def _run_lloyd_inner(data, init_centroids, max_iters):
    init_labels = assign_clusters(init_centroids, data)
    init_centroids = compute_centroids(data, init_labels, init_centroids.shape[0])

    # old_labels seeded to init_labels - 1 so the first pass can't trip the stop condition;
    # loss/counter as concrete jax scalars so the while_loop carry types stay fixed
    # (mirrors _run_minibatch_hartigan_inner).
    carry = (init_centroids, init_labels, init_labels - 1, jnp.asarray(0.0), jnp.asarray(0))
    return jax.lax.while_loop(
        cond_fun=lambda c: _kmeans_stop_condition(c, max_steps=max_iters),
        body_fun=lambda c: _kmeans_step(c, data),
        init_val=carry,
    )


@partial(jax.jit, static_argnums=(2, 3))
def _run_lloyd_inner_from_labels(data, init_labels, n_clusters, max_iters):
    # Starting from a PARTITION. Lloyd's step from a grouping is "recompute the centroids,
    # then reassign", so the partition's block means are the first centroids and the loop's
    # first pass reassigns every point to the nearest of them -- an ordinary Lloyd iteration,
    # counted like the rest. _run_lloyd_inner instead has to manufacture a grouping from
    # centroids first; that assignment is init there and is not counted. Kept as a separate
    # function so the centroid path's traced graph is untouched.
    init_centroids = compute_centroids(data, init_labels, n_clusters)
    carry = (init_centroids, init_labels, init_labels - 1, jnp.asarray(0.0), jnp.asarray(0))
    return jax.lax.while_loop(
        cond_fun=lambda c: _kmeans_stop_condition(c, max_steps=max_iters),
        body_fun=lambda c: _kmeans_step(c, data),
        init_val=carry,
    )


def run_lloyd_kmeans(
    data: Float[Array, "n d"],
    init_centroids: Float[Array, "K d"],
    max_iters: Int = 1000,
    init_labels=None,
) -> Tuple[Float[Array, "K d"], Int[Array, " n"], Float, Int]:
    """
    Run Lloyd's k-means clustering algorithm from a given initial set of cluster centers,
    or from a given initial partition.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        init_centroids: The initial centroids, shape (K, d). When `init_labels` is given it
            only carries K and its values are never read (the convention of
            run_minibatch_hartigan_kmeans_certified).
        max_iters: The maximum number of iterations to run the algorithm.
        init_labels: Optional starting partition, shape (n,), labels in [0, K). When given it
            IS the starting state: the first centroids are its block means, and no
            nearest-centroid assignment runs before the first iteration.
    **Returns:**
        centroids: The final centroids, shape (K, d).
        labels: The cluster labels for each data point, shape (n,).
        loss: The final k-means loss.
        counter: The number of iterations performed. From a partition this counts the first
            reassignment; from centroids the assignment that builds the first grouping is
            init and is not counted.
    """
    # data/init are TRACED args to the jitted inner (max_iters static), so the while_loop
    # compiles once per (shape, max_iters) and is reused across calls — no per-call
    # recompile, and the data array is never baked into the HLO as a constant.
    if init_labels is None:
        centroids, assigments, _, _loss, counter = _run_lloyd_inner(
            data, init_centroids, max_iters)
    else:
        centroids, assigments, _, _loss, counter = _run_lloyd_inner_from_labels(
            data, jnp.asarray(init_labels), init_centroids.shape[0], max_iters)
    loss = compute_loss(data, centroids, assigments)   # computed once from the final state
    return centroids, assigments, loss, counter
