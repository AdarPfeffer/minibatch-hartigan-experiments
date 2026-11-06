from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


@jax.jit
def compute_loss(
    data: Float[Array, "n d"],
    centroids: Float[Array, "K d"],
    labels: Int[Array, " n"],
) -> Float:
    """
    Compute the loss function for k-means clustering.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        centroids: The centroids of the clusters, shape (K, d).
        labels: The cluster labels for each data point, shape (n,).
    **Returns:**
        loss: The loss function value.
    """
    return jnp.sum(jnp.square(data - centroids[labels]))


@jax.jit
def assign_clusters(
    centroids: Float[Array, "K d"], data: Float[Array, "n d"]
) -> Int[Array, " n"]:
    """
    Assign each data point to the nearest centroid.

    **Arguments:**
        centroids: The centroids of the clusters, shape (K, d).
        data: The data to cluster, shape (n, d).
    **Returns:**
        labels: The cluster labels for each data point, shape (n,).
    """
    # Nearest-centroid argmin via the matmul expansion ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2.
    # This avoids the (n, K, d) broadcast tensor and runs as a GEMM. Two argmin-invariant
    # simplifications: ||x||^2 is constant across clusters (dropped) and sqrt is monotonic
    # (dropped). precision="highest" forces true float32 (no TF32) so the argmin matches the
    # direct distance to float rounding.
    cn = jnp.sum(centroids * centroids, axis=1)                       # (K,) ||c_k||^2
    xc = jnp.matmul(data, centroids.T, precision="highest")          # (n, K) x . c_k
    return jnp.argmin(cn[None, :] - 2.0 * xc, axis=1)


@partial(jax.jit, static_argnums=(2,))
def compute_centroids(
    data: Float[Array, "n d"], labels: Int[Array, " n"], n_clusters: Int
) -> Float[Array, "K d"]:
    """
    Update the centroids of the clusters, given the data and cluster labels.

    **Arguments:**
        data: The data to cluster, shape (n, d).
        labels: The cluster labels for each data point, shape (n,).
        n_clusters: The number of clusters.
    **Returns:**
        centroids: The updated centroids, shape (K, d).
    """

    # One-hot matmul: sums = onehot^T @ data is a single deterministic GEMM at full
    # float32 precision, instead of K separate masked reductions each scanning all n points
    # (O(n*d) vs O(K*n*d)). An empty cluster has count 0 -> sum 0 -> 0/max(0,1) = 0 vector,
    # matching the previous 0 / (0 + 1e-16) = 0 convention.
    onehot = (labels[:, None] == jnp.arange(n_clusters)[None, :]).astype(data.dtype)  # (n, K)
    sums = jnp.matmul(onehot.T, data, precision="highest")                            # (K, d)
    counts = jnp.sum(onehot, axis=0)                                                  # (K,)
    return sums / jnp.maximum(counts, 1.0)[:, None]
