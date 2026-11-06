"""
Core batch Hartigan clustering algorithm.

All functions are pure (no globals). Each takes explicit parameters and returns
explicit results. By convention, every function lists its return values in
alphabetical order by name.
"""

import warnings

import numpy as np

# Suppress "mean of empty slice" warnings that occur when a zero-size cluster is
# detected — this is expected and handled via the zero flag.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def create_rng():
    """Generate a random seed and return it with a numpy Generator.

    Returns (alphabetical):
        rng:  numpy.random.Generator instance
        seed: int used to seed the generator (recorded for reproducibility)
    """
    seed = np.random.randint(0, 414748364)
    rng = np.random.default_rng(seed)
    return rng, seed


def generate_data(rng, cluster_size, amount_of_clusters, dim, intracluster_variance):
    """LEGACY single-blob generator (no ground-truth labels).

    All points are drawn from ONE zero-mean Gaussian, so there is no true cluster
    structure. Kept for reference / comparison with the original results.csv run;
    the v2 simulation uses generate_ground_truth_data instead.

    Args:
        rng:                   numpy Generator
        cluster_size:          number of points per cluster
        amount_of_clusters:    number of clusters
        dim:                   dimensionality of each point
        intracluster_variance: variance of the isotropic Gaussian noise

    Returns (alphabetical):
        centroids:    (amount_of_clusters, dim) mutable working copy of centers
        data:         (cluster_size * amount_of_clusters, dim) array of points
        init_centers: (amount_of_clusters, dim) immutable record of the centers
    """
    size = cluster_size * amount_of_clusters

    cov = intracluster_variance * np.identity(dim)
    data = rng.multivariate_normal(np.zeros(dim), cov, size=size)

    indices = rng.choice(size, size=amount_of_clusters, replace=False)
    centroids = data[indices].copy()
    init_centers = centroids.copy()

    return centroids, data, init_centers


def generate_ground_truth_data(rng, cluster_size, amount_of_clusters, dim,
                               intracluster_variance, intercluster_variance):
    """Sample data from KNOWN cluster centers, returning the true labels.

    Each of `amount_of_clusters` centers is drawn from a zero-mean Gaussian with
    variance `intercluster_variance`; then `cluster_size` points are drawn around
    each center with variance `intracluster_variance`. The ratio
    intercluster_variance / intracluster_variance controls how separable the
    clusters are.

    The returned `labels` give the true cluster membership of every row of `data`
    (labels[i] is the cluster that generated data[i]).

    Args:
        rng:                   numpy Generator
        cluster_size:          number of points per cluster
        amount_of_clusters:    number of clusters
        dim:                   dimensionality of each point
        intracluster_variance: within-cluster (intra) variance
        intercluster_variance: variance of the cluster-center distribution (inter)

    Returns (alphabetical):
        centroids:    (amount_of_clusters, dim) mutable working copy of centers
        data:         (cluster_size * amount_of_clusters, dim) array of points
        init_centers: (amount_of_clusters, dim) immutable record of the centers
        labels:       (cluster_size * amount_of_clusters,) true cluster ids
    """
    # np.errstate guards against spurious "divide/overflow/invalid encountered in
    # matmul" warnings that Apple's Accelerate BLAS emits from multivariate_normal
    # under multiprocessing on macOS; the sampled values are always finite
    # (verified), so ignoring the FP flags here is safe and a no-op on Linux.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        centers = rng.multivariate_normal(
            np.zeros(dim), intercluster_variance * np.identity(dim),
            size=amount_of_clusters)

        data = []
        labels = []
        cov = intracluster_variance * np.identity(dim)
        for k, c in enumerate(centers):
            pts = rng.multivariate_normal(c, cov, size=cluster_size)
            data.append(pts)                  # each block is (cluster_size, dim)
            labels.extend([k] * cluster_size)
    data = np.vstack(data)                    # stack blocks -> (n, dim) matrix
    labels = np.array(labels)                 # labels[i] = true cluster of data[i]

    # Initial centers: distinct random data points (same scheme as the legacy
    # generator, so loop behavior is comparable).
    indices = rng.choice(len(data), size=amount_of_clusters, replace=False)
    centroids = data[indices].copy()
    init_centers = centroids.copy()

    return centroids, data, init_centers, labels


def compare_init(data, centroids):
    """Assign every point to its nearest centroid (squared Euclidean).

    Args:
        data:      (n, dim) array of points
        centroids: (amount_of_clusters, dim) array of cluster centers

    Returns:
        assigns: (n,) array of cluster indices
    """
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1) ** 2
    return distances.argmin(-1)


def compare_batch_hart_reference(data, centroids, assigns, amount_of_clusters, zero):
    """One full batch-Hartigan reassignment pass — REFERENCE (loop) implementation.

    Kept only for the equivalence test and as a one-line rollback; the sweep uses
    the vectorized `compare_batch_hart` below. For each point, compute the
    Hartigan transfer cost to every cluster, reassign to the cheapest, then
    recompute centroids. If any cluster has population 0, set the zero flag.

    Args:
        data:               (n, dim) data matrix
        centroids:          (amount_of_clusters, dim) current centers (mutated)
        assigns:            (n,) current assignments
        amount_of_clusters: number of clusters
        zero:               bool flag — True if an empty cluster was already seen

    Returns (alphabetical):
        assigns: (n,) updated assignments
        zero:    bool — True if an empty cluster was detected
    """
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1) ** 2
    populations = np.bincount(assigns, minlength=amount_of_clusters)

    new_assigns = []
    for i in range(len(data)):
        for k in range(amount_of_clusters):
            d = distances[i, k]

            # Empty cluster is a real failure — report it
            if populations[k] == 0:
                zero = True
                break

            if assigns[i] == k:
                # Cost of staying: scale by n_k / (n_k - 1)
                if populations[k] == 1:
                    d = 0
                else:
                    d = populations[k] / (populations[k] - 1) * d
            else:
                # Cost of moving: scale by n_k / (n_k + 1)
                d = populations[k] / (populations[k] + 1) * d

            distances[i, k] = d

        new_assigns.append(distances[i].argmin())

    assigns = np.array(new_assigns)

    # Recompute centroids from new assignments
    for k in range(amount_of_clusters):
        centroids[k] = np.mean(data[assigns == k], axis=0)

    return assigns, zero


def compare_batch_hart(data, centroids, assigns, amount_of_clusters, zero):
    """One full batch-Hartigan reassignment pass (VECTORIZED).

    Scale each point's squared distance to every cluster by the Hartigan transfer
    cost (n_k/(n_k-1) to stay in its current cluster, n_k/(n_k+1) to move to
    another), reassign each point to the cheapest cluster, then recompute
    centroids. Vectorized equivalent of compare_batch_hart_reference: produces
    bit-identical assignments whenever no cluster is empty. If a cluster is empty
    the `zero` flag is set and the caller (batch_hartigan) returns immediately,
    discarding this pass's assignment — so the assignment in that case need not
    match the reference exactly.

    Args:
        data:               (n, dim) data matrix
        centroids:          (amount_of_clusters, dim) current centers (mutated)
        assigns:            (n,) current assignments
        amount_of_clusters: number of clusters
        zero:               bool flag — True if an empty cluster was already seen

    Returns (alphabetical):
        assigns: (n,) updated assignments
        zero:    bool — True if an empty cluster was detected
    """
    n = len(data)
    # Identical to the reference distance computation -> bit-identical distances.
    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=-1) ** 2
    populations = np.bincount(assigns, minlength=amount_of_clusters)

    # Empty cluster is a real failure. The caller returns immediately on zero and
    # discards this assignment, so only the flag must be correct here.
    if (populations == 0).any():
        zero = True

    pop = populations.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        move_factor = pop / (pop + 1.0)                          # n_k / (n_k + 1)
        stay_factor = np.where(pop == 1, 0.0, pop / (pop - 1.0))  # n_k / (n_k - 1), 0 if alone

    # Per-point cost-scaling row: move_factor for every cluster, with the point's
    # CURRENT cluster overwritten by its stay_factor.
    scale = np.broadcast_to(move_factor, (n, amount_of_clusters)).copy()
    scale[np.arange(n), assigns] = stay_factor[assigns]

    assigns = (distances * scale).argmin(axis=1)

    # Recompute centroids from the new assignments.
    for k in range(amount_of_clusters):
        centroids[k] = np.mean(data[assigns == k], axis=0)

    return assigns, zero


def loop_checker(assigns, memory, counter):
    """Detect whether this assignment has been seen before, and if so, how many
    iterations ago (the size of the cycle).

    Only the HASH of each assignment vector is stored (via `tobytes()`), not the
    vector itself — Python compares hashes directly, so detection is fast and
    memory-light. The chance of a false positive from a hash collision is about
    1 in 2**64, negligible across an entire sweep.

    A converged fixed point is NOT a loop here: the caller checks convergence
    before calling loop_checker, so `loop_checker` only ever fires for genuine
    oscillating cycles (loop_size >= 2).

    Args:
        assigns: (n,) current cluster assignments
        memory:  dict mapping assignment-hash -> iteration index first seen
        counter: current iteration index

    Returns (alphabetical):
        loop:      True if assigns was already in memory
        loop_size: number of iterations in the cycle (0 if no loop)
        memory:    updated dict
    """
    h = hash(assigns.tobytes())
    if h in memory:
        return True, counter - memory[h], memory
    memory[h] = counter
    return False, 0, memory


def batch_hartigan(data, init_centers, centroids, amount_of_clusters, max_iter=500):
    """Run the batch Hartigan algorithm until convergence, loop, empty cluster,
    or the iteration budget.

    Args:
        data:               (n, dim) data matrix
        init_centers:       (amount_of_clusters, dim) initial centers (record)
        centroids:          (amount_of_clusters, dim) mutable working centers
        amount_of_clusters: number of clusters
        max_iter:           maximum reassignment iterations (default 500)

    Returns (alphabetical):
        assigns:   (n,) final assignments
        centroids: (amount_of_clusters, dim) final centers
        counter:   iterations performed before stopping
        loop:      True if a repeated assignment pattern was detected
        loop_size: length of the detected cycle (0 if none)
        zero:      True if an empty cluster was encountered
    """
    zero = False
    memory = {}

    size = len(data)

    # Initial assignment: each point to its nearest initial center
    assigns = compare_init(data, init_centers)
    old_assigns = np.full(size, -1)
    counter = 0

    # Compute centroids from the initial assignment
    for k in range(amount_of_clusters):
        centroids[k] = np.mean(data[assigns == k], axis=0)

    while counter < max_iter:
        # Clean convergence (a fixed point) is checked BEFORE loop detection, so
        # a converged state is never mistaken for a size-1 loop.
        if np.array_equal(assigns, old_assigns):
            return assigns, centroids, counter, False, 0, zero

        loop, loop_size, memory = loop_checker(assigns, memory, counter)
        if loop:
            # Stop the moment a real cycle is detected and report its size (no
            # need to spin out to max_iter).
            return assigns, centroids, counter, True, loop_size, zero

        old_assigns = assigns.copy()
        assigns, zero = compare_batch_hart(
            data, centroids, assigns, amount_of_clusters, zero)
        counter += 1

        if zero:
            return assigns, centroids, counter, False, 0, True

    # Budget exhausted without converging or repeating a state.
    return assigns, centroids, counter, False, 0, zero

rng, seed = create_rng()
centroids, data, init_centers, labels = generate_ground_truth_data(
    rng, cluster_size=100, amount_of_clusters=5, dim=2,
    intracluster_variance=1.0, intercluster_variance=1.0)
assigns, centroids, counter, loop, loop_size, zero = batch_hartigan(
    data, init_centers, centroids, amount_of_clusters=5, max_iter=500)
