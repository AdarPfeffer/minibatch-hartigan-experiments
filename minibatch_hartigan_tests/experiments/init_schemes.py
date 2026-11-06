"""Initialization schemes for the best-of-N initialization benchmark.

Every scheme deterministically selects K initial centroids (a K x d float64 array) from a
GIVEN dataset and an integer seed. Determinism + a seed keyed by (scheme, d, N, K,
dataset_id, init_id) is what makes the SELECTION reproducible and identical across every
backend (numba / JAX / CUDA / Fortran all regenerate the same init), so the (timed)
selection is a legitimate, comparable part of runtime.

IMPORTANT (per the study spec): "random centroids" means K data SAMPLES are randomly
selected AS the initial centroids -- NOT that random vectors are drawn as centroids.

The selection is timed as part of `convergence` (see bestof_init_benchmark.py), so this
function must contain exactly the selection work and nothing else. It returns float64;
each backend casts (float32 for CUDA, jnp for JAX, column-major for Fortran) inside its own
timed region, so per-backend cast/transfer overheads are attributed to that backend.

Only `random_centroid` is exercised by the first (random-centroid) job. `random_partition`
and `kmeanspp` are implemented in their own follow-up jobs (stubs below document the exact
intended construction + guards), run after the random-centroid pipeline is validated.
"""
import numpy as np

SCHEMES = ("random_centroid", "random_partition", "kmeanspp")

# Schemes that are defined as a PARTITION of the index set rather than as a set of centroids.
# These go through `select_init_partition` and enter each algorithm via its `init_labels=`
# parameter, NOT through `select_init_centroids`.
#
# WHY THIS DISTINCTION EXISTS (2026-09-16). Hartigan's criterion weights each distance by the
# cluster sizes -- stay costs n/(n-1), move costs n/(n+1) -- so it cannot be evaluated until a
# grouping exists. An algorithm handed only centroids therefore has to manufacture one first,
# with a single UNWEIGHTED nearest-centroid pass; AS-136 does exactly this in KMNS's first loop
# before it can set AN1/AN2, and so do our numba and JAX entry points.
#
# For `random_centroid` that pass is unavoidable: K data points are not a grouping, and there
# is no other way to obtain cluster sizes. For `random_partition` it is NOT needed -- the
# grouping and its sizes are given -- and running it anyway DISCARDS the specified partition,
# replacing it with the nearest-centroid cells of K block means that all sit within a hair of
# the global mean. Measured: the recovered grouping agrees with the specified one at chance.
#
# So a partition-defined scheme must enter as a partition. AS-136 cannot (its KMNS overwrites
# `ic1` before reading it), which is why the Fortran arms are random_centroid only.
ASSIGNMENT_SCHEMES = frozenset({"random_partition"})


def select_init_centroids(scheme, data, K, seed):
    """Return a K x d float64 array of initial centroids, deterministic in (data, K, seed).

    Args:
        scheme: one of SCHEMES.
        data:   (n, d) dataset (float64); GIVEN, never modified.
        K:      number of clusters (1 <= K <= n).
        seed:   integer seed; the SAME seed reproduces the SAME centroids in every backend.

    Returns:
        (K, d) float64, C-contiguous.
    """
    n = data.shape[0]
    if K < 1 or K > n:
        raise ValueError(f"need 1 <= K <= n, got K={K}, n={n}")
    rng = np.random.default_rng(seed)

    if scheme == "random_centroid":
        # K distinct data points chosen uniformly at random, used verbatim as centroids.
        # Mirrors the existing benchmark's init (benchmark_new_impls.py) so the two studies
        # are directly comparable. replace=False => K distinct rows.
        idx = rng.choice(n, size=K, replace=False)
        return np.ascontiguousarray(data[idx], dtype=np.float64)

    if scheme in ASSIGNMENT_SCHEMES:
        # Deliberately NOT served here. Until 2026-09-16 this returned the block means, and
        # every caller then fed them to an algorithm whose first act is an unweighted
        # nearest-centroid pass -- which discarded the partition and replaced it with the
        # Voronoi cells of K nearly-coincident centroids (agreement with the specified
        # partition: at chance). Raising is what makes that failure loud instead of silent.
        raise ValueError(
            f"{scheme!r} is a PARTITION-defined scheme: call select_init_partition() and pass "
            f"the result to the algorithm's init_labels= parameter. See ASSIGNMENT_SCHEMES.")

    if scheme == "kmeanspp":
        # FOLLOW-UP JOB (not run here). Construction: k-means++ D^2 seeding, numpy-Generator
        # pure -- pick center 0 uniformly, then each next center with prob proportional to
        # the squared distance to the nearest chosen center (guard D2.sum()==0 for heavy
        # duplicates). Kept off sklearn to stay on the single Generator and avoid a hot-path
        # dependency in the numba/CUDA paths.
        raise NotImplementedError(
            "kmeanspp is a separate follow-up job; implemented + audited then")

    raise ValueError(f"unknown init scheme {scheme!r}; expected one of {SCHEMES}")


def select_init_partition(scheme, data, K, seed):
    """Return an (n,) int64 array of initial cluster labels, deterministic in (data, K, seed).

    For PARTITION-defined schemes (ASSIGNMENT_SCHEMES). The algorithms take this through their
    `init_labels=` parameter and compute the centroids from it themselves, exactly as they
    would have computed them from any grouping — so no unweighted nearest-centroid pass runs,
    and the partition the caller specified is the one the Hartigan sweeps start from.

    `random_partition`: shuffle the index set, cut it into K contiguous blocks, block k is
    cluster k. `array_split` gives the first (n mod K) blocks one extra index, so sizes differ
    by at most 1 and no cluster is empty for any K <= n.

    Drawn from the same `default_rng(seed).permutation(n)` the centroid form used before
    2026-09-16, so a cell's partition here is the one whose block means that code averaged.

    Computed once on the CPU and handed to every backend, so all backends start from the
    identical grouping — the integer labels transfer exactly, with none of the last-bit drift
    that per-backend mean computation would introduce.
    """
    n = data.shape[0]
    if K < 1 or K > n:
        raise ValueError(f"need 1 <= K <= n, got K={K}, n={n}")
    if scheme not in ASSIGNMENT_SCHEMES:
        raise ValueError(
            f"{scheme!r} is not a partition-defined scheme; use select_init_centroids(). "
            f"Partition-defined schemes: {sorted(ASSIGNMENT_SCHEMES)}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=np.int64)
    for k, block in enumerate(np.array_split(perm, K)):
        labels[block] = k
    return labels
