"""Tests for the JAX Hartigan implementations.

LOCAL tests (CPU jax) cover the mini-batch patience port and the Lloyd control.
GPU tests cover the backend, Lloyd and the certified mini-batch driver on the GPU and the
matmul precision; they auto-skip when no GPU is visible to JAX.

    python -m pytest minibatch_hartigan_tests/experiments/tests/test_jax_hartigan.py -v
    python minibatch_hartigan_tests/experiments/tests/test_jax_hartigan.py   # local set only
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # experiments/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                                "batch_hartigan_tests", "experiments")))

import numpy as np
import jax
import jax.numpy as jnp

from jax_minibatch_hartigan import (
    run_minibatch_hartigan_kmeans as gpu_mb,
    _minibatch_hartigan_stop_condition,
)
from _lloyd import run_lloyd_kmeans as gpu_lloyd

PROPORTIONS = [0.25, 0.5, 0.75, 0.9, 0.95]


def _patience(p):
    return math.ceil(1.0 / p ** 2) if p > 0.33 else 10


def _blobs(rng, K=4, d=8, per=60, spread=12.0):
    centers = rng.normal(size=(K, d)) * spread
    data = np.vstack([centers[k] + rng.normal(size=(per, d)) for k in range(K)]).astype(np.float64)
    init = data[rng.choice(data.shape[0], size=K, replace=False)].copy()
    return data, init, K


def _on_gpu():
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


try:
    import pytest
    requires_gpu = pytest.mark.skipif(not _on_gpu(), reason="needs a GPU visible to JAX")
except ImportError:                       # script mode: define a no-op decorator
    def requires_gpu(fn):
        return fn


# ---------------------------- LOCAL: mini-batch patience ----------------------------

def test_patience_formula():
    assert _patience(0.25) == 10
    assert _patience(0.33) == 10          # boundary: not > 0.33
    assert _patience(0.34) == math.ceil(1.0 / 0.34 ** 2)
    assert _patience(0.5) == 4
    assert _patience(0.75) == 2
    assert _patience(0.9) == 2
    assert _patience(0.95) == 2


def test_stop_condition_logic():
    def carry(counter, quiet):
        return (0, 0, 0, jnp.asarray(counter), jnp.asarray(quiet), 0)
    sc = _minibatch_hartigan_stop_condition
    assert bool(sc(carry(0, 0), patience=10, max_steps=100))       # fresh -> continue
    assert bool(sc(carry(5, 9), patience=10, max_steps=100))       # quiet<patience -> continue
    assert not bool(sc(carry(5, 10), patience=10, max_steps=100))  # quiet==patience -> stop
    assert not bool(sc(carry(101, 0), patience=10, max_steps=100)) # over budget -> stop


def test_patience_engaged_lower_bounds_counter():
    # To satisfy quiet >= patience the run must complete `patience` consecutive quiet passes,
    # each incrementing counter -> a converged run has counter >= patience. Directly proves
    # patience is enforced (the old single-pass stop would allow counter as low as 1).
    rng = np.random.default_rng(5)
    data, init, K = _blobs(rng, per=100)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    for p in PROPORTIONS:
        for s in range(4):
            counter = int(gpu_mb(dj, ij, jax.random.PRNGKey(s), p, 10000)[3])
            if counter < 10000:                 # converged, not budget-capped
                assert counter >= _patience(p)


def test_minibatch_converges_and_valid():
    rng = np.random.default_rng(0)
    data, init, K = _blobs(rng)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    for p in PROPORTIONS:
        c, lbl, loss, counter = gpu_mb(dj, ij, jax.random.PRNGKey(0), p, 10000)
        lbl = np.asarray(lbl)
        assert int(counter) < 10000
        assert lbl.min() >= 0 and lbl.max() < K
        assert np.isfinite(float(loss)) and float(loss) > 0.0
        assert tuple(c.shape) == (K, data.shape[1])


def test_determinism():
    rng = np.random.default_rng(1)
    data, init, K = _blobs(rng)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    key = jax.random.PRNGKey(7)
    a = gpu_mb(dj, ij, key, 0.5, 10000)
    b = gpu_mb(dj, ij, key, 0.5, 10000)
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
    assert int(a[3]) == int(b[3])
    assert np.allclose(np.asarray(a[0]), np.asarray(b[0]))


def test_parity_with_numba_reference():
    # jax random.choice != numpy rng.choice, so iteration counts differ; final loss magnitudes
    # should be close on well-separated data.
    from mini_batch import mini_batch_hartigan_nb
    rng = np.random.default_rng(3)
    data, init, K = _blobs(rng, per=80)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    for p in [0.5, 0.9]:
        assigns_nb, cents_nb, _, _ = mini_batch_hartigan_nb(
            data, init, init.copy(), K, np.random.default_rng(3), p, 10000)
        loss_nb = float(np.sum((data - cents_nb[assigns_nb]) ** 2))
        _, _, loss_jax, _ = gpu_mb(dj, ij, jax.random.PRNGKey(0), p, 10000)
        assert abs(float(loss_jax) - loss_nb) <= 0.30 * loss_nb + 1.0


def test_float32_default():
    rng = np.random.default_rng(0)
    data, init, K = _blobs(rng)
    c, _, _, _ = gpu_mb(jnp.asarray(data), jnp.asarray(init), jax.random.PRNGKey(0), 0.5, 10000)
    assert c.dtype == jnp.float32          # x64 disabled by default


# ------------------------------------- GPU tests -------------------------------------

@requires_gpu
def test_gpu_backend():
    assert jax.default_backend() == "gpu"
    assert any(getattr(d, "platform", "") == "gpu" for d in jax.devices())


@requires_gpu
def test_minibatch_certified_gpu():
    # Certified wrapper on the GPU: deterministic, certifies on separated blobs, and the
    # result passes an INDEPENDENT numpy fixed-point check (no shared code path).
    from jax_minibatch_hartigan import (
        run_minibatch_hartigan_kmeans_certified,
    )
    rng = np.random.default_rng(15)
    data, init, K = _blobs(rng, per=100)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    for p in [0.5, 0.9]:
        key = jax.random.PRNGKey(2)
        c1, l1, _, n1, cert1 = run_minibatch_hartigan_kmeans_certified(dj, ij, key, p, 10000)
        c2, l2, _, n2, cert2 = run_minibatch_hartigan_kmeans_certified(dj, ij, key, p, 10000)
        assert np.array_equal(np.asarray(l1), np.asarray(l2))
        assert int(n1) == int(n2) and cert1 == cert2
        assert cert1, f"p={p}: did not certify on well-separated blobs"
        lab = np.asarray(l1)
        cen = np.asarray(c1, dtype=np.float64)
        dat = np.asarray(dj, dtype=np.float64)
        pops = np.bincount(lab, minlength=K).astype(np.float64)
        sq = ((dat[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        w = sq * (pops / (pops + 1.0))[None, :]
        rows = np.arange(dat.shape[0])
        own = np.where(pops[lab] <= 1, -1.0,
                       sq[rows, lab] * (pops[lab] / np.maximum(pops[lab] - 1.0, 1.0)))
        w[rows, lab] = own
        assert np.array_equal(np.argmin(w, axis=1), lab), f"p={p}: not a fixed point"


# ------------------------- LOCAL: tolerance convergence rule -------------------------

def test_default_tolerance_semantics():
    # Explicit tolerance_fraction=0.01 must be bit-identical to omitting it; and for small n
    # where floor(0.01 * n) == 0, the default must equal the classic zero-move rule.
    rng = np.random.default_rng(11)
    data, init, K = _blobs(rng)                    # n = 240 -> tolerance = 2
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    key = jax.random.PRNGKey(3)
    a = gpu_mb(dj, ij, key, 0.5, 10000)
    b = gpu_mb(dj, ij, key, 0.5, 10000, tolerance_fraction=0.01)
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1])) and int(a[3]) == int(b[3])

    small, sinit, _ = _blobs(rng, per=20)          # n = 80 -> floor(0.8) = 0
    sj, sij = jnp.asarray(small), jnp.asarray(sinit)
    c = gpu_mb(sj, sij, key, 0.5, 10000)
    d = gpu_mb(sj, sij, key, 0.5, 10000, tolerance_fraction=0.0)
    assert np.array_equal(np.asarray(c[1]), np.asarray(d[1])) and int(c[3]) == int(d[3])


def test_tolerance_reduces_iterations():
    # Same data/init/key: counter(frac=0.01) <= counter(frac=0.0) always (same key => same
    # subsets => identical passes until the tolerant run stops first), strictly less somewhere.
    rng = np.random.default_rng(12)
    data, init, K = _blobs(rng, per=100, spread=3.0)   # overlapping -> boundary churn
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    strictly_less = False
    for p in PROPORTIONS:
        for s in range(3):
            key = jax.random.PRNGKey(s)
            c0 = int(gpu_mb(dj, ij, key, p, 10000, tolerance_fraction=0.0)[3])
            c1 = int(gpu_mb(dj, ij, key, p, 10000)[3])
            assert c1 <= c0            # relaxing the rule can never ADD iterations
            strictly_less |= c1 < c0
    assert strictly_less


def test_quality_vs_streamed():
    # Mini-batch (default 1% tolerance) must not be more than 5% WORSE than streamed
    # Hartigan on the same data/init. One-sided: lower loss than streamed also passes.
    # Best-of-3 keys per proportion to damp restart luck (matches n_init usage).
    from streamed_hartigan import run_hartigan_kmeans as streamed_nb
    rng = np.random.default_rng(13)
    data, init, K = _blobs(rng, per=100)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    _, _, loss_streamed, _ = streamed_nb(data, init.copy(), max_iters=10000,
                                         checks_convergence=True)
    loss_streamed = float(loss_streamed)
    for p in PROPORTIONS:
        loss_mb = min(float(gpu_mb(dj, ij, jax.random.PRNGKey(s), p, 10000)[2])
                      for s in range(3))
        assert loss_mb <= 1.05 * loss_streamed, (
            f"p={p}: mini-batch loss {loss_mb:.1f} is >5% above streamed {loss_streamed:.1f}")


def test_numpy_numba_tolerance_parity():
    # numpy and numba drivers share the same rng sampling path -> identical results.
    from mini_batch import mini_batch_hartigan, mini_batch_hartigan_nb
    rng = np.random.default_rng(14)
    data, init, K = _blobs(rng, per=60)
    for frac in (0.0, 0.01):
        a = mini_batch_hartigan(data, init, init.copy(), K,
                                np.random.default_rng(5), 0.5, 10000, tolerance_fraction=frac)
        b = mini_batch_hartigan_nb(data, init, init.copy(), K,
                                   np.random.default_rng(5), 0.5, 10000, tolerance_fraction=frac)
        assert np.array_equal(a[0], b[0]) and a[2] == b[2]


# ------------------------------ LOCAL: Lloyd (control) ------------------------------

def test_lloyd_stop_condition_logic():
    from _lloyd import _kmeans_stop_condition as sc
    def carry(labels, old_labels, counter):
        # carry layout: (centroids, labels, old_labels, loss, counter); sc reads [1], [2], [4]
        return (0, jnp.asarray(labels), jnp.asarray(old_labels), jnp.asarray(0.0),
                jnp.asarray(counter))
    assert bool(sc(carry([0, 1, 2], [2, 1, 0], 3), max_steps=100))       # labels changed -> continue
    assert not bool(sc(carry([0, 1, 2], [0, 1, 2], 3), max_steps=100))   # labels stable -> stop
    assert not bool(sc(carry([0, 1, 2], [2, 1, 0], 101), max_steps=100)) # over budget -> stop


def test_lloyd_converges_and_valid():
    rng = np.random.default_rng(0)
    data, init, K = _blobs(rng)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    c, lbl, loss, counter = gpu_lloyd(dj, ij, 10000)
    lbl = np.asarray(lbl)
    assert int(counter) < 10000                       # converged, not budget-capped
    assert lbl.min() >= 0 and lbl.max() < K
    assert np.isfinite(float(loss)) and float(loss) > 0.0
    assert tuple(c.shape) == (K, data.shape[1])


def test_lloyd_determinism():
    rng = np.random.default_rng(1)
    data, init, K = _blobs(rng)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    a = gpu_lloyd(dj, ij, 10000)
    b = gpu_lloyd(dj, ij, 10000)
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
    assert int(a[3]) == int(b[3])
    assert np.allclose(np.asarray(a[0]), np.asarray(b[0]))


def test_lloyd_float32_default():
    rng = np.random.default_rng(0)
    data, init, K = _blobs(rng)
    c, _, _, _ = gpu_lloyd(jnp.asarray(data), jnp.asarray(init), 10000)
    assert c.dtype == jnp.float32                     # x64 disabled by default


def test_lloyd_reaches_fixed_point():
    # At Lloyd convergence the returned (centroids, labels) is a true fixed point: reassigning
    # every point to the returned centroids reproduces the labels, and recomputing centroids
    # from those labels reproduces the centroids. Directly validates the while_loop stop
    # semantics (it halts only when assignments stop changing).
    from _common_functions_for_minibatch import assign_clusters, compute_centroids
    rng = np.random.default_rng(2)
    data, init, K = _blobs(rng)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    c, lbl, _, counter = gpu_lloyd(dj, ij, 10000)
    assert int(counter) < 10000
    re_lbl = np.asarray(assign_clusters(c, dj))
    assert np.array_equal(re_lbl, np.asarray(lbl))
    re_c = np.asarray(compute_centroids(dj, jnp.asarray(lbl), K))
    assert np.allclose(re_c, np.asarray(c), atol=1e-3)


def test_lloyd_loss_monotone_non_increasing():
    # Lloyd's defining guarantee: each pass (assign to nearest, then recompute means) can only
    # lower or hold the SSE, never raise it. Step the jax kernel directly and check the loss is
    # non-increasing pass to pass. Overlapping blobs so several non-trivial passes happen.
    from _lloyd import _kmeans_step
    from _common_functions_for_minibatch import (
        assign_clusters, compute_centroids, compute_loss)
    rng = np.random.default_rng(4)
    data, init, K = _blobs(rng, per=100, spread=3.0)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    labels = assign_clusters(ij, dj)
    carry = (compute_centroids(dj, labels, K), labels, labels - 1, jnp.asarray(0.0), jnp.asarray(0))
    prev = np.inf
    for _ in range(25):
        carry = _kmeans_step(carry, dj)
        # loss is no longer carried per step (hoisted out of the loop); compute it from the
        # step's returned centroids+labels — same value, preserves the monotonicity check.
        loss = float(compute_loss(dj, carry[0], carry[1]))
        assert loss <= prev * (1 + 1e-4) + 1e-3       # never rises (small float32 slack)
        prev = loss


def test_lloyd_recovers_wellseparated_clusters():
    # On well-separated blobs Lloyd should recover all K clusters (mirrors the streamed test).
    rng = np.random.default_rng(1)
    data, init, K = _blobs(rng, spread=20.0)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    _, lbl, _, _ = gpu_lloyd(dj, ij, 10000)
    assert len(np.unique(np.asarray(lbl))) == K


# -------------------------------- GPU: Lloyd on GPU --------------------------------

@requires_gpu
def test_lloyd_gpu_runs_and_recovers():
    rng = np.random.default_rng(8)
    data, init, K = _blobs(rng, spread=20.0)
    dj, ij = jnp.asarray(data), jnp.asarray(init)
    res = gpu_lloyd(dj, ij, 10000)
    jax.block_until_ready(res)
    c, lbl, loss, counter = res
    assert c.dtype == jnp.float32
    assert len(np.unique(np.asarray(lbl))) == K
    assert np.isfinite(float(loss)) and float(loss) > 0.0
    assert int(counter) >= 1


# -------------------- LOCAL: optimization regression tests (matmul / one-hot) --------------------

def test_assign_clusters_matmul_matches_direct():
    # The matmul-expansion assign_clusters (||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2) must give the
    # same nearest-centroid labels as a direct ||x-c||^2 computation. Guards the distance refactor;
    # a sign/transpose/factor bug would collapse agreement toward 1/K.
    from _common_functions_for_minibatch import assign_clusters
    rng = np.random.default_rng(20)
    for K, d, n in [(4, 8, 200), (8, 100, 500), (16, 1000, 300)]:
        data = (rng.normal(size=(n, d)) * 5.0)
        cents = data[rng.choice(n, K, replace=False)].copy()
        got = np.asarray(assign_clusters(jnp.asarray(cents), jnp.asarray(data)))
        ref = np.argmin(((data[:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)  # float64 direct
        agree = np.mean(got == ref)
        assert agree >= 0.99, f"K={K} d={d}: matmul-assign agrees only {agree:.3f} with direct argmin"


def test_one_hot_centroids_match_group_means():
    # The one-hot-matmul compute_centroids must equal the exact per-cluster means, and an empty
    # cluster must map to the zero vector (the shared convention).
    from _common_functions_for_minibatch import compute_centroids
    rng = np.random.default_rng(21)
    n, d, K = 300, 16, 8
    data = rng.normal(size=(n, d))
    labels = rng.integers(0, K, size=n)
    labels[labels == 5] = 0                       # force cluster 5 empty
    got = np.asarray(compute_centroids(jnp.asarray(data), jnp.asarray(labels), K))
    ref = np.zeros((K, d))
    for k in range(K):
        m = labels == k
        if m.any():
            ref[k] = data[m].mean(0)
    assert np.allclose(got, ref, atol=1e-4), "one-hot centroids != group means"
    assert np.allclose(got[5], 0.0), "empty cluster must be the zero vector"


# -------------------- GPU: optimization regression tests --------------------

@requires_gpu
def test_matmul_precision_not_tf32():
    # The distance/centroid matmuls pin precision="highest". Verify the GPU centroid matmul matches
    # a float64 reference to ~1e-4 relative, NOT the ~1e-3 that TF32 (JAX's GPU default for float32
    # matmuls) would give. Large-magnitude data stresses the mantissa. Fails if a future edit drops
    # precision="highest" (which would silently flip boundary assignments).
    from _common_functions_for_minibatch import compute_centroids
    rng = np.random.default_rng(23)
    n, d, K = 500, 2000, 8
    data = (rng.normal(size=(n, d)) * 30.0).astype(np.float32)
    labels = rng.integers(0, K, size=n)
    got = np.asarray(compute_centroids(jnp.asarray(data), jnp.asarray(labels), K)).astype(np.float64)
    ref = np.zeros((K, d))
    for k in range(K):
        m = labels == k
        if m.any():
            ref[k] = data[m].astype(np.float64).mean(0)
    rel = np.abs(got - ref) / (np.abs(ref) + 1e-6)
    assert np.median(rel) < 1e-4, f"centroid matmul imprecise (median rel {np.median(rel):.2e}) — TF32 leaking?"


LOCAL_TESTS = [test_patience_formula, test_stop_condition_logic,
               test_patience_engaged_lower_bounds_counter, test_minibatch_converges_and_valid,
               test_determinism, test_parity_with_numba_reference, test_float32_default,
               test_default_tolerance_semantics, test_tolerance_reduces_iterations,
               test_quality_vs_streamed, test_numpy_numba_tolerance_parity,
               test_lloyd_stop_condition_logic, test_lloyd_converges_and_valid,
               test_lloyd_determinism, test_lloyd_float32_default, test_lloyd_reaches_fixed_point,
               test_lloyd_loss_monotone_non_increasing, test_lloyd_recovers_wellseparated_clusters,
               test_assign_clusters_matmul_matches_direct, test_one_hot_centroids_match_group_means]
GPU_TESTS = [test_gpu_backend, test_lloyd_gpu_runs_and_recovers,
             test_matmul_precision_not_tf32, test_minibatch_certified_gpu]

if __name__ == "__main__":
    tests = LOCAL_TESTS + (GPU_TESTS if _on_gpu() else [])
    print(f"Running {len(tests)} tests ({'GPU present' if _on_gpu() else 'local only'})...")
    for t in tests:
        t(); print("  PASS", t.__name__)
    print("All passed.")
