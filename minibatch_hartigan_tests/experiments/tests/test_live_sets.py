"""Tests for the live-set port (streamed_hartigan.run_hartigan_kmeans_ls) and the
certified mini-batch driver (mini_batch.mini_batch_hartigan_nb_certified).

The contract under test:
  * ONLINE: run_hartigan_kmeans_ls is BIT-IDENTICAL to run_hartigan_kmeans — same labels,
    bitwise-same centroids, same loss, same n_iters, on every regime, at every pass.
  * MINI-BATCH: certify=False is bit-identical to mini_batch_hartigan_nb with
    tolerance_fraction=0.0 (the only apples-to-apples setting — the certified driver has
    no tolerance by design); certify=True returns certified=True only at a TRUE fixed
    point, validated by an INDEPENDENT vectorized numpy checker.

    python -m pytest minibatch_hartigan_tests/experiments/tests/test_live_sets.py -v
    python minibatch_hartigan_tests/experiments/tests/test_live_sets.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # experiments/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                                "batch_hartigan_tests", "experiments")))

import numpy as np
from numba import jit

from streamed_hartigan import (
    _assign_label_hartigan_np,
    _full_eval,
    _reduced_eval,
    _weighted_cost,
    run_hartigan_kmeans,
    run_hartigan_kmeans_ls,
    assign_labels_lloyd_all,
    compute_centroids_np,
)
from mini_batch import mini_batch_hartigan_nb, mini_batch_hartigan_nb_certified


# ------------------------------------ helpers -----------------------------------------

def _blobs(rng, K=4, d=8, per=60, spread=12.0):
    centers = rng.normal(size=(K, d)) * spread
    data = np.vstack([centers[k] + rng.normal(size=(per, d)) for k in range(K)]).astype(np.float64)
    init = data[rng.choice(data.shape[0], size=K, replace=False)].copy()
    return data, init, K


# the six regimes the numba<->C crosschecks use, plus overlap stressors. NOTE: the
# spread=3.0/seed=3 regime is inert for ONLINE (converges in 1 pass); the spread=2.0/
# seed=1 regime was chosen empirically to make online churn for 7 passes, so the online
# reduced-path/live-set machinery is genuinely exercised.
REGIMES = [
    dict(seed=0, K=4, d=8, per=60, spread=12.0),
    dict(seed=1, K=8, d=16, per=40, spread=12.0),
    dict(seed=2, K=4, d=128, per=50, spread=12.0),
    dict(seed=3, K=4, d=8, per=80, spread=3.0),    # overlapping (MB churn; online inert)
    dict(seed=4, K=16, d=32, per=25, spread=12.0),
    dict(seed=5, K=8, d=64, per=125, spread=12.0),
    dict(seed=1, K=4, d=8, per=100, spread=2.0),   # overlapping, online churns 7 passes
]


def _make(reg):
    rng = np.random.default_rng(reg["seed"])
    return _blobs(rng, reg["K"], reg["d"], reg["per"], reg["spread"])


def numpy_fixed_point_check(data, centroids, assigns):
    """INDEPENDENT fixed-point verifier (vectorized numpy — deliberately not sharing
    code with the implementations): no point may prefer another cluster under the
    Hartigan weighting, given the returned centroids and current populations."""
    n = data.shape[0]
    K = centroids.shape[0]
    pops = np.bincount(assigns, minlength=K).astype(np.float64)
    sq = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)      # (n, K)
    wmat = sq * (pops / (pops + 1.0))[None, :]                          # move costs
    rows = np.arange(n)
    own_pop = pops[assigns]
    own_w = np.where(own_pop <= 1, -1.0,
                     sq[rows, assigns] * (own_pop / np.maximum(own_pop - 1.0, 1.0)))
    wmat[rows, assigns] = own_w                                         # stay costs
    return np.array_equal(np.argmin(wmat, axis=1), assigns)


# jitted probes of the ORIGINAL cost expressions (numba-lowered, so the comparison is
# numba-vs-numba — python numpy uses pairwise summation and would not be representative)

@jit
def _orig_online_costs(centroids, cluster_populations, data_point, label_point):
    """verbatim cost computation from _assign_label_hartigan_np, returning the vector"""
    distances = np.sum((data_point[None, ...] - centroids) ** 2, axis=-1)
    for i in range(centroids.shape[0]):
        if label_point == i:
            if cluster_populations[i] <= 1:
                distances[i] = -1.0
            else:
                scale_factor = cluster_populations[i] / (cluster_populations[i] - 1)
                distances[i] *= scale_factor
        else:
            scale_factor = cluster_populations[i] / (cluster_populations[i] + 1)
            distances[i] *= scale_factor
    return distances


@jit
def _orig_mb_costs(data_j, centroids, cluster_populations, a):
    """verbatim cost computation from _mini_batch_sweep, returning the vector"""
    K = centroids.shape[0]
    distances = np.zeros(K)
    for k in range(K):
        d = np.sum((data_j - centroids[k]) ** 2)
        if k == a:
            if cluster_populations[k] <= 1:
                d = -1.0
            else:
                d *= cluster_populations[k] / (cluster_populations[k] - 1)
        else:
            d *= cluster_populations[k] / (cluster_populations[k] + 1)
        distances[k] = d
    return distances


@jit
def _helper_costs(centroids, cluster_populations, data_point, label_point):
    K = centroids.shape[0]
    out = np.zeros(K)
    for l in range(K):
        out[l] = _weighted_cost(centroids, cluster_populations, data_point, l,
                                l == label_point)
    return out


# paranoid driver: run_hartigan_kmeans_ls's exact loop, but computing the FULL evaluation
# beside every reduced one and counting disagreements — a direct test of the invariant,
# independent of end-to-end trajectories. Proceeds on the full decision either way and
# maintains the caches exactly as the _ls driver does.

@jit
def _paranoid_online(data, init_centroids, max_iters):
    labels = assign_labels_lloyd_all(init_centroids, data)
    centroids = compute_centroids_np(data, labels, init_centroids.copy())
    cluster_populations = np.bincount(labels, minlength=init_centroids.shape[0])

    n = data.shape[0]
    last_update = np.zeros(init_centroids.shape[0], dtype=np.int64)
    last_eval = np.full(n, -1, dtype=np.int64)
    d_own = np.zeros(n, dtype=np.float64)
    step = np.int64(0)

    mismatches = 0
    old_labels = labels.copy()
    for _ in range(max_iters):
        for j in range(n):
            a = labels[j]
            full_label, w_own = _full_eval(centroids, cluster_populations, data[j], a)
            if last_update[a] > last_eval[j]:
                new_label = full_label                     # ls would take the full path
                if new_label == a:
                    d_own[j] = w_own
                    last_eval[j] = step
                    continue
            else:
                reduced = _reduced_eval(centroids, cluster_populations, data[j], a,
                                        d_own[j], last_update, last_eval[j])
                if reduced != full_label:
                    mismatches += 1                        # the invariant failed
                new_label = full_label                     # stay on the true trajectory
                if new_label == a:
                    last_eval[j] = step                    # ls refreshes ONLY the stamp
                    continue

            n_clust = cluster_populations[a]
            centroids[a] = (centroids[a] * n_clust - data[j]) / (n_clust - 1.0)
            n_clust = cluster_populations[new_label]
            centroids[new_label] = (centroids[new_label] * n_clust + data[j]) / (
                n_clust + 1.0
            )
            cluster_populations[a] -= 1
            cluster_populations[new_label] += 1
            labels[j] = new_label
            step += 1
            last_update[a] = step
            last_update[new_label] = step

        if np.array_equal(labels, old_labels):
            break
        old_labels = labels.copy()
    return mismatches


# ----------------------------------- the tests ----------------------------------------

def test_micro_bitwise_costs():
    # random draws: shared helper must equal BOTH original expressions to the bit, and
    # its argmin must equal the IMPORTED original's decision (mechanically tying the
    # hand-copied probes above to the real _assign_label_hartigan_np). pops drawn from
    # 0 upward so the empty-cluster branch (0/(0+1) weighting) is covered; the own
    # cluster is forced non-empty (a point's own cluster always holds the point).
    rng = np.random.default_rng(7)
    for _ in range(2000):
        K = int(rng.integers(2, 20))
        d = int(rng.integers(1, 40))
        cents = rng.normal(size=(K, d))
        x = rng.normal(size=d)
        pops = rng.integers(0, 50, size=K)
        a = int(rng.integers(0, K))
        pops[a] = max(1, int(pops[a]))
        ours = _helper_costs(cents, pops, x, a)
        assert ours.tobytes() == _orig_online_costs(cents, pops, x, a).tobytes()
        assert ours.tobytes() == _orig_mb_costs(x, cents, pops, a).tobytes()
        assert int(np.argmin(ours)) == int(_assign_label_hartigan_np(cents, pops, x, a))

    # constructed EXACT ties: integer coordinates, symmetric centroids, equal pops
    cents = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    x = np.array([0.0, 0.0])
    pops = np.array([5, 5, 5, 5], dtype=np.int64)
    for a in range(4):
        ours = _helper_costs(cents, pops, x, a)
        assert ours.tobytes() == _orig_online_costs(cents, pops, x, a).tobytes()
        # ties really are exact here (all squared distances equal 1.0)
        others = [ours[l] for l in range(4) if l != a]
        assert others[0] == others[1] == others[2]


def test_online_bit_identity():
    for reg in REGIMES:
        data, init, K = _make(reg)
        for checks in (True, False):
            iters = 10000 if checks else 12
            c0, l0, loss0, n0 = run_hartigan_kmeans(
                data, init.copy(), iters, checks_convergence=checks)
            c1, l1, loss1, n1 = run_hartigan_kmeans_ls(
                data, init.copy(), iters, checks_convergence=checks)
            assert np.array_equal(l0, l1), f"labels differ: {reg}, checks={checks}"
            assert c0.tobytes() == c1.tobytes(), f"centroid bits differ: {reg}"
            assert loss0 == loss1, f"loss differs: {reg}"
            assert n0 == n1, f"n_iters differ: {reg}: {n0} vs {n1}"


def test_online_per_pass_traces():
    # pass-by-pass equality: running k passes from scratch must agree for every k
    # (catches "right final answer, wrong path")
    for reg in REGIMES[:3] + [REGIMES[3]]:
        data, init, K = _make(reg)
        for k in range(1, 9):
            _, l0, _, _ = run_hartigan_kmeans(data, init.copy(), k,
                                              checks_convergence=False)
            _, l1, _, _ = run_hartigan_kmeans_ls(data, init.copy(), k,
                                                 checks_convergence=False)
            assert np.array_equal(l0, l1), f"pass-{k} labels differ: {reg}"


def test_online_paranoid_invariant():
    # every reduced decision must equal the full decision, at every point of every pass
    for reg in REGIMES:
        data, init, K = _make(reg)
        assert _paranoid_online(data, init.copy(), 10000) == 0, f"invariant broke: {reg}"


def test_online_fixed_point():
    # the converged state is a true fixed point per the independent checker
    for reg in REGIMES[:2]:
        data, init, K = _make(reg)
        cents, labels, _, _ = run_hartigan_kmeans_ls(data, init.copy(), 10000)
        assert numpy_fixed_point_check(data, cents, labels)


def test_mb_equivalence():
    # certify=False must be bit-identical to the old driver at tolerance 0
    for reg in REGIMES:
        data, init, K = _make(reg)
        for p in (0.25, 0.5, 0.75, 0.9, 1.0):
            a0, c0, n0, z0 = mini_batch_hartigan_nb(
                data, init.copy(), init.copy(), K, np.random.default_rng(42), p, 500,
                tolerance_fraction=0.0)
            a1, c1, cert, n1, z1 = mini_batch_hartigan_nb_certified(
                data, init.copy(), init.copy(), K, np.random.default_rng(42), p, 500,
                certify=False)
            assert np.array_equal(a0, a1), f"assigns differ: {reg} p={p}"
            assert c0.tobytes() == c1.tobytes(), f"centroid bits differ: {reg} p={p}"
            assert n0 == n1 and z0 == z1, f"counter/zero differ: {reg} p={p}"
            assert cert is False


def test_mb_certificate_validity():
    # certified=True must imply a true fixed point (independent checker); certified
    # runs never stop earlier than the heuristic (prefix property: same rng stream)
    for reg in REGIMES:
        data, init, K = _make(reg)
        for p in (0.25, 0.5, 0.9, 1.0):
            a0, c0, n0, z0 = mini_batch_hartigan_nb(
                data, init.copy(), init.copy(), K, np.random.default_rng(9), p, 2000,
                tolerance_fraction=0.0)
            a1, c1, cert, n1, z1 = mini_batch_hartigan_nb_certified(
                data, init.copy(), init.copy(), K, np.random.default_rng(9), p, 2000)
            if z1:
                continue                       # empty-cluster abort: nothing to certify
            assert n1 >= n0, f"certified stopped early: {reg} p={p}"
            if cert:
                assert numpy_fixed_point_check(data, c1, a1), \
                    f"certified but not a fixed point: {reg} p={p}"
            else:
                assert n1 == 2000, f"uncertified without exhausting budget: {reg} p={p}"
            if n1 == n0 and not z0:
                # certificate passed at the heuristic's own stopping point ->
                # trajectories are identical prefixes
                assert np.array_equal(a0, a1)


def test_mb_p1_shortcut():
    # at p=1.0 the zero-move full-batch pass itself is the certificate
    data, init, K = _make(REGIMES[0])
    a, c, cert, n, z = mini_batch_hartigan_nb_certified(
        data, init.copy(), init.copy(), K, np.random.default_rng(3), 1.0, 500)
    assert cert is True and not z
    assert numpy_fixed_point_check(data, c, a)


def test_certificate_failure_resume():
    # (adversarial-review witness) a regime + seed where the FIRST certificate check
    # FAILS: the heuristic stops at 14 passes on a state that is NOT a fixed point; the
    # certified run rejects it, resumes mini-batching, and certifies at 19. This is the
    # only test exercising the fail-resume path (quiet reset + stamp counter+1) — a
    # stamp-poisoning mutant passes every other test but produces a FALSE certificate
    # detectable only here.
    # Witness re-derived when the sampler gained the leave-one-per-cluster hold-out
    # (sample_batch_holdout): that changed the batch RNG stream, so the old mb seed 1017 now
    # stops ON a fixed point and no longer exercises the fail path. Same blobs/proportion,
    # mb seed 1017 -> 1004. Equivalent alternatives if this one ever goes stale:
    # (blobs seed 23, mb 1002, p=0.25, 33->58) and (blobs seed 41, mb 1023, p=0.5, 10->15).
    rng = np.random.default_rng(17)
    data, init, K = _blobs(rng, 4, 8, 80, 3.0)
    a0, c0, n0, z0 = mini_batch_hartigan_nb(
        data, init.copy(), init.copy(), K, np.random.default_rng(1004), 0.5, 800,
        tolerance_fraction=0.0)
    a1, c1, cert, n1, z1 = mini_batch_hartigan_nb_certified(
        data, init.copy(), init.copy(), K, np.random.default_rng(1004), 0.5, 800)
    assert not z0 and not z1
    assert not numpy_fixed_point_check(data, c0, a0), \
        "witness regressed: the heuristic stop is a fixed point (no failed check)"
    assert n1 > n0, f"expected a failed certificate to add passes: {n1} vs {n0}"
    assert cert is True
    assert numpy_fixed_point_check(data, c1, a1)


def test_tie_break_exactness():
    # (adversarial-review witnesses) exact-tie regressions: the (w == best and l < best)
    # clauses in _reduced_eval/_decide_ls are load-bearing — deleting them diverges from
    # bit-identity on these integer datasets, while every Gaussian regime is tie-blind
    # (exact float ties have measure zero there).
    # 1) online end-to-end: a point hits an exact weighted-cost tie between a live
    #    lower-index cluster and its cached own cost; argmin-first must MOVE.
    d2 = np.array([[3, 4], [3, 0], [4, 0], [4, 4], [3, 1], [2, 2], [3, 3], [3, 2]],
                  dtype=np.float64)
    i2 = np.array([[3, 2], [2, 2], [3, 3]], dtype=np.float64)
    r_ref = run_hartigan_kmeans(d2, i2.copy(), 50)
    r_ls = run_hartigan_kmeans_ls(d2, i2.copy(), 50)
    assert r_ref[1].tolist() == [2, 0, 0, 2, 0, 1, 1, 1] and r_ref[3] == 4
    assert np.array_equal(r_ref[1], r_ls[1]) and r_ref[3] == r_ls[3]
    assert r_ref[0].tobytes() == r_ls[0].tobytes()
    # 2) mini-batch equivalence on tie-heavy integer data
    d3 = np.array([[4, 4], [2, 3], [2, 0], [3, 4], [2, 1], [1, 1], [3, 3], [2, 2]],
                  dtype=np.float64)
    i3 = np.array([[1, 1], [3, 3], [2, 1]], dtype=np.float64)
    m0 = mini_batch_hartigan_nb(d3, i3.copy(), i3.copy(), 3, np.random.default_rng(4117),
                                0.3, 500, tolerance_fraction=0.0)
    m1 = mini_batch_hartigan_nb_certified(d3, i3.copy(), i3.copy(), 3,
                                          np.random.default_rng(4117), 0.3, 500,
                                          certify=False)
    assert np.array_equal(m0[0], m1[0]) and m0[2] == m1[3]
    # 3) direct unit ties through the reduced evaluator. pops=3 makes both weight
    #    factors exact binary (1.5 and 0.75), and small-integer coordinates make the
    #    squared distances exact, so w == d_own is an EXACT float64 tie.
    pops = np.array([3, 3, 3], dtype=np.int64)
    x = np.array([0.0, 0.0])
    # tie with a LOWER-index live candidate: own a=1 has d_own = 4*1.5 = 6.0; live
    # cluster 0 offers 8*0.75 = 6.0 -> argmin-first takes index 0: MOVE
    cents_a = np.array([[2.0, 2.0], [2.0, 0.0], [9.0, 9.0]])
    lu_a = np.array([5, 0, 0], dtype=np.int64)          # only cluster 0 live for stamp 3
    assert _reduced_eval(cents_a, pops, x, 1, 6.0, lu_a, 3) == 0
    # tie with a HIGHER-index live candidate: own a=0 has d_own = 8*1.5 = 12.0; live
    # cluster 2 offers 16*0.75 = 12.0 -> argmin-first keeps the lower index: STAY
    cents_b = np.array([[2.0, 2.0], [9.0, 9.0], [4.0, 0.0]])
    lu_b = np.array([0, 0, 5], dtype=np.int64)          # only cluster 2 live for stamp 3
    assert _reduced_eval(cents_b, pops, x, 0, 12.0, lu_b, 3) == 0


def test_jax_certified():
    # JAX certified wrapper (runs on CPU jax locally): certified=True implies a true
    # fixed point per the independent checker; deterministic; never stops before the
    # tolerance-0 heuristic run (same key => same pass stream, checks consume no rng)
    import jax
    import jax.numpy as jnp
    from jax_minibatch_hartigan import (
        run_minibatch_hartigan_kmeans,
        run_minibatch_hartigan_kmeans_certified,
    )
    for reg in REGIMES[:3] + [REGIMES[3]]:
        data, init, K = _make(reg)
        dj, ij = jnp.asarray(data), jnp.asarray(init)
        for p in (0.5, 0.9, 1.0):
            key = jax.random.PRNGKey(reg["seed"])
            c0, l0, _, n0 = run_minibatch_hartigan_kmeans(
                dj, ij, key, p, 2000, tolerance_fraction=0.0)
            c1, l1, _, n1, cert = run_minibatch_hartigan_kmeans_certified(
                dj, ij, key, p, 2000)
            assert int(n1) >= int(n0), f"certified stopped early: {reg} p={p}"
            if cert:
                assert numpy_fixed_point_check(
                    np.asarray(dj, dtype=np.float64),
                    np.asarray(c1, dtype=np.float64), np.asarray(l1)), \
                    f"certified but not a fixed point: {reg} p={p}"
            else:
                assert int(n1) >= 2000, f"uncertified below budget: {reg} p={p}"
            if int(n1) == int(n0):
                assert np.array_equal(np.asarray(l0), np.asarray(l1))
            # determinism of the certified wrapper
            c2, l2, _, n2, cert2 = run_minibatch_hartigan_kmeans_certified(
                dj, ij, key, p, 2000)
            assert np.array_equal(np.asarray(l1), np.asarray(l2))
            assert int(n1) == int(n2) and cert == cert2


LOCAL_TESTS = [test_micro_bitwise_costs, test_online_bit_identity,
               test_online_per_pass_traces, test_online_paranoid_invariant,
               test_online_fixed_point, test_mb_equivalence,
               test_mb_certificate_validity, test_mb_p1_shortcut,
               test_certificate_failure_resume, test_tie_break_exactness,
               test_jax_certified]

if __name__ == "__main__":
    print(f"Running {len(LOCAL_TESTS)} live-set tests...")
    for t in LOCAL_TESTS:
        t()
        print("  PASS", t.__name__)
    print("All passed.")
