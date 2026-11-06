"""
audit_jax_vs_numpy.py — fail-loud equivalence gate for the JAX/GPU Batch-Hartigan port.

Proves the JAX driver reproduces the numpy study's failure SIGNALS (empty cluster, loop +
loop_size, converged/max_iter) and quantifies nothing on faith. Structure: negative controls
FIRST (a checker that cannot fail proves nothing), then numpy<->JAX-float64 exact match on a
shared battery with the decisive fixtures (forced empty, period-2 AND period-3 cycles,
singletons, all-identical, K near n), batched-vs-single parity, isotropic-generator validation,
and reasonableness invariants. Nonzero exit = red. Honors KMEANS_REQUIRE_GPU.

    python audit_jax_vs_numpy.py            # CPU parity gate (KMEANS_REQUIRE_GPU=0)
    KMEANS_REQUIRE_GPU=1 python audit_jax_vs_numpy.py   # + x64/GPU backend asserts
"""
import os
import sys

import numpy as np

import batch_hartigan_jax as bhj
import jax
import jax.numpy as jnp
from batch_hartigan import batch_hartigan as np_bh          # numpy oracle (unchanged reference)
from batch_hartigan import generate_ground_truth_data       # legacy multivariate_normal generator

REQUIRE_GPU = os.environ.get("KMEANS_REQUIRE_GPU", "0") == "1"

_fails = 0
_checks = 0


def check(cond, msg):
    global _fails, _checks
    _checks += 1
    if cond:
        print(f"  [ok  ] {msg}", flush=True)
    else:
        _fails += 1
        print(f"  [FAIL] {msg}", flush=True)


def sig(t):
    """(counter, loop, loop_size, zero) from a driver's 6-tuple."""
    _a, _c, counter, loop, lsz, zero = t
    return (int(counter), bool(loop), int(lsz), bool(zero))


def np_trial(data_h, init_h, K, max_iter=500):
    """numpy oracle on host float64 data. Returns (assigns, sig)."""
    a, c, counter, loop, lsz, zero = np_bh(data_h.copy(), init_h.copy(), init_h.copy(),
                                           K, max_iter=max_iter)
    return a, (int(counter), bool(loop), int(lsz), bool(zero))


def jax_trial(data_h, init_h, K, max_iter=500):
    """JAX single-trial driver on the SAME host data. Returns (assigns, sig)."""
    t = bhj.batch_hartigan_jax(jnp.asarray(data_h), jnp.asarray(init_h), K, max_iter=max_iter)
    return np.asarray(t[0]), sig(t)


# ─────────────────────────────────────────────────────────────────────────────
def section_backend():
    print("== backend / precision ==", flush=True)
    x = bhj.compare_batch_hart_jax(jnp.zeros((4, 3)), jnp.zeros((2, 3)),
                                   jnp.zeros((4,), dtype=jnp.int32), 2)
    check(x[1].dtype == jnp.float64, f"x64 engaged — kernel centroids are float64 ({x[1].dtype})")
    be = jax.default_backend()
    if REQUIRE_GPU:
        check(be == "gpu", f"KMEANS_REQUIRE_GPU=1 and backend is gpu ({be}, {jax.devices()})")
    else:
        print(f"  [info] backend={be} (KMEANS_REQUIRE_GPU not set) — CPU parity gate", flush=True)


def section_nc():
    print("== NC negative controls (checkers must REJECT known-bad) ==", flush=True)
    # partition comparator
    check(not np.array_equal(np.array([0, 1, 1]), np.array([0, 1, 0])),
          "NC: partition comparator reports inequality on different partitions")
    # loop_checker: fires on a period-3 sequence, never on a monotone one
    s0, s1, s2 = (np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([2, 0, 1]))
    mem = {}
    fired = []
    for c, s in enumerate([s0, s1, s2, s0]):     # s0 repeats at index 3 -> loop_size 3
        loop, lsz, mem = bhj.loop_checker(s, mem, c)
        fired.append((loop, lsz))
    check(fired[:3] == [(False, 0), (False, 0), (False, 0)] and fired[3] == (True, 3),
          f"NC: loop_checker fires on a period-3 cycle with loop_size==3 ({fired[3]})")
    mem = {}
    monotone_fire = any(bhj.loop_checker(np.array([i, i, i]), mem, i)[0] for i in range(6))
    check(not monotone_fire, "NC: loop_checker never fires on a monotone (all-distinct) sequence")
    # zero detector inside the kernel
    data = jnp.asarray(np.random.default_rng(0).normal(size=(6, 3)))
    cent = jnp.asarray(np.random.default_rng(1).normal(size=(3, 3)))
    a_all0 = jnp.zeros((6,), dtype=jnp.int32)                       # clusters 1,2 empty
    _na, _nc, zf = bhj.compare_batch_hart_jax(data, cent, a_all0, 3)
    check(bool(zf), "NC: kernel zero-flag True when the incoming partition has an empty cluster")
    a_full = jnp.asarray(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))
    _na, _nc, zf2 = bhj.compare_batch_hart_jax(data, cent, a_full, 3)
    check(not bool(zf2), "NC: kernel zero-flag False when every cluster is populated")


def _make_host(dim, K, intra, inter, cluster_size, trial_id):
    """jax-iso data pulled to host float64 (fed byte-identically to both numpy and JAX)."""
    d, i = bhj.generate_one(dim, K, intra, inter, cluster_size, trial_id)
    return np.asarray(d), np.asarray(i)


def section_A_exact_match():
    print("== A numpy <-> JAX-float64 EXACT match on a shared small-d battery ==", flush=True)
    dims = [2, 5, 10, 50]
    Ks = [2, 4, 8]
    css = [2, 5, 20, 100]
    SNR = [(1, 10), (4, 7), (7, 4), (10, 1)]
    total = 0
    diverge = 0
    diverge_examples = []
    for dim in dims:
        for K in Ks:
            for cs in css:
                if cs < 2:
                    continue
                for (intra, inter) in SNR:
                    for tid in range(6):        # seeds per cell
                        dh, ih = _make_host(dim, K, intra, inter, cs, tid)
                        na, s_np = np_trial(dh, ih, K)
                        ja, s_jx = jax_trial(dh, ih, K)
                        total += 1
                        # partition compared only when not a zero-stop (zero discards assigns)
                        part_ok = s_np[3] or np.array_equal(na, ja)
                        if s_np != s_jx or not part_ok:
                            diverge += 1
                            if len(diverge_examples) < 8:
                                diverge_examples.append(
                                    (dim, K, cs, intra, inter, tid, s_np, s_jx, part_ok))
    for ex in diverge_examples:
        print(f"     DIVERGE d={ex[0]} K={ex[1]} cs={ex[2]} snr=({ex[3]},{ex[4]}) t={ex[5]}: "
              f"numpy{ex[6]} jax{ex[7]} part_ok={ex[8]}", flush=True)
    check(diverge == 0,
          f"A: {total} shared trials — numpy==JAX (counter,loop,loop_size,zero,partition), "
          f"{diverge} divergence(s) [any = exact-tie argmin flip]")


def section_fixtures():
    print("== A' decisive fixtures (forced empty, period-2, period-3, singleton, identical) ==",
          flush=True)
    rng = np.random.default_rng(20260724)

    # forced empty at init: one init center far from all data -> its cluster is empty on pass 1.
    d, K = 4, 3
    data = rng.normal(size=(12, d)) * 0.5
    init = np.vstack([data[0], data[5], np.full(d, 1e6)])          # 3rd center unreachable
    na, s_np = np_trial(data, init, K)
    ja, s_jx = jax_trial(data, init, K)
    check(s_np[3] and s_jx[3] and s_np[0] == s_jx[0] == 1,
          f"forced-empty: both report zero at counter=1 (numpy{s_np} jax{s_jx})")

    # all-identical points -> everything collapses to cluster 0 (first-index tie) -> empty
    # clusters -> zero-stop; both backends must agree on the signal.
    data = np.ones((10, 3)) * 2.0
    init = data[:4].copy()
    _na, s_np = np_trial(data, init, 4)
    _ja, s_jx = jax_trial(data, init, 4)
    check(s_np == s_jx, f"all-identical points: numpy==jax signals ({s_np} vs {s_jx})")

    # seed-searched cycles from the numpy oracle, then cross-check JAX. Period-2 is common in
    # the low-SNR corner; period-3+ is rare, so it is best-effort (the NC above already proves
    # loop_checker catches period-3, and section A cross-checks any real cycle that occurs).
    found = {}          # period -> (dh, ih, s_np, tid, K)
    search_cfgs = [(3, 6, 10, 1, 3), (2, 8, 10, 1, 3), (4, 10, 10, 1, 2),
                   (3, 12, 10, 1, 2), (5, 8, 10, 1, 3), (2, 14, 10, 1, 2)]
    for (dim, Ks, intra, inter, cs) in search_cfgs:
        for tid in range(4000):
            dh, ih = _make_host(dim, Ks, intra, inter, cs, tid)
            _na, s = np_trial(dh, ih, Ks)
            if s[1] and s[2] not in found:
                found[s[2]] = (dh, ih, s, tid, Ks)
        if 2 in found and 3 in found:
            break
    check(2 in found, "period-2 cycle exists in the loop-rich corner")
    for period in sorted(found):
        dh, ih, s_np, tid, Ks = found[period]
        _ja, s_jx = jax_trial(dh, ih, Ks)
        check(s_np == s_jx and s_np[2] == period,
              f"period-{period} cycle (K={Ks}, tid={tid}): numpy==jax, loop_size=={period} "
              f"(numpy{s_np} jax{s_jx})")
    if 3 not in found:
        print("  [info] no period-3 real trial found in budget (rare; NC + section A cover "
              "period-3 detection and parity)", flush=True)

    # singletons: K near n so several clusters have population 1 (stay-factor 0.0 branch).
    for tid in range(20):
        dh, ih = _make_host(5, 10, 4, 7, 2, tid)          # n=20, K=10
        na, s_np = np_trial(dh, ih, 10)
        ja, s_jx = jax_trial(dh, ih, 10)
        part_ok = s_np[3] or np.array_equal(na, ja)
        if not (s_np == s_jx and part_ok):
            check(False, f"singleton/K-near-n tid={tid}: numpy{s_np} jax{s_jx} part_ok={part_ok}")
            break
    else:
        check(True, "singleton / K-near-n (n=20,K=10) x20: numpy==jax (signals + partitions)")


def section_B_batched():
    print("== B batched driver == single-trial driver (per-lane) ==", flush=True)
    mism = 0
    for (dim, K, cs) in [(10, 4, 20), (5, 8, 3), (50, 5, 10), (3, 6, 2), (100, 3, 30)]:
        for (intra, inter) in [(1, 10), (10, 1)]:
            tids = list(range(12))
            data_b, init_b = bhj.generate_batch(dim, K, intra, inter, cs, tids)
            batched = bhj.batch_hartigan_jax_batched(data_b, init_b, K, max_iter=500)
            for i in tids:
                single = bhj.batch_hartigan_jax(data_b[i], init_b[i], K, max_iter=500)
                if sig(batched[i]) != sig(single) or not np.array_equal(batched[i][0],
                                                                        np.asarray(single[0])):
                    mism += 1
    check(mism == 0, f"B: batched lanes reproduce single-trial 6-tuples + partitions exactly "
                     f"({mism} mismatch(es))")


def section_C_datagen():
    print("== C isotropic generator: variance recovery + downstream-rate match ==", flush=True)
    # within-cluster variance ~ intra: one big high-d trial (tight over d coords x cs points).
    dim, K, cs, intra, inter = 2000, 6, 300, 4, 7
    data, _init = bhj.generate_batch(dim, K, intra, inter, cs, [0])
    X = np.asarray(data[0])                                       # (K*cs, dim)
    blocks = X.reshape(K, cs, dim)
    block_means = blocks.mean(axis=1)                            # (K, dim)
    within = ((blocks - block_means[:, None, :]) ** 2).mean()     # ~ intra
    check(abs(within / intra - 1.0) < 0.05,
          f"C: within-cluster variance ~ intra ({within:.3f} vs {intra}) at d={dim}")
    per_coord = (blocks - block_means[:, None, :]).reshape(-1, dim).var(axis=0)
    check(per_coord.max() / per_coord.min() < 1.5,
          f"C: isotropy — per-coordinate variance ratio bounded ({per_coord.max()/per_coord.min():.2f})")
    check(np.isfinite(X).all(), "C: generated data is all-finite")
    # between-center variance ~ inter: pool block means across MANY trials (each has independent
    # true centers ~ N(0,inter)); var(ddof=1) - intra/cs removes the finite-cs bias. Pooling many
    # centers beats the (K-1)/K + small-sample bias of estimating from a single trial's K centers.
    Bd, Kd, csd, intd, intrd = 500, 15, 200, 7, 4
    dm, _ = bhj.generate_batch(Bd, Kd, intrd, intd, csd, list(range(40)))
    bmeans = np.asarray(dm).reshape(40, Kd, csd, Bd).mean(axis=2).reshape(40 * Kd, Bd)
    between = bmeans.var(axis=0, ddof=1).mean() - intrd / csd
    check(abs(between / intd - 1.0) < 0.10,
          f"C: between-center variance ~ inter ({between:.3f} vs {intd}, pooled over {40*Kd} centers)")

    # downstream invariance: failure rates from jax-iso vs the legacy multivariate_normal
    # generator agree within Monte-Carlo error (both clustered by the SAME numpy oracle).
    dim2, K2, cs2, intra2, inter2, M = 6, 6, 3, 10, 1, 400
    z_j = l_j = z_m = l_m = 0
    for tid in range(M):
        dh, ih = _make_host(dim2, K2, intra2, inter2, cs2, tid)
        _a, s = np_trial(dh, ih, K2)
        z_j += s[3]; l_j += s[1]
        rng = np.random.default_rng(10_000 + tid)
        cen, dat, ic, _lab = generate_ground_truth_data(rng, cs2, K2, dim2, intra2, inter2)
        _a2, s2 = np_trial(dat, ic, K2)
        z_m += s2[3]; l_m += s2[1]
    band = 4.0 * np.sqrt(0.25 / M)      # ~4 sigma binomial band on a rate difference
    check(abs(z_j - z_m) / M < band,
          f"C: zero-rate matches legacy generator ({z_j/M:.3f} vs {z_m/M:.3f}, band {band:.3f})")
    check(abs(l_j - l_m) / M < band,
          f"C: loop-rate matches legacy generator ({l_j/M:.3f} vs {l_m/M:.3f}, band {band:.3f})")


def section_D_sanity():
    print("== D reasonableness invariants on a small real sweep ==", flush=True)
    import run_simulation_jax as rj
    rates = {}
    for (dim, K, cs) in [(50, 8, 3), (50, 8, 30)]:
        for (intra, inter) in rj.SNR_PAIRS:
            row = rj.run_regime_jax((0, K, dim, intra, inter, cs), trials_per_regime=200)
            conv = row["trials"] - row["zero_count"] - row["loop_count"] - row["max_iter_count"]
            ok = (row["zero_count"] + row["loop_count"] + row["max_iter_count"] + conv
                  == row["trials"] and conv >= 0)
            check(ok, f"D: counts partition trials exactly "
                     f"(d={dim} K={K} cs={cs} snr=({intra},{inter}): "
                     f"z={row['zero_count']} l={row['loop_count']} m={row['max_iter_count']} conv={conv})")
            check(0 <= row["loop_count"] <= row["trials"] and row["max_loop_size"] != 1,
                  f"D: loop_count in range and loop_size!=1 (max_loop_size={row['max_loop_size']})")
            check(row["mean_iters_before_loop"] >= 0.0 and row["max_iters_before_loop"] >= 0,
                  "D: iters_before_loop >= 0")
            rates[(dim, K, cs, intra, inter)] = row["loop_count"] / max(row["trials"], 1)
    # soft SNR-monotonicity flag: overlapping (10,1) should loop >= well-separated (1,10)
    for (dim, K, cs) in [(50, 8, 3), (50, 8, 30)]:
        hi = rates[(dim, K, cs, 10, 1)]      # low SNR (overlap)
        lo = rates[(dim, K, cs, 1, 10)]      # high SNR (separated)
        print(f"  [info] SNR monotonicity d={dim} K={K} cs={cs}: "
              f"loop_rate(overlap)={hi:.3f} >= loop_rate(separated)={lo:.3f}? "
              f"{'yes' if hi >= lo else 'NO (flag)'}", flush=True)


def main():
    print(f"audit_jax_vs_numpy | backend={jax.default_backend()} | "
          f"REQUIRE_GPU={REQUIRE_GPU}", flush=True)
    section_backend()
    section_nc()
    section_A_exact_match()
    section_fixtures()
    section_B_batched()
    section_C_datagen()
    section_D_sanity()
    print(f"\nsummary: {_checks - _fails}/{_checks} passed, {_fails} failed", flush=True)
    if _fails:
        print("AUDIT FAILED", flush=True)
        sys.exit(1)
    print("AUDIT PASSED — numpy<->JAX-float64 parity + fixtures + generator + sanity verified",
          flush=True)


if __name__ == "__main__":
    main()
