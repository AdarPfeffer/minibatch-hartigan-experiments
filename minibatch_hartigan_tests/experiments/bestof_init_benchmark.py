"""bestof_init_benchmark.py — "how effective is best-of-N initialization?" Hartigan study.

For each (d, N, K) cell and each of N_DATASETS datasets, cluster the dataset from N_INITS
different initializations and report, per dataset, statistics over those inits:
  quality : nmi_mean, nmi_std, nmi_max, sse_mean, sse_std, sse_min, nmi_at_min_sse
  runtime : conv_mean/std, singleit_mean/std, epoch2_mean/std, n_iters_mean/std
plus certified_frac (numba/JAX certified mb) and fp_movers_mean (Fortran).

TIMING DECOMPOSITION (exact — see the plan):
  - dataset generation           : timed for NOTHING (the algorithm is given a dataset).
  - init-centroid SELECTION      : timed -> CONVERGENCE only (part of initialization).
  - Lloyd init (assign+recompute) : timed -> CONVERGENCE only.
  - Hartigan iterations           : timed -> CONVERGENCE and SINGLE-ITER.
  => convergence = wall(select init + Lloyd + passes)      [Python clock, block for JAX]
     single-iter = wall(passes only) / n_iters             [post-init clock / n_iters]
     epoch-2     = mean wall of the FIRST post-init pass, replayed R_EPOCH2 (=100) times
                   ("the second epoch": init counts as epoch 1). Distinct from single-iter:
                   epoch-2 isolates the special from-fresh-init first pass.

Every trial runs STRICTLY SINGLE-THREADED. CPU work is parallelized across DATASETS only
(never within a clustering). All backends cluster BYTE-IDENTICAL datasets + inits
(deterministic seeding), so comparisons are apples-to-apples; online impls are exactly
comparable, mini-batch distributionally (per-backend samplers).

Implementations (IMPLS): numba_online_ls, numba_mb_cert, fortran_ls, fortran_nols (cpu group)
and jax_mb_cert, jax_lloyd (gpu group). --init-scheme picks random_centroid or random_partition.
Run:
  # local CPU smoke (numba only):
  KMEANS_SMOKE=1 python minibatch_hartigan_tests/experiments/bestof_init_benchmark.py \
      --backend-group cpu --impls numba_online_ls,numba_mb_cert
  # cluster CPU job:  --backend-group cpu   (numba + fortran, many workers)
  # cluster GPU job:  --backend-group gpu   (jax, single process per GPU)
The Slurm jobs that produced the shipped data are in jobs/.
"""
import os
import sys

# single-thread pinning MUST precede numpy/numba/jax imports (re-runs in spawned workers;
# inherited by forked workers). RED LINE: each trial is single-threaded.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                    # numba + JAX modules
sys.path.insert(0, os.path.join(_HERE, "..", "..", "batch_hartigan_tests",
                                "experiments"))              # mini_batch dependency
sys.path.insert(0, os.path.join(_HERE, "fortran"))           # hartigan_fortran_ctypes

import argparse
import math
import time

import numpy as np

# ---- study configuration --------------------------------------------------------------
SMOKE = os.environ.get("KMEANS_SMOKE", "0") == "1"

# N: Hybrid dataset-size scale (superset of a logspace-7; denser + 5000/6500/8000
# across the 3.8k-10k gap).
N_TO_TEST = np.array([32, 51, 83, 134, 217, 350, 565, 913, 1473, 2378,
                      3838, 5000, 6500, 8000, 10000])
D_TO_TEST = [100, 1000, 10000]
K_TO_TEST = [4, 8, 16]
# Mini-batch sampling rates. 0.95 is GONE: the batch may never contain every point of a
# cluster (else the cluster can be emptied), so the top rate is the per-cell
# p_max = (n-K)/n supplied by proportions_for(). See sample_batch_holdout in mini_batch.py.
PROPORTIONS = [0.25, 0.5, 0.75, 0.9]
N_DATASETS = 100                 # datasets per (d, N, K) cell  (== "trial count")
N_INITS = 50                     # inits per dataset            (best-of-N)
R_EPOCH2 = 100                   # replays of the first post-init pass (timing stability)
NOISE_VARIANCE, PRIOR_VARIANCE = 10.0, 1.0
MAX_ITERS = 10000                # online epoch budget
# Mini-batch pass budget. Lowered 10000 -> 1000 deliberately: at p_max the assignment sequence
# enters a bounded oscillation (a measured run revisited a state at pass 16 with lag 2, then took
# 7,385 more passes to escape a 50-state recurrent set), so the long tail is a random waiting time
# that produces no new information. The tail is confined to p_max — no grid proportion (<=0.9)
# exceeded 206 passes anywhere in the grid. Runs truncated by this budget are recorded via
# capped_frac, and the oscillation itself via confinement_mean/cycle_pass_mean, so the truncation
# is reported rather than silent. Raise with KMEANS_CERT_MAX_ITERS to study the tail.
CERT_MAX_ITERS = int(os.environ.get("KMEANS_CERT_MAX_ITERS", "1000"))

if SMOKE:
    N_TO_TEST = np.array([32, 100])
    D_TO_TEST = [8, 16]
    K_TO_TEST = [4]
    PROPORTIONS = [0.5, 0.9]
    N_DATASETS = 3
    N_INITS = 4
    R_EPOCH2 = 5
    print("SMOKE MODE — tiny grid, results are NOT benchmarks", flush=True)

# ---- deterministic seeding (identical inputs across backends AND across the gpu/cpu jobs)
ROOT_SEED = 20260723
_PURPOSE_DATASET, _PURPOSE_INIT, _PURPOSE_MB = 1, 2, 3


def _seed64(*parts):
    ss = np.random.SeedSequence([int(ROOT_SEED)] + [int(p) for p in parts])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def dataset_seed(d, n, K, dataset_id):
    # dataset does NOT depend on the init scheme -> the 3 init-scheme jobs share datasets.
    return _seed64(_PURPOSE_DATASET, d, n, K, dataset_id)


def init_seed(d, n, K, dataset_id, init_id):
    # init selection RNG source; scheme-independent (the scheme decides how it's consumed).
    return _seed64(_PURPOSE_INIT, d, n, K, dataset_id, init_id)


def mb_seed(d, n, K, dataset_id, init_id, prop_idx):
    # mini-batch batch-sampling seed; fresh per (dataset, init, proportion) so numba mb is
    # per-cell reproducible without touching its sampling mechanism.
    return _seed64(_PURPOSE_MB, d, n, K, dataset_id, init_id, prop_idx)


def generate_dataset(n, d, K, seed):
    """GMM: K centers ~ N(0, PRIOR), uniform labels (redrawn until every class has at least
    one point), points += N(0, NOISE). Deterministic in `seed`."""
    if K > n:
        raise ValueError(f"need K <= n so that every class can be non-empty, got K={K}, n={n}")
    rng = np.random.default_rng(seed)
    rng_centers, rng_labels, rng_noise = rng.spawn(3)
    centroids = rng_centers.normal(0.0, np.sqrt(PRIOR_VARIANCE), size=(K, d))
    labels = rng_labels.integers(0, K, size=n)
    while np.bincount(labels, minlength=K).min() == 0:
        labels = rng_labels.integers(0, K, size=n)
    data = centroids[labels] + rng_noise.normal(0.0, np.sqrt(NOISE_VARIANCE), size=(n, d))
    return data, labels


# ---- scoring (float64, uniform across impls) -------------------------------------------
def sse_f64(data, centers, assigns):
    centers = np.asarray(centers, dtype=np.float64)
    return float(np.sum((np.asarray(data, dtype=np.float64)
                         - centers[np.asarray(assigns)]) ** 2))


_nmi = None


def nmi_score(true_labels, pred):
    global _nmi
    if _nmi is None:
        from sklearn.metrics import normalized_mutual_info_score
        _nmi = normalized_mutual_info_score
    return float(_nmi(true_labels, np.asarray(pred)))


# =======================================================================================
# epoch-2: mean wall of the FIRST post-init pass, replayed R times on a frozen snapshot.
# Reuses the frozen jitted epoch/sweep primitives; per-rep bit-equality is asserted (a
# difference would mean the clock timed different work). No budget cap (no early stops).
# =======================================================================================
def _epoch2_online_numba(data, init, K, repeats):
    from streamed_hartigan import (assign_labels_lloyd_all, compute_centroids_np,
                                    _online_epoch_ls)
    n = len(data)
    labels0 = assign_labels_lloyd_all(init, data)
    cents0 = compute_centroids_np(data, labels0, init.copy())
    pops0 = np.bincount(labels0, minlength=K)

    def one_rep():
        labels, cents, pops = labels0.copy(), cents0.copy(), pops0.copy()
        last_update = np.zeros(K, dtype=np.int64)
        last_eval = np.full(n, -1, dtype=np.int64)
        d_own = np.zeros(n, dtype=np.float64)
        step = np.int64(0)
        t0 = time.perf_counter()
        _online_epoch_ls(data, cents, labels, pops, last_update, last_eval, d_own, step)
        return labels, cents, time.perf_counter() - t0

    ref_l, ref_c, _ = one_rep()
    ref_lb, ref_cb = ref_l.tobytes(), ref_c.tobytes()
    total = 0.0
    for _ in range(repeats):
        l, c, el = one_rep()
        if l.tobytes() != ref_lb or c.tobytes() != ref_cb:
            raise RuntimeError("online epoch-2 rep diverged from priming rep")
        total += el
    return total / repeats


def _epoch2_mb_numba(data, init, K, proportion, mb_seed_val, repeats):
    from streamed_hartigan import assign_labels_lloyd_all, compute_centroids_np
    from mini_batch import _mini_batch_sweep_ls
    n = len(data)
    bs = max(1, math.ceil(proportion * n))
    assigns0 = assign_labels_lloyd_all(init, data)
    cents0 = compute_centroids_np(data, assigns0, init.copy())

    def one_rep():
        assigns, cents = assigns0.copy(), cents0.copy()
        last_update = np.zeros(K, dtype=np.int64)
        last_eval = np.full(n, -1, dtype=np.int64)
        d_own = np.zeros(n, dtype=np.float64)
        changed = np.zeros(K, dtype=np.bool_)
        rng = np.random.default_rng(mb_seed_val)     # SAME first batch every rep
        t0 = time.perf_counter()
        populations = np.bincount(assigns, minlength=K)
        batch_idx = rng.choice(n, size=bs, replace=False)
        _mini_batch_sweep_ls(data, cents, populations, batch_idx, assigns,
                             last_update, last_eval, d_own, 1, changed)
        cents = compute_centroids_np(data, assigns, cents)
        last_update[changed] = 1
        return assigns, cents, time.perf_counter() - t0

    ref_a, ref_c, _ = one_rep()
    ref_ab, ref_cb = ref_a.tobytes(), ref_c.tobytes()
    total = 0.0
    for _ in range(repeats):
        a, c, el = one_rep()
        if a.tobytes() != ref_ab or c.tobytes() != ref_cb:
            raise RuntimeError("mini-batch epoch-2 rep diverged from priming rep")
        total += el
    return total / repeats


# =======================================================================================
# per-impl runners: select init (timed) -> cluster (timed) -> single-iter -> epoch-2.
# Each returns a per-init measurement dict. `scheme` is the job's init scheme.
# =======================================================================================
def _epoch2_off(R):
    """R <= 0 turns the epoch-2 measurement OFF (--no-epoch2); epoch2_s is recorded NaN.

    Epoch-2 is R+1 replays of ONE pass, so its cost is (R+1)/n_iters times the convergence
    run it accompanies — measured at 3.7x for numba mini-batch, 17.8x for numba online and
    32.6x for Fortran, i.e. ~68% of a CPU cell and ~91% of a GPU cell. It is also completely
    independent of the convergence and single-iter measurements (it replays from a fresh
    post-init snapshot and shares no state), so splitting it into its own job costs nothing
    and lets the convergence data land first.
    """
    return R is None or R <= 0


def run_numba_online_ls(scheme, prep, K, i_seed, proportion, m_seed, R):
    from init_schemes import (ASSIGNMENT_SCHEMES, select_init_centroids,
                              select_init_partition)
    from streamed_hartigan import run_hartigan_kmeans_ls_timed
    data, true_labels = prep["data"], prep["true_labels"]
    part = scheme in ASSIGNMENT_SCHEMES
    t0 = time.perf_counter()
    if part:                                                       # timed: part of conv
        lab0 = select_init_partition(scheme, data, K, i_seed)
        init = np.zeros((K, data.shape[1]), dtype=np.float64)      # (K,d) buffer only
    else:
        lab0 = None
        init = select_init_centroids(scheme, data, K, i_seed)      # timed: part of conv
    c, l, _loss, n_iters, postinit = run_hartigan_kmeans_ls_timed(
        data, init.copy(), MAX_ITERS, init_labels=lab0)
    conv = time.perf_counter() - t0
    # epoch-2 replays from CENTROIDS by construction, so it is not defined for a
    # partition-defined scheme; the RP jobs pass --no-epoch2 anyway.
    epoch2 = (float("nan") if (_epoch2_off(R) or part)
              else _epoch2_online_numba(data, init, K, R))
    return dict(nmi=nmi_score(true_labels, l), sse=sse_f64(data, c, l),
                conv_s=conv, singleit_s=postinit / max(int(n_iters), 1),
                epoch2_s=epoch2, n_iters=int(n_iters),
                certified=float("nan"), fp_movers=float("nan"))


def run_numba_mb_cert(scheme, prep, K, i_seed, proportion, m_seed, R):
    from init_schemes import (ASSIGNMENT_SCHEMES, select_init_centroids,
                              select_init_partition)
    from mini_batch import mini_batch_hartigan_nb_certified_timed
    data, true_labels = prep["data"], prep["true_labels"]
    part = scheme in ASSIGNMENT_SCHEMES
    t0 = time.perf_counter()
    if part:                                                       # timed: part of conv
        lab0 = select_init_partition(scheme, data, K, i_seed)
        init = np.zeros((K, data.shape[1]), dtype=np.float64)      # (K,d) buffer only
    else:
        lab0 = None
        init = select_init_centroids(scheme, data, K, i_seed)      # timed: part of conv
    rng_mb = np.random.default_rng(m_seed)                         # fresh per cell
    diag = {}                                                      # observational only
    a, c, cert, counter, _zero, postinit = mini_batch_hartigan_nb_certified_timed(
        data, init.copy(), init.copy(), K, rng_mb, proportion, CERT_MAX_ITERS, diag=diag,
        init_labels=lab0)
    conv = time.perf_counter() - t0
    epoch2 = (float("nan") if (_epoch2_off(R) or part)             # see run_numba_online_ls
              else _epoch2_mb_numba(data, init, K, proportion, m_seed, R))
    return dict(nmi=nmi_score(true_labels, a), sse=sse_f64(data, c, a),
                conv_s=conv, singleit_s=postinit / max(int(counter), 1),
                epoch2_s=epoch2, n_iters=int(counter),
                certified=float(bool(cert)), fp_movers=float("nan"),
                empty=empty_flag(a, K),          # numba's own `zero` flag agrees exactly
                capped=float(bool(diag.get("capped", False))),
                confinement=float(diag.get("confinement", float("nan"))),
                cycle_pass=(float(diag["cycle_pass"]) if diag.get("cycle_pass", -1) >= 0
                            else float("nan")))


# ---- JAX (certified mini-batch and Lloyd) ---------------------------------------------
_JAXCTX = None


def _jax_ctx():
    """Lazily import JAX + cache the jitted mini-batch step ONCE per process (creating a new
    jax.jit wrapper per call would recompile every call)."""
    global _JAXCTX
    if _JAXCTX is None:
        import jax
        import jax.numpy as jnp
        from jax_minibatch_hartigan import (
            run_minibatch_hartigan_kmeans_certified, _minibatch_hartigan_step, _would_any_move)
        from _common_functions_for_minibatch import (
            assign_clusters as jx_assign, compute_centroids as jx_centroids)
        from _lloyd import run_lloyd_kmeans
        _JAXCTX = dict(
            jax=jax, jnp=jnp,
            run_mb_cert=run_minibatch_hartigan_kmeans_certified,
            jit_step=jax.jit(_minibatch_hartigan_step, static_argnums=(2, 3)),
            would_any_move=_would_any_move,
            assign=jx_assign, centroids=jx_centroids,
            run_lloyd=run_lloyd_kmeans)
    return _JAXCTX


def _mb_params(n, K, proportion):
    """(batch_size, patience, tolerance) — delegates to mini_batch.mb_batch_params so the JAX
    path uses the SAME leave-one-per-cluster cap and effective-proportion patience as numba."""
    from mini_batch import mb_batch_params
    bs, patience = mb_batch_params(n, K, proportion)
    tol = math.floor(0.01 * n)
    return bs, patience, tol


def _epoch2_mb_jax(J, dj, ij, key, proportion, repeats):
    jax, jnp = J["jax"], J["jnp"]
    n, K = dj.shape[0], ij.shape[0]
    bs, _patience, tol = _mb_params(n, K, proportion)
    labels0 = J["assign"](ij, dj)
    cents0 = J["centroids"](dj, labels0, K)
    carry = (cents0, labels0, jnp.asarray(0.0), jnp.asarray(0), jnp.asarray(0), key)
    jax.block_until_ready(carry)
    ref = J["jit_step"](carry, dj, bs, tol)          # priming rep (compiles), untimed
    jax.block_until_ready(ref)
    ref_c, ref_l = np.asarray(ref[0]).tobytes(), np.asarray(ref[1]).tobytes()
    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = J["jit_step"](carry, dj, bs, tol)
        jax.block_until_ready(out)
        total += time.perf_counter() - t0
        if np.asarray(out[0]).tobytes() != ref_c or np.asarray(out[1]).tobytes() != ref_l:
            raise RuntimeError("jax epoch-2 rep diverged from priming rep")
    return total / repeats


# XLA compiles a fresh executable per (shape, batch_size, patience, max_iters, tolerance).
# Every JAX runner is warmed by _warm_jax_runner (below), called from process_dataset: it runs
# the runner's own timed code path once, untimed, so exactly the executables the timed calls
# use are compiled before the clock starts -- and nothing they don't use (numba, by contrast,
# compiles per type-signature and is covered by warmup_numba).
#
# An earlier warm-up compiled a fixed list of executables that did not match the certified
# runner (it built tolerance=floor(0.01*n) and max_iters=2 variants, while the timed call uses
# _resume_minibatch_inner with max_iters=CERT_MAX_ITERS, tolerance=0, plus _would_any_move), so
# for n >= 100 the first timed start per (process, shape, proportion) compiled inside the clock.
# The shipped JAX timings (bestof_jaxtime_*_29208159) were produced with _warm_jax_runner and
# tests/gate_jax_compile_in_clock.py proves no compile fires inside a timed start.
_WARMED_JAX = set()


def _warm_jax_runner(impl, runner, scheme, prep, d, n, K, dataset_id, pi, proportion, swept):
    """Compile everything a JAX runner's timed region needs, OUTSIDE the clock.

    Runs the runner itself once, untimed, on this very dataset, and discards the result. The
    timed starts that follow then reuse the executables it compiled: same shapes, same static
    arguments (batch size, patience, max_iters, tolerance), same init path (random centroid or
    random partition), and nothing else is compiled. Once per process per (impl, scheme, n, d,
    K, proportion) -- everything an XLA compile key can depend on here.

    The seeds use init_id = N_INITS, which no timed start uses (they are 0..N_INITS-1), so the
    warm-up is a different initialisation of the same dataset and touches no timed state. The
    certified runner's fixed-point check is also compiled explicitly, in case this one run ran
    out of budget before ever reaching a check.
    """
    key = (impl, scheme, n, d, K, float(proportion))
    if key in _WARMED_JAX:
        return
    i_seed = init_seed(d, n, K, dataset_id, N_INITS)
    m_seed = mb_seed(d, n, K, dataset_id, N_INITS, pi) if swept else 0
    runner(scheme, prep, K, i_seed, float(proportion) if swept else -1.0, m_seed, -1)
    if impl == "jax_mb_cert":
        J = _jax_ctx()
        dj = prep["dj"]
        l0 = J["assign"](dj[:K], dj)
        J["jax"].block_until_ready(J["would_any_move"](dj, J["centroids"](dj, l0, K), l0))
    _WARMED_JAX.add(key)


def run_jax_mb_cert(scheme, prep, K, i_seed, proportion, m_seed, R):
    """JAX certified mini-batch — the paper's MAIN algorithm on the backend carrying the
    headline speedup.

    singleit_s is measured INSIDE this run: the certified driver's post-initialization clock
    (return_postinit=True) covers the mini-batch iterations and the certificate checks they
    trigger, and excludes initialization (the nearest-centroid assignment, or the supplied
    partition's block means). singleit_s = postinit / n_iters, where n_iters counts mini-batch
    iterations only -- the same definition numba_mb_cert uses. (Until 2026-10-11 it came from a
    separate tolerance-stopped rerun of each start; that rerun is gone.)
    """
    from init_schemes import (ASSIGNMENT_SCHEMES, select_init_centroids,
                              select_init_partition)
    J = _jax_ctx()
    jax, jnp = J["jax"], J["jnp"]
    data, true_labels, dj = prep["data"], prep["true_labels"], prep["dj"]
    key = jax.random.PRNGKey(int(m_seed % (2 ** 31 - 1)))   # compiled off the clock by _warm_jax_runner
    part = scheme in ASSIGNMENT_SCHEMES

    t0 = time.perf_counter()
    if part:
        lj = jnp.asarray(select_init_partition(scheme, data, K, i_seed))
        ij = jnp.zeros((K, data.shape[1]))           # carries K only; never read as centroids
    else:
        lj = None
        ij = jnp.asarray(select_init_centroids(scheme, data, K, i_seed))
    c, l, loss, counter, cert, postinit = J["run_mb_cert"](
        dj, ij, key, proportion, CERT_MAX_ITERS, init_labels=lj, return_postinit=True)
    jax.block_until_ready((c, l, loss))
    conv = time.perf_counter() - t0

    epoch2 = (float("nan") if (_epoch2_off(R) or part)   # see run_numba_online_ls
              else _epoch2_mb_jax(J, dj, ij, key, proportion, R))
    return dict(nmi=nmi_score(true_labels, np.asarray(l)),
                sse=sse_f64(data, np.asarray(c), np.asarray(l)),
                conv_s=conv, singleit_s=postinit / max(int(counter), 1), epoch2_s=epoch2,
                n_iters=int(counter), certified=float(bool(cert)),
                fp_movers=float("nan"),
                empty=empty_flag(np.asarray(l), K),   # jax mb has NO internal detection
                capped=float(int(counter) >= CERT_MAX_ITERS))

# ---- Fortran AS-136 (native ctypes; libas136.so must be built -> gfortran) ------------
def _fp_movers(data, cents, labels, K):
    """Number of points that would still move under the Hartigan criterion given (cents,
    populations) — the fixed-point audit that quantifies AS-136's live(l2) guard deviation
    (0 = a genuine fixed point). Distances via the GEMM form (no (n,K,d) tensor)."""
    data = np.asarray(data, dtype=np.float64)
    cents = np.asarray(cents, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(data)
    pops = np.bincount(labels, minlength=K).astype(np.float64)
    xn = (data * data).sum(1)[:, None]
    cn = (cents * cents).sum(1)[None, :]
    # errstate: macOS Accelerate BLAS raises a spurious "divide by zero in matmul" FPE flag
    # on clean inputs (same artifact the batch-Hartigan study guards against); the result is
    # correct. Harmless on Linux BLAS.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sq = np.maximum(xn + cn - 2.0 * (data @ cents.T), 0.0)      # (n, K)
    w = sq * (pops / (pops + 1.0))[None, :]
    rows = np.arange(n)
    own = pops[labels]
    w[rows, labels] = np.where(own <= 1, -1.0,
                               sq[rows, labels] * own / np.maximum(own - 1.0, 1.0))
    return int(np.sum(np.argmin(w, axis=1) != labels))


def _run_fortran(scheme, prep, K, i_seed, R, uselive):
    from init_schemes import select_init_centroids
    import hartigan_fortran_ctypes as hf
    data, true_labels, data_f = prep["data"], prep["true_labels"], prep["data_f"]
    t0 = time.perf_counter()
    init = select_init_centroids(scheme, data, K, i_seed)            # timed: part of conv
    labels, niter, postinit, _ifault, cents = hf.run_kmns(data_f, init, MAX_ITERS, uselive)
    conv = time.perf_counter() - t0
    # epoch-2: mean post-init time of an iter=1 run = the FIRST outer pass alone (kmns's
    # post-init clock excludes the redone init). R replays -> a stable first-pass time.
    if _epoch2_off(R):
        epoch2 = float("nan")
    else:
        ep2 = 0.0
        for _ in range(R):
            _l, _ni, p1, _if, _c = hf.run_kmns(data_f, init, 1, uselive)
            ep2 += p1
        epoch2 = ep2 / R
    return dict(nmi=nmi_score(true_labels, labels), sse=sse_f64(data, cents, labels),
                conv_s=conv, singleit_s=postinit / max(niter, 1), epoch2_s=epoch2,
                n_iters=niter, certified=float("nan"),
                fp_movers=float(_fp_movers(data, cents, labels, K)))


def run_fortran_ls(scheme, prep, K, i_seed, proportion, m_seed, R):
    return _run_fortran(scheme, prep, K, i_seed, R, uselive=1)


def run_fortran_nols(scheme, prep, K, i_seed, proportion, m_seed, R):
    return _run_fortran(scheme, prep, K, i_seed, R, uselive=0)


def run_jax_lloyd(scheme, prep, K, i_seed, proportion, m_seed, R):
    """Lloyd's algorithm (JAX) — the standard k-means baseline, for the paper's quality table.

    Every quality figure here is computed over the same N_INITS initializations per dataset, so
    Lloyd runs under the same protocol (same datasets, same initializations) to be comparable.

    NOT swept over proportions (Lloyd has no batch), so it records proportion = -1 and one row per
    cell, exactly like the online impls. `n_iters` counts Lloyd iterations: one full assignment
    plus one centroid recomputation over all n points.

    singleit_s and epoch2_s are deliberately NaN. `_run_lloyd_inner` does its own Lloyd init
    inside the jitted while_loop, so there is no post-init clock comparable to the one the other
    runners report, and emitting conv_s/n_iters there would silently mix init cost into a
    per-iteration figure. conv_s IS comparable (same boundaries as every other impl: init
    selection + the driver, blocked).

    RANDOM PARTITION. The partition comes from select_init_partition with this (dataset, init)'s
    i_seed -- the same call, and therefore the same labels, that numba_online_ls, numba_mb_cert
    and jax_mb_cert received -- and enters via run_lloyd_kmeans(init_labels=). Lloyd's step from
    a grouping is centroid update then reassignment, so the partition is used, not discarded.
    n_iters then counts that first reassignment (see run_lloyd_kmeans).
    """
    from init_schemes import (ASSIGNMENT_SCHEMES, select_init_centroids,
                              select_init_partition)
    J = _jax_ctx()
    jax, jnp = J["jax"], J["jnp"]
    data, true_labels, dj = prep["data"], prep["true_labels"], prep["dj"]
    part = scheme in ASSIGNMENT_SCHEMES

    t0 = time.perf_counter()
    if part:                                                   # timed: part of conv
        lj = jnp.asarray(select_init_partition(scheme, data, K, i_seed))
        ij = jnp.zeros((K, data.shape[1]))           # carries K only; never read as centroids
    else:
        lj = None
        ij = jnp.asarray(select_init_centroids(scheme, data, K, i_seed))   # timed: part of conv
    c, l, loss, counter = J["run_lloyd"](dj, ij, MAX_ITERS, init_labels=lj)
    jax.block_until_ready((c, l, loss))
    conv = time.perf_counter() - t0

    return dict(nmi=nmi_score(true_labels, np.asarray(l)),
                sse=sse_f64(data, np.asarray(c), np.asarray(l)),
                conv_s=conv, singleit_s=float("nan"), epoch2_s=float("nan"),
                n_iters=int(counter), certified=float("nan"), fp_movers=float("nan"),
                empty=empty_flag(np.asarray(l), K))


# registry: impl -> (backend, algo, proportion_swept, runner)
IMPLS = {
    "jax_lloyd":       ("gpu", "lloyd", False, run_jax_lloyd),
    "numba_online_ls": ("cpu", "online", False, run_numba_online_ls),
    "numba_mb_cert":   ("cpu", "minibatch", True, run_numba_mb_cert),
    "jax_mb_cert":     ("gpu", "minibatch", True, run_jax_mb_cert),
    "fortran_ls":      ("cpu", "online", False, run_fortran_ls),
    "fortran_nols":    ("cpu", "online", False, run_fortran_nols),
}


# ---- aggregation of the N_INITS per-init dicts into one CSV row ------------------------
def _std(a):
    a = np.asarray(a, dtype=np.float64)
    return float(a.std(ddof=1)) if a.size > 1 else 0.0


def empty_flag(labels, K):
    """1.0 if this run's final partition has FEWER than K non-empty clusters (an empty-cluster
    failure), else 0.0. Impl-agnostic and validated against numba's own `zero` flag: on 3,000
    reproduced study runs the two agreed exactly, and on 2,000 control runs both were 0.

    Mini-batch can empty a cluster because it reassigns a whole sampled batch against FIXED
    start-of-pass populations, so every member of a small cluster can leave at once — the
    singleton pin that protects online Hartigan (streamed_hartigan.py:58-59) never engages.
    JAX mini-batch has NO internal detection (an emptied cluster silently becomes a
    zero-vector centroid), so this post-hoc check is the only signal for it."""
    return float(len(np.unique(np.asarray(labels))) < int(K))


def aggregate(per_init):
    nmi = np.array([r["nmi"] for r in per_init], dtype=np.float64)
    sse = np.array([r["sse"] for r in per_init], dtype=np.float64)
    conv = [r["conv_s"] for r in per_init]
    sit = [r["singleit_s"] for r in per_init]
    ep2 = [r["epoch2_s"] for r in per_init]
    nit = [r["n_iters"] for r in per_init]
    cert = np.array([r["certified"] for r in per_init], dtype=np.float64)
    fp = np.array([r["fp_movers"] for r in per_init], dtype=np.float64)
    emp = np.array([r.get("empty", float("nan")) for r in per_init], dtype=np.float64)
    cap = np.array([r.get("capped", float("nan")) for r in per_init], dtype=np.float64)
    cyc = np.array([r.get("confinement", float("nan")) for r in per_init], dtype=np.float64)
    cycp = np.array([r.get("cycle_pass", float("nan")) for r in per_init], dtype=np.float64)
    imin = int(np.argmin(sse))                       # the selected (best-SSE) run
    cert_frac = float("nan") if np.all(np.isnan(cert)) else float(np.nanmean(cert))
    fp_mean = float("nan") if np.all(np.isnan(fp)) else float(np.nanmean(fp))
    # fraction of this cell's inits whose final partition lost a cluster (see empty_flag)
    empty_frac = float("nan") if np.all(np.isnan(emp)) else float(np.nanmean(emp))
    # fraction of inits truncated by the pass budget (all mini-batch impls: n_iters >= budget)
    capped_frac = float("nan") if np.all(np.isnan(cap)) else float(np.nanmean(cap))
    # oscillation observables (numba mini-batch only; NaN elsewhere -- see CycleTracker):
    #   confinement_mean = mean fraction of passes revisiting an already-seen state. CONTINUOUS
    #     on purpose: a boolean "cycled" flag fires on ~15% of healthy runs, while confinement
    #     separates by severity (grid <=0.46, p_max ordinary 0.55-0.75, pathological 0.99).
    #   cycle_pass_mean  = mean pass of the first lag>=2 revisit, over the inits that had one
    confinement_mean = float("nan") if np.all(np.isnan(cyc)) else float(np.nanmean(cyc))
    cycle_pass_mean = (float("nan") if np.all(np.isnan(cycp))
                       else float(np.nanmean(cycp)))
    return dict(
        n_inits=len(per_init),
        nmi_mean=float(nmi.mean()), nmi_std=_std(nmi), nmi_max=float(nmi.max()),
        sse_mean=float(sse.mean()), sse_std=_std(sse), sse_min=float(sse.min()),
        nmi_at_min_sse=float(nmi[imin]),
        conv_mean_s=float(np.mean(conv)), conv_std_s=_std(conv),
        singleit_mean_s=float(np.mean(sit)), singleit_std_s=_std(sit),
        epoch2_mean_s=float(np.mean(ep2)), epoch2_std_s=_std(ep2),
        n_iters_mean=float(np.mean(nit)), n_iters_std=_std(nit),
        certified_frac=cert_frac, fp_movers_mean=fp_mean, empty_frac=empty_frac,
        capped_frac=capped_frac, confinement_mean=confinement_mean,
        cycle_pass_mean=cycle_pass_mean)


CSV_COLUMNS = [
    "init_scheme", "backend", "algo", "impl", "d", "N", "K", "proportion",
    "dataset_id", "n_inits",
    "nmi_mean", "nmi_std", "nmi_max", "sse_mean", "sse_std", "sse_min", "nmi_at_min_sse",
    "conv_mean_s", "conv_std_s", "singleit_mean_s", "singleit_std_s",
    "epoch2_mean_s", "epoch2_std_s", "n_iters_mean", "n_iters_std",
    "certified_frac", "fp_movers_mean",
    "empty_frac",          # fraction of the cell's inits that lost a cluster (mini-batch risk)
    "cert_budget",         # the per-run pass budget in force (CERT_MAX_ITERS / MAX_ITERS)
    "capped_frac",         # fraction of inits truncated by that budget
    "confinement_mean",    # mean fraction of passes revisiting a state (numba mb; severity)
    "cycle_pass_mean",     # mean pass of the first revisit, over the inits that cycled
]


def _fmt(v):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.9g}"
    return str(v)


# ---- one dataset: run all requested impls over its inits -> list of row dicts ----------
def _prepare(data, true_labels, impls):
    """Backend-specific dataset forms built ONCE per dataset (device transfer amortized —
    only the tiny per-init init transfer is inside the convergence clock)."""
    prep = {"data": data, "true_labels": true_labels}
    if any(i.startswith("jax") for i in impls):
        prep["dj"] = _jax_ctx()["jnp"].asarray(data)
    if any(i.startswith("fortran") for i in impls):
        prep["data_f"] = np.asfortranarray(data.astype(np.float64))
    return prep


_GRID_ONLY = False          # set by --grid-only; see the flag's help text for why it matters


def proportions_for(n, K, grid=None):
    """This cell's mini-batch proportion list under the leave-one-per-cluster constraint.

    The batch may never contain every point of a cluster, so batch_size <= n - K and the
    largest usable sampling rate is p_max = (n - K)/n. Rather than silently CAPPING a
    requested proportion (which would collapse several grid points onto the same batch size —
    at n=32,K=16 the sizes for 0.25/0.5/0.75/0.9 are 8/16/16/16), the sweep keeps only the grid
    values that fit strictly under the cap and then STOPS at p_max. Every reported proportion is
    therefore exactly the one used, and no two entries share a batch size.

        n=32,K=16    -> [0.25, 0.5]                    (p_max = 0.5)
        n=32,K=4     -> [0.25, 0.5, 0.75, 0.875]
        n=83,K=16    -> [0.25, 0.5, 0.75, 0.8072]
        n=10000,K=4  -> [0.25, 0.5, 0.75, 0.9, 0.9996]
    """
    grid = PROPORTIONS if grid is None else grid
    b_max = max(1, n - K)
    out, seen = [], set()
    for p in grid:
        b = math.ceil(p * n)
        if b < b_max and b not in seen:          # strictly under the cap -> exact, distinct
            out.append(float(p))
            seen.add(b)
    if b_max not in seen and not _GRID_ONLY:     # top point: the largest constrained batch
        out.append(b_max / n)
    return out


def process_dataset(scheme, d, n, K, dataset_id, impls):
    data, true_labels = generate_dataset(n, d, K, dataset_seed(d, n, K, dataset_id))
    prep = _prepare(data, true_labels, impls)
    rows = []
    cell_props = proportions_for(n, K)
    for impl in impls:
        # per-impl resume: skip an impl whose rows for THIS cell are already complete in the
        # --resume CSV(s), unless it is listed in --fresh-impls (data invalidated -> recompute).
        if _RESUME_DONE and impl not in _FRESH_IMPLS:
            if _RESUME_DONE.get((impl, d, n, K, dataset_id), 0) >= rows_needed(impl, n, K):
                continue
        backend, algo, swept, runner = IMPLS[impl]
        props = cell_props if swept else [-1]
        for pi, proportion in enumerate(props):
            if impl.startswith("jax"):           # compile everything the timed starts use, untimed
                _warm_jax_runner(impl, runner, scheme, prep, d, n, K, dataset_id, pi,
                                 proportion, swept)
            per_init = []
            for init_id in range(N_INITS):
                i_seed = init_seed(d, n, K, dataset_id, init_id)
                m_seed = mb_seed(d, n, K, dataset_id, init_id, pi) if swept else 0
                per_init.append(runner(scheme, prep, K, i_seed,
                                       float(proportion) if swept else -1.0,
                                       m_seed, R_EPOCH2))
            agg = aggregate(per_init)
            rows.append(dict(init_scheme=scheme, backend=backend, algo=algo, impl=impl,
                             d=d, N=n, K=K, proportion=proportion, dataset_id=dataset_id,
                             cert_budget=(CERT_MAX_ITERS if swept else MAX_ITERS),
                             **agg))
    return rows


def warmup_numba(impls):
    """Compile the numba paths once (per process), UNTIMED, before any recorded run."""
    if not any(i.startswith("numba") for i in impls):
        return
    rng = np.random.default_rng(0)
    data = np.ascontiguousarray(
        np.vstack([rng.normal(size=(20, 8)) + 20.0 * rng.normal(size=(1, 8))
                   for _ in range(3)]), dtype=np.float64)
    prep = {"data": data, "true_labels": np.zeros(len(data), dtype=int)}
    K, i_seed = 3, 12345
    if "numba_online_ls" in impls:
        run_numba_online_ls("random_centroid", prep, K, i_seed, -1.0, 0, 2)
    if "numba_mb_cert" in impls:
        run_numba_mb_cert("random_centroid", prep, K, i_seed, 0.5, 777, 2)


_RESUME_DONE = {}          # (impl, d, N, K, dataset_id) -> rows already present in prior CSV(s)
_FRESH_IMPLS = frozenset()  # impls to ALWAYS recompute, ignoring _RESUME_DONE


def _worker_init(scheme, impls, resume_done=None, fresh_impls=()):
    global _RESUME_DONE, _FRESH_IMPLS
    if resume_done is not None:                      # inherited via initargs (fork AND spawn)
        _RESUME_DONE = resume_done
    _FRESH_IMPLS = frozenset(fresh_impls)
    warmup_numba(impls)                              # each worker compiles once, untimed


def open_csv(path):
    f = open(path, "w", buffering=1)
    f.write(",".join(CSV_COLUMNS) + "\n")
    return f


def write_rows(f, rows):
    for r in rows:
        f.write(",".join(_fmt(r[c]) for c in CSV_COLUMNS) + "\n")


def rows_needed(impl, n, K):
    """Rows ONE impl contributes for one (d,N,K,dataset) cell: online = 1 (proportion -1),
    mini-batch = this cell's proportion count (cell-dependent — see proportions_for)."""
    return len(proportions_for(n, K)) if IMPLS[impl][2] else 1


def completed_impl_cells(paths):
    """{(impl, d, N, K, dataset_id): row_count} from prior CSV(s). PER-IMPL, so a run can
    resume the impls whose data is still valid while recomputing others from scratch."""
    import csv as _csv
    from collections import Counter
    cnt = Counter()
    for path in paths:
        with open(path, newline="") as fh:
            for row in _csv.DictReader(fh):
                try:
                    key = (row["impl"], int(row["d"]), int(row["N"]), int(row["K"]),
                           int(row["dataset_id"]))
                except (KeyError, ValueError, TypeError):
                    continue                    # skip a truncated/malformed final line
                cnt[key] += 1
    return dict(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-scheme", default="random_centroid")
    ap.add_argument("--backend-group", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--dims", default="", help="comma list to restrict D (e.g. 100,1000)")
    ap.add_argument("--n-values", default="",
                    help="comma list to restrict N (e.g. 32,217,913). Same role as --dims; "
                         "lets a probe select a specific slice of the grid.")
    ap.add_argument("--k-values", default="", help="comma list to restrict K (e.g. 4,16)")
    ap.add_argument("--grid-only", action="store_true",
                    help="drop the p_max top proportion, keeping only the {0.25,0.5,0.75,0.9} grid "
                         "points that fit strictly under the cap. p_max is where the certified "
                         "pathology lives (all float64 certification failures are at "
                         "p_max, max n_iters 4,889 there vs 206 at grid), so it dominates cost: "
                         "~86%% of a certified GPU run's GPU-hours for ~20%% of the rows. The "
                         "headline best-of-N numbers use the best GRID proportion per cell, so "
                         "grid-only is sufficient for them and ~7x cheaper.")
    ap.add_argument("--impls", default="",
                    help="comma list to restrict impls (default: all in the backend group)")
    ap.add_argument("--workers", type=int, default=1,
                    help="CPU worker processes (datasets in parallel; single-thread each)")
    ap.add_argument("--datasets", type=int, default=0,
                    help="override N_DATASETS (0=default; use a few, e.g. 2, to CALIBRATE "
                         "per-cell time at full fidelity before the real run)")
    ap.add_argument("--n-inits", type=int, default=0, help="override N_INITS (0=default)")
    ap.add_argument("--repeats-epoch2", type=int, default=0,
                    help="override R_EPOCH2 (0=default)")
    ap.add_argument("--no-epoch2", action="store_true",
                    help="SKIP the epoch-2 measurement entirely; epoch2_mean_s/epoch2_std_s "
                         "are written NaN. Epoch-2 is ~68%% of a CPU cell and ~91%% of a GPU "
                         "cell and is independent of convergence/single-iter, so it can be "
                         "run afterwards as its own job.")
    ap.add_argument("--csv", default=os.path.join(_HERE, "bestof_init_results.csv"))
    ap.add_argument("--resume", default="",
                    help="comma-sep CSV path(s)/globs; skip (impl,d,N,K,dataset) rows already "
                         "complete in them (PER-IMPL, to finish a timed-out run).")
    ap.add_argument("--resume-online", default="",
                    help="comma-sep CSV path(s)/globs trusted for the ONLINE impls only — use for "
                         "PRE-FIX data, whose mini-batch rows are invalid (empty-cluster bug) but "
                         "whose online/fortran rows are still good. Combine with --resume pointing "
                         "at the new post-fix parts so repeat submissions don't redo valid work.")
    ap.add_argument("--fresh-impls", default="",
                    help="impls to ALWAYS recompute even if --resume has them (data invalidated). "
                         "Accepts impl names, or the shorthands 'minibatch' / 'online'. E.g. "
                         "--resume old.csv --fresh-impls minibatch  -> resume the online impls, "
                         "recompute every mini-batch impl from scratch.")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (use with --num-shards)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the REMAINING tasks across N nodes (task i goes to shard i %% N)")
    args = ap.parse_args()
    if args.grid_only:            # must be set BEFORE any proportions_for() call
        globals()['_GRID_ONLY'] = True
        print('--grid-only: p_max excluded from every cell')

    global N_DATASETS, N_INITS, R_EPOCH2
    if args.datasets:
        N_DATASETS = args.datasets
    if args.n_inits:
        N_INITS = args.n_inits
    if args.repeats_epoch2:
        R_EPOCH2 = args.repeats_epoch2
    if args.no_epoch2:
        R_EPOCH2 = -1                       # <= 0 -> _epoch2_off() -> epoch2_s = NaN

    dims = [int(x) for x in args.dims.split(",")] if args.dims else D_TO_TEST
    if args.impls:
        impls = [x.strip() for x in args.impls.split(",")]
    else:
        impls = [i for i, v in IMPLS.items() if v[0] == args.backend_group]
    bad = [i for i in impls if i not in IMPLS]
    if bad:
        raise SystemExit(f"unknown impls: {bad}")

    print(f"bestof_init | scheme={args.init_scheme} group={args.backend_group} "
          f"impls={impls} dims={dims} workers={args.workers} smoke={SMOKE} "
          f"datasets={N_DATASETS} inits={N_INITS} R={R_EPOCH2} csv={args.csv}", flush=True)

    n_vals = ([int(x) for x in args.n_values.split(",")] if args.n_values
              else list(N_TO_TEST))
    k_vals = ([int(x) for x in args.k_values.split(",")] if args.k_values else K_TO_TEST)
    tasks = [(args.init_scheme, d, int(n), K, ds)
             for d in dims for n in n_vals for K in k_vals
             for ds in range(N_DATASETS)]

    # --- shard FIRST (on the full, deterministic task list) so concurrent array shards own a
    #     fixed, disjoint slice regardless of resume timing; then each shard skips its done cells.
    if args.num_shards > 1:
        k = args.shard % args.num_shards
        tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == k]
        print(f"shard {k}/{args.num_shards}: {len(tasks)} tasks in this shard's slice", flush=True)
    # --- resume: PER-IMPL. An impl's rows for a cell are skipped when already complete in the
    #     prior CSV(s); an impl named in --fresh-impls is always recomputed. A whole cell is
    #     dropped only when EVERY requested impl is already complete for it.
    global _RESUME_DONE, _FRESH_IMPLS
    fresh = set()
    for x in (y.strip() for y in args.fresh_impls.split(",") if y.strip()):
        if x in ("minibatch", "mb"):
            fresh |= {i for i, v in IMPLS.items() if v[1] == "minibatch"}
        elif x == "online":
            fresh |= {i for i, v in IMPLS.items() if v[1] == "online"}
        else:
            fresh.add(x)
    bad_fresh = [i for i in fresh if i not in IMPLS]
    if bad_fresh:
        raise SystemExit(f"unknown --fresh-impls: {bad_fresh}")
    _FRESH_IMPLS = frozenset(fresh)
    def _expand(spec):
        import glob
        out = []
        for p in (x.strip() for x in spec.split(",") if x.strip()):
            m = glob.glob(p)
            out.extend(m if m else ([p] if os.path.exists(p) else []))
        return out

    paths = _expand(args.resume) if args.resume else []
    paths_online = _expand(args.resume_online) if args.resume_online else []
    if paths or paths_online:
        _RESUME_DONE = completed_impl_cells(paths) if paths else {}
        if paths_online:
            # PRE-FIX data: trust the online/fortran rows, ignore every mini-batch row
            online = {i for i, v in IMPLS.items() if v[1] == "online"}
            for k, v in completed_impl_cells(paths_online).items():
                if k[0] in online:
                    _RESUME_DONE[k] = max(_RESUME_DONE.get(k, 0), v)

        def _cell_done(t):
            for i in impls:
                if i in fresh:
                    return False                     # a fresh impl always needs this cell
                if _RESUME_DONE.get((i, t[1], t[2], t[3], t[4]), 0) < rows_needed(i, t[2], t[3]):
                    return False
            return True

        before = len(tasks)
        tasks = [t for t in tasks if not _cell_done(t)]
        per_impl = {i: sum(1 for k in _RESUME_DONE if k[0] == i) for i in impls}
        print(f"resume(all impls): {paths}\n"
              f"resume(online only, pre-fix): {paths_online}\n"
              f"  resumable cells per impl: {per_impl}\n"
              f"  fresh (always recompute): {sorted(fresh) or 'none'}\n"
              f"  cells fully done and dropped: {before - len(tasks)}; {len(tasks)} remain",
              flush=True)
    if not tasks:
        print("nothing to do — all requested cells already complete", flush=True)
        open_csv(args.csv).close()               # header-only CSV so the merge still finds it
        return

    f = open_csv(args.csv)
    t_start = time.time()

    if args.backend_group == "cpu" and args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork" if sys.platform != "darwin" else "spawn")
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=(args.init_scheme, impls, _RESUME_DONE,
                                tuple(sorted(_FRESH_IMPLS)))) as pool:
            it = pool.imap_unordered(
                _star_process, [(t, impls) for t in tasks], chunksize=1)
            for rows in it:
                write_rows(f, rows)
    else:
        warmup_numba(impls)                          # single-process warmup
        done = 0
        for t in tasks:
            write_rows(f, process_dataset(*t, impls))
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                print(f"  {done}/{len(tasks)} datasets "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    f.close()
    print(f"DONE — {args.csv} ({time.time() - t_start:.0f}s)", flush=True)


def _star_process(arg):
    task, impls = arg
    return process_dataset(*task, impls)


if __name__ == "__main__":
    main()
