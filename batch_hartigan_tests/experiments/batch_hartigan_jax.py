"""
batch_hartigan_jax.py — JAX/GPU port of the Batch-Hartigan convergence failure-mode
study (high-dimensional restructure).

The FAILURE DETECTION is kept byte-for-byte the numpy reference (`batch_hartigan.py`):
the `loop_checker` hash-memo (copied verbatim below), the `np.array_equal` convergence
guard, the empty-cluster `zero` check, and the zero > loop > max_iter precedence all run
on the HOST. Only the per-pass compute — the (n,K) weighted-distance argmin and the
centroid recompute — is re-implemented on the GPU (`compare_batch_hart_jax`). So the three
failure modes (empty cluster, oscillation cycle, max-iter) are detected by the same logic;
the only re-implemented numeric is the per-pass distance, which can differ from numpy only
on an exact tie (measure-zero), quantified by `audit_jax_vs_numpy.py`.

Precision: float64 (jax_enable_x64) — the numpy study is a float64 dynamical system.

Two corrections vs the mini-batch JAX candidate (`minibatch_hartigan_tests/experiments/jax_minibatch_hartigan.py`)
required for exact numpy parity:
  * `zero` is flagged from the INCOMING partition's populations (numpy: bincount(assigns)
    before reassignment, batch_hartigan.py:212), and `counter` increments before the zero
    return (:316-318).
  * the singleton stay-factor is 0.0 (batch_hartigan.py:222), not the candidate's -1.0
    (jax_minibatch_hartigan.py:131) — they differ on exact-duplicate ties.

Data generation is isotropic (mean + sqrt(v)*normal, O(n*d) not the O(d^3) dense-Cholesky
`multivariate_normal`) and reproducible: keyed by the trial coordinates via a numpy
SeedSequence, generated on the GPU with jax.random (Threefry -> deterministic and
cross-platform). The audit's byte-identical cross-check holds by generating once and feeding
BOTH numpy and JAX the same pulled-to-host array.
"""
import os

# float64 must be set before any array is created.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
import jax                                             # noqa: E402
jax.config.update("jax_enable_x64", True)

# Persistent compilation cache: compiling the float64 GEMM for each (B,n,d) shape is the main
# per-shape overhead (Triton autotuning at high d). Caching compiled executables to disk means
# each shape is compiled ONCE EVER — reused across regimes, re-runs, and (critically) across the
# N sharded jobs, so the overnight sweep never recompiles from scratch on each Ada. Guarded so an
# unknown config key on a given jax version degrades gracefully instead of crashing.
_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.jax_cache"))
for _k, _v in (("jax_compilation_cache_dir", _cache_dir),
               ("jax_persistent_cache_min_compile_time_secs", 0.0),
               ("jax_persistent_cache_min_entry_size_bytes", 0)):
    try:
        jax.config.update(_k, _v)
    except Exception:
        pass
import jax.numpy as jnp                                # noqa: E402
import numpy as np                                     # noqa: E402
from functools import partial                          # noqa: E402

ROOT_SEED = 20260724  # fixed root entropy; recorded so the whole study is reproducible.


# ─────────────────────────────────────────────────────────────────────────────
# Host failure-detection — COPIED VERBATIM from batch_hartigan.py:238-265 (this directory).
# (Copied, not imported, because importing that module runs its module-scope demo.)
# ─────────────────────────────────────────────────────────────────────────────
def loop_checker(assigns, memory, counter):
    """Detect whether this assignment vector has been seen before, and if so, how many
    iterations ago (the cycle length). Only the hash of `assigns.tobytes()` is stored, so
    detection is fast and catches cycles of ANY period >= 2 (convergence is checked before
    this, so a fixed point is never counted as a size-1 loop).

    Returns (alphabetical): loop (bool), loop_size (int, 0 if none), memory (updated dict).
    """
    h = hash(assigns.tobytes())
    if h in memory:
        return True, counter - memory[h], memory
    memory[h] = counter
    return False, 0, memory


# ─────────────────────────────────────────────────────────────────────────────
# JAX kernels (unjitted cores; jitted single + vmapped batched wrappers built below).
# assign / centroid helpers mirror minibatch_hartigan_tests/experiments/_common_functions_for_minibatch.py:27-72.
# ─────────────────────────────────────────────────────────────────────────────
def _assign_core(centroids, data):
    """Nearest-centroid assignment (init step). argmin of ||x-c||^2 via the GEMM expansion
    with ||x||^2 dropped (constant across clusters -> argmin-invariant)."""
    cn = jnp.sum(centroids * centroids, axis=1)                       # (K,)
    xc = jnp.matmul(data, centroids.T, precision="highest")          # (n, K)
    return jnp.argmin(cn[None, :] - 2.0 * xc, axis=1).astype(jnp.int32)


def _centroids_core(data, labels, K):
    """Per-cluster mean via a one-hot GEMM. An empty cluster -> 0 vector (harmless: the
    driver returns immediately on `zero`, discarding these centroids)."""
    onehot = (labels[:, None] == jnp.arange(K)[None, :]).astype(data.dtype)  # (n, K)
    sums = jnp.matmul(onehot.T, data, precision="highest")                   # (K, d)
    counts = jnp.sum(onehot, axis=0)                                         # (K,)
    return sums / jnp.maximum(counts, 1.0)[:, None]


def _pass_core(data, centroids, assigns, K):
    """One full batch-Hartigan reassignment pass — the GPU twin of numpy
    `compare_batch_hart` (batch_hartigan.py:186-235).

    Returns (new_assigns int32 [n], new_centroids f64 [K,d], zero_flag bool). `zero_flag`
    reflects the INCOMING partition's populations (matching numpy), so the driver records
    the same `counter` on an empty-cluster stop.
    """
    n = data.shape[0]
    # Incoming populations (one-hot sum -> vmap-safe, unlike jnp.bincount).
    pop = jnp.sum(assigns[:, None] == jnp.arange(K)[None, :], axis=0).astype(data.dtype)  # (K,)
    zero_flag = jnp.any(pop == 0.0)

    # Squared distances via ||x||^2 + ||c||^2 - 2 x.c as a GEMM (keep ||x||^2: distances are
    # scaled per-cluster before the argmin). GEMM rather than the (n,K,d) broadcast because
    # that tensor is ~5.4 GB at d=10000. clamp the rare float cancellation to 0.
    xn = jnp.sum(data * data, axis=1)[:, None]                       # (n, 1)
    cn = jnp.sum(centroids * centroids, axis=1)[None, :]             # (1, K)
    dist = jnp.maximum(
        xn + cn - 2.0 * jnp.matmul(data, centroids.T, precision="highest"), 0.0)  # (n, K)

    # Hartigan transfer scaling (numpy compare_batch_hart:219-229; singleton stay = 0.0).
    move = pop / (pop + 1.0)                                         # (K,) n/(n+1)
    stay = jnp.where(pop == 1.0, 0.0, pop / (pop - 1.0))            # (K,) n/(n-1), 0 if alone
    scale = jnp.broadcast_to(move[None, :], (n, K))
    scale = scale.at[jnp.arange(n), assigns].set(stay[assigns])     # own cluster -> stay
    new_assigns = jnp.argmin(dist * scale, axis=1).astype(jnp.int32)
    new_centroids = _centroids_core(data, new_assigns, K)
    return new_assigns, new_centroids, zero_flag


# jitted single-trial wrappers (K is a compile-time constant).
_assign = jax.jit(_assign_core)
_centroids = jax.jit(_centroids_core, static_argnums=(2,))
compare_batch_hart_jax = jax.jit(_pass_core, static_argnums=(3,))

# vmapped batched wrappers (leading batch axis on the array args; K stays static).
_assign_v = jax.jit(jax.vmap(_assign_core, in_axes=(0, 0)))
_centroids_v = jax.jit(jax.vmap(_centroids_core, in_axes=(0, 0, None)), static_argnums=(2,))
_pass_v = jax.jit(jax.vmap(_pass_core, in_axes=(0, 0, 0, None)), static_argnums=(3,))


# ─────────────────────────────────────────────────────────────────────────────
# Single-trial driver — the AUDIT ORACLE. Mirrors batch_hartigan.py:301-322 line-for-line,
# with the per-pass compute on the GPU and the detection on the host.
# ─────────────────────────────────────────────────────────────────────────────
def batch_hartigan_jax(data, init_centers, K, max_iter=500):
    """Run batch-Hartigan on one trial. `data`/`init_centers` are jax arrays (float64).

    Returns (alphabetical): assigns [n], centroids [K,d], counter, loop, loop_size, zero —
    the same 6-tuple as numpy `batch_hartigan`.
    """
    a_dev = _assign(init_centers, data)                 # init nearest-centroid assignment
    c_dev = _centroids(data, a_dev, K)                  # centroids of the init partition
    a = np.asarray(a_dev)
    old = np.full(a.shape, -1, dtype=a.dtype)
    memory = {}
    counter = 0
    while counter < max_iter:
        if np.array_equal(a, old):                      # clean convergence (fixed point)
            return a, np.asarray(c_dev), counter, False, 0, False
        loop, loop_size, memory = loop_checker(a, memory, counter)
        if loop:                                        # oscillation cycle detected
            return a, np.asarray(c_dev), counter, True, loop_size, False
        old = a.copy()
        a_dev, c_dev, z_dev = compare_batch_hart_jax(data, c_dev, a_dev, K)
        a = np.asarray(a_dev)
        counter += 1
        if bool(z_dev):                                 # empty cluster (from the pass input)
            return a, np.asarray(c_dev), counter, False, 0, True
    return a, np.asarray(c_dev), counter, False, 0, False   # budget exhausted (max_iter)


# ─────────────────────────────────────────────────────────────────────────────
# Batched production driver — B trials of ONE (K, cluster_size, dim) shape advanced through
# one vmapped pass per step, with the identical per-lane host detection and lane compaction.
# Per-lane semantics are IDENTICAL to batch_hartigan_jax (verified by the audit's
# batched-vs-single parity check). Compaction only regroups which lanes share a GPU call.
# ─────────────────────────────────────────────────────────────────────────────
def batch_hartigan_jax_batched(data, init, K, max_iter=500, compact_below=0.4):
    """Run a batch of B trials. `data` (B,n,d) and `init` (B,K,d) are jax arrays (float64).

    Returns a list of B 6-tuples (assigns, centroids, counter, loop, loop_size, zero), one
    per lane in the original order.
    """
    B0 = int(data.shape[0])
    a_dev = _assign_v(init, data)                       # (B0, n)
    c_dev = _centroids_v(data, a_dev, K)                # (B0, K, d)
    A = np.array(a_dev)                                 # (B0, n) host source of truth (writable)
    n = A.shape[1]
    old = np.full((B0, n), -1, dtype=A.dtype)
    counter = np.zeros(B0, dtype=np.int64)
    memory = [dict() for _ in range(B0)]
    result = [None] * B0

    active = np.arange(B0)          # original lane ids present in the device arrays (aligned order)
    M = B0
    step = 0
    while active.size > 0 and step < max_iter:
        # 1) host detection at the top of the step (convergence, then loop) — mirror numpy.
        advance_pos = []
        for pos in range(active.size):
            b = int(active[pos])
            if result[b] is not None:                   # already finished, dead weight until compaction
                continue
            if np.array_equal(A[b], old[b]):
                result[b] = (A[b].copy(), np.asarray(c_dev[pos]), int(counter[b]), False, 0, False)
            else:
                loop, lsz, memory[b] = loop_checker(A[b], memory[b], int(counter[b]))
                if loop:
                    result[b] = (A[b].copy(), np.asarray(c_dev[pos]), int(counter[b]),
                                 True, int(lsz), False)
                else:
                    advance_pos.append(pos)
        if not advance_pos:
            break
        # 2) one vmapped pass over ALL M current lanes (finished lanes computed but ignored).
        newA_dev, newc_dev, z_dev = _pass_v(data, c_dev, jnp.asarray(A[active]), K)
        newA = np.asarray(newA_dev)
        zf = np.asarray(z_dev)
        c_dev = newc_dev
        # 3) host apply for the advancing lanes.
        for pos in advance_pos:
            b = int(active[pos])
            old[b] = A[b]
            A[b] = newA[pos]
            counter[b] += 1
            if bool(zf[pos]):
                result[b] = (A[b].copy(), np.asarray(c_dev[pos]), int(counter[b]), False, 0, True)
        step += 1
        # 4) compaction — drop finished lanes from the device arrays once they thin out, so the
        #    tail runs in a small batch. Pure work-regrouping: per-lane results are unaffected.
        keep = np.array([result[int(active[pos])] is None for pos in range(active.size)])
        n_keep = int(keep.sum())
        if 0 < n_keep < compact_below * M:
            keep_pos = np.nonzero(keep)[0]
            data = data[keep_pos]
            c_dev = c_dev[keep_pos]
            active = active[keep_pos]
            M = active.size
    # 5) survivors that never terminated -> max_iter (counter == max_iter).
    for pos in range(active.size):
        b = int(active[pos])
        if result[b] is None:
            result[b] = (A[b].copy(), np.asarray(c_dev[pos]), int(counter[b]), False, 0, False)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reproducible isotropic data generation (on-GPU, float64). Same LAW as the numpy
# `generate_ground_truth_data` (centers ~ N(0, inter*I); points ~ N(center, intra*I)) but
# O(n*d) instead of the O(d^3) dense-Cholesky multivariate_normal — required for d=10000.
# ─────────────────────────────────────────────────────────────────────────────
def trial_seed(dim, K, intra, inter, cluster_size, trial_id, root=ROOT_SEED):
    """Deterministic 32-bit seed for one trial, keyed by its full coordinate tuple. Any trial
    is regenerable in isolation from its coordinates."""
    ss = np.random.SeedSequence([int(root), int(dim), int(K), int(intra), int(inter),
                                 int(cluster_size), int(trial_id)])
    return int(ss.generate_state(1)[0])


def _gen_one(key, cluster_size, K, dim, intra, inter):
    kc, kp, ki = jax.random.split(key, 3)
    centers = jnp.sqrt(inter) * jax.random.normal(kc, (K, dim), dtype=jnp.float64)      # (K,d)
    pts = centers[:, None, :] + jnp.sqrt(intra) * jax.random.normal(
        kp, (K, cluster_size, dim), dtype=jnp.float64)                                  # (K,cs,d)
    data = pts.reshape(K * cluster_size, dim)
    idx = jax.random.choice(ki, K * cluster_size, (K,), replace=False)                  # distinct rows
    init = data[idx]
    return data, init


_gen_batch = jax.jit(jax.vmap(_gen_one, in_axes=(0, None, None, None, None, None)),
                     static_argnums=(1, 2, 3))


def generate_batch(dim, K, intra, inter, cluster_size, trial_ids):
    """Generate a batch of trials' (data, init_centers) on-device, float64. `trial_ids` is a
    sequence of trial indices; each is seeded independently by `trial_seed` (intra/inter are
    the same across a regime and broadcast to all lanes).

    Returns (data [B,n,d], init [B,K,d]) as jax arrays. n = cluster_size * K.
    """
    keys = jnp.stack([jax.random.PRNGKey(
        trial_seed(dim, K, intra, inter, cluster_size, t)) for t in trial_ids])
    data, init = _gen_batch(keys, int(cluster_size), int(K), int(dim),
                            float(intra), float(inter))
    return data, init


def generate_one(dim, K, intra, inter, cluster_size, trial_id):
    """Single-trial (data, init_centers) as jax float64 arrays — used by the audit oracle."""
    data, init = generate_batch(dim, K, intra, inter, cluster_size, [trial_id])
    return data[0], init[0]
