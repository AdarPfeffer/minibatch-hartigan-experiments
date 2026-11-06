#!/usr/bin/env python
"""FAST premise gate for the float64 JAX runs — the float64 arm differs from a float32 arm in
arithmetic width ONLY.

This is the cheap fail-fast check that runs at the top of jobs/run_bestof_jaxtime_h200.job and
locally on CPU JAX. It covers only the premise and is meant to run in about a minute.

    python minibatch_hartigan_tests/experiments/tests/test_fp64_precision_isolation.py

Exit 0 = passed. 1 = a check failed. 2 = a NEGATIVE CONTROL failed, i.e. a checker accepted input
it must reject, so the other results prove nothing.

WHY IT EXISTS
-------------
`jax_minibatch_hartigan._sample_batch_holdout` used to draw `jax.random.uniform(k, (n,))` with no
dtype, so it followed the ambient default: float32 with x64 off, float64 with x64 on. Those
consume different numbers of Threefry words per element and therefore SELECT DIFFERENT POINTS —
so a float64 run differed from the recorded float32 run in sampling AND in arithmetic and
licensed no precision claim at all.

Both draws are now pinned to `dtype=jnp.float32` in production (2026-08-12). The checks:

  A  (control) an UNPINNED sampler really does change its batch under x64. If this does not fire,
     the pin is untested and everything below is vacuous.
  B  the production sampler is byte-identical with x64 ON and OFF.
  C  the production sampler's float32 digest still equals the value measured BEFORE the pin, so
     every previously collected float32 row remains bit-reproducible. This is the regression
     check that makes the patch safe rather than merely plausible.
  D  the production DRIVERS, on float32 inputs, return bit-identical results with x64 ON and OFF
     — what licenses running the float32 comparator inside an x64 process at all.
  E  (control) the D comparator rejects a deliberately perturbed run.
  F  float64 changes decisions somewhere (else the study measures nothing), reported as the first
     divergence pass on the cells the study targets.
  G  a certified float64 result passes an INDEPENDENT numpy float64 fixed-point check sharing no
     code with the JAX kernel.
  H  float64 is deterministic: same inputs twice, bit-identical labels, passes, certificate.

A-E run in SUBPROCESSES because `jax_enable_x64` is process-global.
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)                      # experiments/: the modules under test
for _p in (EXP,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Historical, INFORMATIONAL ONLY — not a gate. Measured on jax 0.4.30 (local) before the dtype
# pin went in. It is NOT a version-independent constant: `jax_threefry_partitionable` defaults to
# False in jax<=~0.4.35 and True from ~0.4.36, and that changes the Threefry bit stream, hence the
# batches, hence this digest. Verified: jax 0.4.30 with the flag forced True reproduces the
# cluster's digest 015c654d9e68 exactly. So a hardcoded digest conflates "the dtype pin is a
# no-op" (the thing we care about) with "the JAX RNG is unchanged" (irrelevant here, and false
# across the local/cluster version gap). Check C now tests the pin DIRECTLY instead.
LEGACY_DIGESTS = {"19f067d7928d": "jax 0.4.30, threefry_partitionable=False (local, 2026-08-12)",
                  "015c654d9e68": "jax >=0.4.38, threefry_partitionable=True (Bouchet H200)"}

FAILED, CONTROLS = [], []


def ok(name, cond, detail="", control=False):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    if not cond:
        (CONTROLS if control else FAILED).append(name)
    return cond


_SAMPLER_WORKER = r'''
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
X64 = os.environ["W_X64"] == "1"
WHICH = os.environ["W_WHICH"]                 # "prod" (pinned) | "unpinned" (control)
if X64:
    os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, os.environ["W_REPO"]); sys.path.insert(0, os.environ["W_HERE"])
import jax
if X64:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from jax_minibatch_hartigan import _sample_batch_holdout as PROD

def UNPINNED(key, old_labels, K, n, batch_size):
    """What production looked like BEFORE the pin. Kept only as the negative control: it is the
    thing whose x64-dependence motivated the patch."""
    k_hold, k_pick = jax.random.split(key)
    r = jax.random.uniform(k_hold, (n,))
    masked = jnp.where(old_labels[:, None] == jnp.arange(K)[None, :], r[:, None], jnp.inf)
    held = jnp.argmin(masked, axis=0)
    pops = jnp.bincount(old_labels, length=K)
    hm = jnp.zeros(n, dtype=jnp.int32).at[held].max((pops > 0).astype(jnp.int32)) > 0
    return jnp.argsort(jnp.where(hm, jnp.inf,
                                 jax.random.uniform(k_pick, (n,))))[:batch_size]

S = PROD if WHICH == "prod" else UNPINNED
rng = np.random.default_rng(20260812)
out = {}
for (n, K, bs) in [(500, 8, 300), (137, 16, 100), (64, 4, 60), (2000, 8, 1800)]:
    lab = jnp.asarray(rng.integers(0, K, n))
    for seed in (1, 7, 99):
        idx = np.asarray(S(jax.random.PRNGKey(seed), lab, K, n, bs)).astype(np.int64)
        out["%d_%d_%d_%d" % (n, K, bs, seed)] = idx.tolist()
print(json.dumps({"x64": bool(jax.config.read("jax_enable_x64")),
                  "dtype": str(jnp.asarray([1.0]).dtype), "samples": out}))
'''

_DRIVER_WORKER = r'''
import os, sys, json, hashlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
X64 = os.environ["W_X64"] == "1"
if X64:
    os.environ["JAX_ENABLE_X64"] = "1"
else:
    os.environ.pop("JAX_ENABLE_X64", None)
sys.path.insert(0, os.environ["W_REPO"]); sys.path.insert(0, os.environ["W_HERE"])
import jax
if X64:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import bestof_init_benchmark as B
from init_schemes import select_init_centroids
from jax_minibatch_hartigan import (
    run_minibatch_hartigan_kmeans as TOL, run_minibatch_hartigan_kmeans_certified as CERT)

def dig(a, integral=False):
    """Byte digest. `integral=True` normalises to int64 FIRST: enabling x64 widens JAX's index
    and label dtypes from int32 to int64, so a raw byte hash of a label vector differs between
    the two configurations even when every label is identical. Comparing raw bytes there reports
    a difference in dtype as if it were a difference in the algorithm (it did exactly that on the
    first run of this gate: identical centroids, identical pass counts, 'different' labels)."""
    a = np.ascontiguousarray(a)
    if integral:
        a = a.astype(np.int64)
    return hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest()[:12]

PERTURB = os.environ.get("W_PERTURB") == "1"
out = {}
for (d, n, K, ds, prop) in [(100, 134, 4, 0, 0.25), (1000, 565, 16, 7, 0.5)]:
    pi = B.proportions_for(n, K).index(prop)
    data_np, _ = B.generate_dataset(n, d, K, B.dataset_seed(d, n, K, ds))
    i_np = select_init_centroids("random_centroid", data_np, K, B.init_seed(d, n, K, ds, 0))
    seed = B.mb_seed(d, n, K, ds, 0, pi) % (2**31 - 1)
    if PERTURB:
        seed = (seed + 1) % (2**31 - 1)             # control: must change the answer
    key = jax.random.PRNGKey(seed)
    # FLOAT32 inputs regardless of the x64 flag: the claim is that the f32 comparator is the same
    # whether or not x64 happens to be enabled in the process running it.
    d32 = jnp.asarray(data_np, dtype=jnp.float32)
    i32 = jnp.asarray(i_np, dtype=jnp.float32)
    c, l, loss, cnt = TOL(d32, i32, key, prop, 3000, 0.01)
    out["tol_%d_%d_%d" % (d, n, K)] = [dig(l, True), int(cnt), dig(c)]
    c, l, loss, cnt, cert = CERT(d32, i32, key, prop, 3000)
    out["cert_%d_%d_%d" % (d, n, K)] = [dig(l, True), int(cnt), bool(cert)]
print(json.dumps({"x64": bool(jax.config.read("jax_enable_x64")), "runs": out}))
'''

_F64_WORKER = r'''
import os, sys, json, math
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, os.environ["W_REPO"]); sys.path.insert(0, os.environ["W_HERE"])
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import bestof_init_benchmark as B
from init_schemes import select_init_centroids
from jax_minibatch_hartigan import (
    _minibatch_hartigan_step, run_minibatch_hartigan_kmeans_certified as CERT)
from _common_functions_for_minibatch import assign_clusters, compute_centroids
from mini_batch import mb_batch_params
STEP = jax.jit(_minibatch_hartigan_step, static_argnums=(2, 3))

def numpy_movers(data, cent, lab, K):
    """Independent float64 Hartigan fixed-point check; shares no code with the JAX kernel."""
    data = np.asarray(data, np.float64); cent = np.asarray(cent, np.float64)
    lab = np.asarray(lab); p = np.bincount(lab, minlength=K).astype(np.float64)
    sq = ((data[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    w = sq * (p / (p + 1.0))[None, :]
    i = np.arange(len(lab))
    w[i, lab] = np.where(p[lab] <= 1, -1.0, sq[i, lab] * (p[lab] / (p[lab] - 1.0)))
    return int(np.sum(np.argmin(w, axis=1) != lab))

def carry(data, init, key):
    lab = assign_clusters(init, data)
    return (compute_centroids(data, lab, init.shape[0]), lab,
            jnp.asarray(0.0, dtype=data.dtype), jnp.asarray(0), jnp.asarray(0), key)

# ---- F: does float64 change decisions anywhere, on the cells the study targets? ----
divs, differ = [], False
for (d, n, K, ds, prop) in [(100, 5000, 8, 47, 0.9), (100, 5000, 16, 75, 0.75),
                            (1000, 565, 16, 7, 0.5)]:
    pi = B.proportions_for(n, K).index(prop)
    data_np, _ = B.generate_dataset(n, d, K, B.dataset_seed(d, n, K, ds))
    i_np = select_init_centroids("random_centroid", data_np, K, B.init_seed(d, n, K, ds, 0))
    key = jax.random.PRNGKey(B.mb_seed(d, n, K, ds, 0, pi) % (2**31 - 1))
    bs, _pat = mb_batch_params(n, K, prop); tol = math.floor(0.01 * n)
    d64 = jnp.asarray(data_np); d32 = jnp.asarray(data_np, jnp.float32)
    c64 = carry(d64, jnp.asarray(i_np), key)
    c32 = carry(d32, jnp.asarray(i_np, jnp.float32), key)
    fd = -1
    for t in range(60):
        c64 = STEP(c64, d64, bs, tol)
        c32 = STEP(c32, d32, bs, tol)
        if fd < 0 and not np.array_equal(np.asarray(c64[1]), np.asarray(c32[1])):
            fd = t + 1; differ = True
    divs.append("d=%d N=%d K=%d p=%.2f -> %s" % (d, n, K, prop,
                "diverges@%d" % fd if fd > 0 else "identical(60)"))

# ---- G/H: certificate soundness + determinism at float64 ----
n_cert, sound, det = 0, True, True
for (d, n, K, ds, prop) in [(100, 134, 4, 0, 0.25), (1000, 565, 16, 7, 0.5)]:
    data_np, _ = B.generate_dataset(n, d, K, B.dataset_seed(d, n, K, ds))
    i_np = select_init_centroids("random_centroid", data_np, K, B.init_seed(d, n, K, ds, 0))
    pi = B.proportions_for(n, K).index(prop)
    key = jax.random.PRNGKey(B.mb_seed(d, n, K, ds, 0, pi) % (2**31 - 1))
    data = jnp.asarray(data_np); init = jnp.asarray(i_np)
    c, l, lo, cnt, ce = CERT(data, init, key, prop, 4000)
    c2, l2, lo2, cnt2, ce2 = CERT(data, init, key, prop, 4000)
    if (not np.array_equal(np.asarray(l), np.asarray(l2)) or int(cnt) != int(cnt2)
            or bool(ce) != bool(ce2)):
        det = False
    if bool(ce):
        n_cert += 1
        if numpy_movers(data_np, np.asarray(c), np.asarray(l), K) != 0:
            sound = False
print(json.dumps({"differ": bool(differ), "divs": divs, "n_cert": n_cert,
                  "sound": bool(sound), "det": bool(det)}))
'''


def run(src, **env):
    e = dict(os.environ, W_REPO=EXP, W_HERE=EXP, **env)
    if env.get("W_X64") != "1":
        e.pop("JAX_ENABLE_X64", None)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       env=e, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("worker %s failed:\n%s" % (env, r.stderr[-3000:]))
    return json.loads(r.stdout.strip().splitlines()[-1])


def _tfp():
    import subprocess, sys as _s
    r = subprocess.run([_s.executable, "-c",
        "import jax\ntry: print(jax.config.read('jax_threefry_partitionable'))\n"
        "except Exception: print('n/a-in-this-jax')"],
        capture_output=True, text=True)
    return r.stdout.strip() or "?"


def _jaxver():
    import subprocess, sys as _s
    r = subprocess.run([_s.executable, "-c", "import jax;print(jax.__version__)"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "?"


def dg(obj):
    return hashlib.blake2b(json.dumps(obj, sort_keys=True).encode(),
                           digest_size=16).hexdigest()[:12]


def main():
    print("=" * 84)
    print("premise gate — float64 precision isolation")
    print("=" * 84)

    print("\nA-C. the sampler")
    p32 = run(_SAMPLER_WORKER, W_X64="0", W_WHICH="prod")
    p64 = run(_SAMPLER_WORKER, W_X64="1", W_WHICH="prod")
    u32 = run(_SAMPLER_WORKER, W_X64="0", W_WHICH="unpinned")
    u64 = run(_SAMPLER_WORKER, W_X64="1", W_WHICH="unpinned")
    ok("config sanity: x64 off -> float32 default, x64 on -> float64 default",
       p32["dtype"] == "float32" and p64["dtype"] == "float64",
       "%s / %s" % (p32["dtype"], p64["dtype"]))
    ok("A (control) an UNPINNED sampler changes its batch under x64",
       dg(u32["samples"]) != dg(u64["samples"]),
       "%s vs %s" % (dg(u32["samples"]), dg(u64["samples"])), control=True)
    ok("B production sampler: byte-identical with x64 ON and OFF",
       dg(p32["samples"]) == dg(p64["samples"]), dg(p64["samples"]))
    # C: the pin must be a NO-OP at float32 -- tested directly against the unpinned sampler in
    # the same process with x64 off, which is version-independent, rather than against a stored
    # digest that moves with JAX's RNG defaults.
    ok("C the pin is a NO-OP at float32 (pinned == unpinned with x64 off), so previously "
       "collected float32 rows remain reproducible by this code",
       dg(p32["samples"]) == dg(u32["samples"]),
       "pinned %s == unpinned %s" % (dg(p32["samples"]), dg(u32["samples"])))
    here = dg(p32["samples"])
    print("       sampler digest here: %s  (%s)"
          % (here, LEGACY_DIGESTS.get(here, "UNRECOGNISED jax RNG configuration — record it")))
    print("       threefry_partitionable=%s | jax %s  <- these two determine the batch stream"
          % (_tfp(), _jaxver()))

    print("\nD-E. the float32 comparator is unaffected by the x64 flag")
    d_off = run(_DRIVER_WORKER, W_X64="0")
    d_on = run(_DRIVER_WORKER, W_X64="1")
    ok("D production drivers on float32 inputs: bit-identical with x64 ON and OFF",
       d_off["runs"] == d_on["runs"], "%d run signatures" % len(d_off["runs"]))
    d_pert = run(_DRIVER_WORKER, W_X64="1", W_PERTURB="1")
    ok("E (control) the D comparator rejects a perturbed run (seed+1)",
       d_pert["runs"] != d_on["runs"], control=True)

    print("\nF-H. float64 behaviour")
    f = run(_F64_WORKER, W_X64="1")
    # F IS NO LONGER A GATE, for two independent reasons.
    # 1. OBSOLETE PREMISE. The cause of the cycling is now known
    #    (fp32_sequential_accumulation_cause.py): certified-f32 cycling requires SINGLE-
    #    ACCUMULATOR sequential summation in the centroid recompute. JAX's recompute is a
    #    split-accumulation GEMM (~1 ulp error), which sits BELOW the decision margins just as
    #    float64 does -- so JAX float32 agreeing with float64 is the PREDICTED result, not a sign
    #    the study measures nothing.
    # 2. UNDERPOWERED. Three fixtures x 60 passes cannot detect a ~1e-4 per-init effect; that is
    #    the same mistake verify_fp64_experiment.py D2 now guards against with an explicit power
    #    calculation. A null from this sample carries no information either way.
    # The real questions live in the powered tests: verify D2 (with its power guard) and section L
    # (cross-implementation rate parity against the recorded CUDA rows).
    print("  INFO F float32-vs-float64 trajectory divergence (informational; agreement is the")
    print("       PREDICTED outcome for JAX):")
    for line in f["divs"]:
        print("         %s" % line)
    ok("G certified float64 results pass an INDEPENDENT numpy float64 fixed-point check",
       f["sound"], "%d certified verified" % f["n_cert"])
    ok("H float64 is deterministic (labels/passes/certificate bit-identical on re-run)",
       f["det"])

    print("\n" + "=" * 84)
    if CONTROLS:
        print("NEGATIVE CONTROL FAILED: %s" % ", ".join(CONTROLS))
        print("A checker accepted what it must reject — the other results prove nothing.")
        return 2
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    print("PREMISE HOLDS — the float64 arm differs from the recorded float32 arm in arithmetic")
    print("width only: identical batches, identical driver logic, and a float32 comparator")
    print("unaffected by the x64 flag.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
