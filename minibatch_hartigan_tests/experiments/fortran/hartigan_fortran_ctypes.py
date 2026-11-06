"""hartigan_fortran_ctypes.py — call the original AS-136 (Hartigan-Wong) Fortran from Python.

Build once (needs gfortran):   make lib     # -> libas136.so
Then:
    from hartigan_fortran_ctypes import run_kmns
    labels, niter, postinit_s, ifault, centroids = run_kmns(data, init, max_iters=10000,
                                                            uselive=1)

`data` is (n_points, n_dims), `init` is (K, n_dims). Internally AS-136 uses column-major
Fortran arrays and 1-based cluster labels; this wrapper converts. `uselive` toggles the
OPTRA live-set guard (1 = original AS-136; 0 = re-evaluate all clusters — same decisions,
more work). `postinit_s` = wall time of the pass loop alone (init excluded), and `niter` =
number of outer passes, both read from as136.f's saved outputs. Deterministic (RNG-free).
"""
import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.environ.get("HARTIGAN_FORTRAN_LIB", os.path.join(_HERE, "libas136.so"))

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise OSError(f"{_LIB_PATH} not found — build it with 'make lib' in "
                      f"minibatch_hartigan_tests/experiments/fortran (requires gfortran), or set HARTIGAN_FORTRAN_LIB.")
    lib = ctypes.CDLL(_LIB_PATH)
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    # kmns ( a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, iter, wss, ifault )
    # all Fortran args are by reference; default integer = int32, double = float64
    lib.kmns_.argtypes = [dp, ip, ip, dp, ip, ip, ip, ip, dp, dp, ip, dp, ip, ip, ip,
                          dp, ip]
    lib.kmns_.restype = None
    lib.hartigan_set_uselive_.argtypes = [ip]
    lib.hartigan_set_uselive_.restype = None
    lib.hartigan_set_verbose_.argtypes = [ip]
    lib.hartigan_set_verbose_.restype = None
    lib.hartigan_get_niter_.argtypes = [ip]
    lib.hartigan_get_niter_.restype = None
    lib.hartigan_get_postinit_.argtypes = [dp]
    lib.hartigan_get_postinit_.restype = None
    _lib = lib
    return lib


def gpu_available():
    """Fortran is CPU-only; exposed for API symmetry with the CUDA wrapper."""
    return False


def set_verbose(v):
    """Toggle AS-136's non-convergence warning (g_verbose; default 0 = silent). Off by
    default so the max_iters=1 epoch-2 timing replays don't flood stdout; IFAULT=2 is
    returned by run_kmns regardless, so non-convergence is never hidden from the caller."""
    lib = _load()
    lib.hartigan_set_verbose_(ctypes.byref(ctypes.c_int(int(v))))


def run_kmns(data, init, max_iters=10000, uselive=1):
    """Run AS-136 kmns. Returns (labels [0-based int64], niter, postinit_seconds, ifault,
    centroids [K, n_dims])."""
    lib = _load()
    # AS-136 naming: M = points, N = dims, K = clusters. Column-major (Fortran) storage.
    a = np.asfortranarray(np.asarray(data, dtype=np.float64))
    m, n = a.shape
    c = np.asfortranarray(np.asarray(init, dtype=np.float64).copy())   # (K, N), overwritten
    K = c.shape[0]
    if c.shape[1] != n:
        raise ValueError(f"init dim {c.shape[1]} != data dim {n}")

    ic1 = np.zeros(m, dtype=np.int32)
    ic2 = np.zeros(m, dtype=np.int32)
    nc = np.zeros(K, dtype=np.int32)
    an1 = np.zeros(K, dtype=np.float64)
    an2 = np.zeros(K, dtype=np.float64)
    ncp = np.zeros(K, dtype=np.int32)
    dwork = np.zeros(m, dtype=np.float64)
    itran = np.zeros(K, dtype=np.int32)
    live = np.zeros(K, dtype=np.int32)
    wss = np.zeros(K, dtype=np.float64)
    cm, cn, ck = ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(K)
    citer, ifault = ctypes.c_int(int(max_iters)), ctypes.c_int(0)
    dp = ctypes.POINTER(ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)

    def D(arr):
        return arr.ctypes.data_as(dp)

    def I(arr):
        return arr.ctypes.data_as(ip)

    lib.hartigan_set_uselive_(ctypes.byref(ctypes.c_int(int(uselive))))
    lib.kmns_(D(a), ctypes.byref(cm), ctypes.byref(cn), D(c), ctypes.byref(ck),
              I(ic1), I(ic2), I(nc), D(an1), D(an2), I(ncp), D(dwork), I(itran), I(live),
              ctypes.byref(citer), D(wss), ctypes.byref(ifault))

    niter = ctypes.c_int(0)
    post = ctypes.c_double(0.0)
    lib.hartigan_get_niter_(ctypes.byref(niter))
    lib.hartigan_get_postinit_(ctypes.byref(post))
    # ic1 is 1-based cluster ids; c is (K, N) with row l = cluster (l+1)'s centroid
    return (ic1.astype(np.int64) - 1, int(niter.value), float(post.value),
            int(ifault.value), np.ascontiguousarray(c))
