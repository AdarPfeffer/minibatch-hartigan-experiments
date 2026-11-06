"""fp_hartigan_ctypes.py — call the float64 and float32 AS-136 variants from Python.

Build:  make lib-fp        # -> libfp136.so  (needs gfortran)

    from fp_hartigan_ctypes import run_kmns_dp, run_kmns_sp
    res64 = run_kmns_dp(data, init, max_iters=10000, uselive=0)
    res32 = run_kmns_sp(data, init, max_iters=10000, uselive=0)

Both take the SAME float64 arrays; `run_kmns_sp` casts to float32 at the boundary, which is
exactly the question under test (identical inputs, different arithmetic width).

`uselive=0` is the default here, unlike the production wrapper. The live-set guard is an
optimisation that lets OPTRA skip clusters it believes cannot win; with it ON, AS-136 can
stop short of a true fixed point (measured: fp_movers > 0 with the guard ON, exactly 0 with
it OFF). This experiment is about whether float32 reaches the fixed point, so the guard must
not be a competing explanation.

Each result carries the full decision trace: an (n, 4) int32 array of
(stage, point, from_cluster, to_cluster), stage 1 = OPTRA, 2 = QTRAN.
"""
import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.environ.get("FP_HARTIGAN_LIB", os.path.join(_HERE, "libfp136.so"))
TRACE_MAX = 2_000_000                      # must match FPTMAX in fp_trace.f

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise OSError(f"{_LIB_PATH} not found — run 'make lib-fp' in minibatch_hartigan_tests/experiments/fortran "
                      f"(needs gfortran), or set FP_HARTIGAN_LIB.")
    lib = ctypes.CDLL(_LIB_PATH)
    dp, sp, ip = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
                  ctypes.POINTER(ctypes.c_int))
    # kmns ( a, m, n, c, k, ic1, ic2, nc, an1, an2, ncp, d, itran, live, iter, wss, ifault )
    lib.kmns_dp_.argtypes = [dp, ip, ip, dp, ip, ip, ip, ip, dp, dp, ip, dp, ip, ip, ip,
                             dp, ip]
    lib.kmns_sp_.argtypes = [sp, ip, ip, sp, ip, ip, ip, ip, sp, sp, ip, sp, ip, ip, ip,
                             sp, ip]
    for f in (lib.kmns_dp_, lib.kmns_sp_):
        f.restype = None
    for name, args in (("fptrc_reset_", []), ("fptrc_count_", [ip]),
                       ("fptrc_overflow_", [ip]), ("fptrc_get_", [ip, ip, ip]),
                       ("hartigan_set_uselive_", [ip]), ("hartigan_set_verbose_", [ip]),
                       ("hartigan_get_niter_", [ip]), ("hartigan_get_postinit_", [dp])):
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = None
    _lib = lib
    return lib


def _trace_out(lib):
    n = ctypes.c_int(0)
    lib.fptrc_count_(ctypes.byref(n))
    ovf = ctypes.c_int(0)
    lib.fptrc_overflow_(ctypes.byref(ovf))
    n = int(n.value)
    buf = np.zeros((n, 4), dtype=np.int32) if n else np.zeros((0, 4), dtype=np.int32)
    if n:
        got = ctypes.c_int(0)
        # Fortran buf(4,nmax) is column-major -> a C-contiguous (n,4) array matches record-wise
        lib.fptrc_get_(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                       ctypes.byref(ctypes.c_int(n)), ctypes.byref(got))
        buf = buf[:int(got.value)]
    return buf, bool(ovf.value)


def _run(single, data, init, max_iters, uselive):
    lib = _load()
    rt = np.float32 if single else np.float64
    ptr = ctypes.POINTER(ctypes.c_float if single else ctypes.c_double)
    a = np.asfortranarray(np.asarray(data, dtype=rt))
    m, n = a.shape
    c = np.asfortranarray(np.asarray(init, dtype=rt).copy())     # (K, N), overwritten
    K = c.shape[0]
    if c.shape[1] != n:
        raise ValueError(f"init dim {c.shape[1]} != data dim {n}")

    ic1 = np.zeros(m, dtype=np.int32)
    ic2 = np.zeros(m, dtype=np.int32)
    nc = np.zeros(K, dtype=np.int32)
    an1 = np.zeros(K, dtype=rt)
    an2 = np.zeros(K, dtype=rt)
    ncp = np.zeros(K, dtype=np.int32)
    dwork = np.zeros(m, dtype=rt)
    itran = np.zeros(K, dtype=np.int32)
    live = np.zeros(K, dtype=np.int32)
    wss = np.zeros(K, dtype=rt)
    cm, cn, ck = ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(K)
    citer, ifault = ctypes.c_int(int(max_iters)), ctypes.c_int(0)
    ip = ctypes.POINTER(ctypes.c_int)

    lib.hartigan_set_uselive_(ctypes.byref(ctypes.c_int(int(uselive))))
    lib.hartigan_set_verbose_(ctypes.byref(ctypes.c_int(0)))
    lib.fptrc_reset_()
    fn = lib.kmns_sp_ if single else lib.kmns_dp_
    fn(a.ctypes.data_as(ptr), ctypes.byref(cm), ctypes.byref(cn), c.ctypes.data_as(ptr),
       ctypes.byref(ck), ic1.ctypes.data_as(ip), ic2.ctypes.data_as(ip),
       nc.ctypes.data_as(ip), an1.ctypes.data_as(ptr), an2.ctypes.data_as(ptr),
       ncp.ctypes.data_as(ip), dwork.ctypes.data_as(ptr), itran.ctypes.data_as(ip),
       live.ctypes.data_as(ip), ctypes.byref(citer), wss.ctypes.data_as(ptr),
       ctypes.byref(ifault))

    trace, ovf = _trace_out(lib)
    niter, post = ctypes.c_int(0), ctypes.c_double(0.0)
    lib.hartigan_get_niter_(ctypes.byref(niter))
    lib.hartigan_get_postinit_(ctypes.byref(post))
    return dict(labels=ic1.astype(np.int64) - 1,          # AS-136 is 1-based
                centroids=np.ascontiguousarray(c).astype(np.float64),
                wss=np.asarray(wss, dtype=np.float64).sum(),
                n_iters=int(niter.value), postinit_s=float(post.value),
                ifault=int(ifault.value), trace=trace, trace_overflow=ovf)


def run_kmns_dp(data, init, max_iters=10000, uselive=0):
    """AS-136 in float64. See module docstring for why uselive defaults to 0."""
    return _run(False, data, init, max_iters, uselive)


def run_kmns_sp(data, init, max_iters=10000, uselive=0):
    """AS-136 in float32 — same source, same order of operations, narrower arithmetic."""
    return _run(True, data, init, max_iters, uselive)
