"""fp_untouched_check.py — reproduce the float32 non-termination with NOTHING of mine in the way.

WHY. The cycling result was produced by fp_as136_{dp,sp}.f, which my generator derived from
as136.f (renamed symbols, re-wrapped one line, two `call fptrc_add` inserted). Those files pass
a byte-exact equivalence audit and two independent re-implementations reproduce the cycle — but
all of that still shares my transformation. This check removes it entirely.

HOW. `as136.f` is compiled TWICE, unmodified, straight from the file the production study uses:

    libas136_ref64.so     gfortran -O3 -std=legacy -ffixed-form            (double precision)
    libas136_ref32.so     gfortran -O3 -std=legacy -ffixed-form -freal-8-real-4

`-freal-8-real-4` demotes every `double precision` in the source to 4-byte real at compile
time. Not one character of as136.f changes. No rename, no trace call, no re-wrap, no generator.

If the float32 build cycles on the same seeds, nothing I wrote is implicated in the finding.

BOTH live-set settings are tested. `uselive=0` is what the sweep used; `uselive=1` is the
original AS-136 path. If the cycle appears under both, the g_uselive modification (which
predates this work and lives in the production as136.f) is not implicated either.

Run:  python fp_untouched_check.py       (after `make lib-ref` in this directory)
"""
import ctypes
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, ".."),
           os.path.join(_HERE, "..", "..", "..", "batch_hartigan_tests", "experiments")):
    sys.path.insert(0, _p)

import bestof_init_benchmark as B
from init_schemes import select_init_centroids

# The 10 runs that never terminated in the sweep, and 3 that were trace-identical (controls).
CASES = [(10000, 350, 4, 53), (10000, 5000, 4, 28), (10000, 5000, 4, 40),
         (10000, 6500, 4, 74), (10000, 3838, 8, 97), (10000, 6500, 8, 64),
         (10000, 8000, 8, 72), (10000, 10000, 8, 37), (10000, 6500, 16, 65),
         (10000, 10000, 16, 98)]
CONTROLS = [(10000, 32, 4, 0), (10000, 32, 4, 1), (10000, 32, 4, 3)]


def load(path, single):
    if not os.path.exists(path):
        sys.exit(f"{path} not found — run 'make lib-ref' in {_HERE}")
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)   # LOCAL: the two libs both export kmns_
    r = ctypes.POINTER(ctypes.c_float if single else ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.kmns_.argtypes = [r, ip, ip, r, ip, ip, ip, ip, r, r, ip, r, ip, ip, ip, r, ip]
    lib.kmns_.restype = None
    lib.hartigan_set_uselive_.argtypes = [ip]
    lib.hartigan_set_verbose_.argtypes = [ip]
    lib.hartigan_get_niter_.argtypes = [ip]
    return lib


def run(lib, single, data, init, max_iters, uselive):
    rt = np.float32 if single else np.float64
    ptr = ctypes.POINTER(ctypes.c_float if single else ctypes.c_double)
    ip = ctypes.POINTER(ctypes.c_int)
    a = np.asfortranarray(np.asarray(data, dtype=rt))
    m, n = a.shape
    c = np.asfortranarray(np.asarray(init, dtype=rt).copy())
    K = c.shape[0]
    ic1 = np.zeros(m, np.int32); ic2 = np.zeros(m, np.int32)
    nc = np.zeros(K, np.int32); ncp = np.zeros(K, np.int32)
    an1 = np.zeros(K, rt); an2 = np.zeros(K, rt); dw = np.zeros(m, rt)
    itran = np.zeros(K, np.int32); live = np.zeros(K, np.int32); wss = np.zeros(K, rt)
    cm, cn, ck = ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(K)
    it, ifl = ctypes.c_int(int(max_iters)), ctypes.c_int(0)
    lib.hartigan_set_uselive_(ctypes.byref(ctypes.c_int(int(uselive))))
    lib.hartigan_set_verbose_(ctypes.byref(ctypes.c_int(0)))
    lib.kmns_(a.ctypes.data_as(ptr), ctypes.byref(cm), ctypes.byref(cn),
              c.ctypes.data_as(ptr), ctypes.byref(ck), ic1.ctypes.data_as(ip),
              ic2.ctypes.data_as(ip), nc.ctypes.data_as(ip), an1.ctypes.data_as(ptr),
              an2.ctypes.data_as(ptr), ncp.ctypes.data_as(ip), dw.ctypes.data_as(ptr),
              itran.ctypes.data_as(ip), live.ctypes.data_as(ip), ctypes.byref(it),
              wss.ctypes.data_as(ptr), ctypes.byref(ifl))
    ni = ctypes.c_int(0)
    lib.hartigan_get_niter_(ctypes.byref(ni))
    return int(ni.value), int(ifl.value)


def main():
    lib64 = load(os.path.join(_HERE, "libas136_ref64.so"), False)
    lib32 = load(os.path.join(_HERE, "libas136_ref32.so"), True)
    MAXIT = 10000

    for uselive in (0, 1):
        tag = "uselive=0 (no live-set guard; what the sweep used)" if uselive == 0 \
              else "uselive=1 (original AS-136 live-set path)"
        print(f"\n{'='*76}\n{tag}\n{'='*76}")
        print(f"{'d':>6} {'N':>6} {'K':>3} {'trial':>5} | "
              f"{'f64 iters':>9} {'ifault':>6} | {'f32 iters':>9} {'ifault':>6} | verdict")
        n_cycle = 0
        for group, label in ((CASES, "CYCLED in the sweep"), (CONTROLS, "control")):
            print(f"-- {label} " + "-" * 50)
            for (d, N, K, t) in group:
                data, _ = B.generate_dataset(N, d, K, B.dataset_seed(d, N, K, t))
                init = select_init_centroids("random_centroid", data, K,
                                             B.init_seed(d, N, K, t, 0))
                i64, f64 = run(lib64, False, data, init, MAXIT, uselive)
                i32, f32 = run(lib32, True, data, init, MAXIT, uselive)
                cyc = (f32 == 2 and i32 >= MAXIT)
                n_cycle += cyc and group is CASES
                print(f"{d:>6} {N:>6} {K:>3} {t:>5} | {i64:>9} {f64:>6} | "
                      f"{i32:>9} {f32:>6} | {'NON-TERMINATING' if cyc else 'terminated'}")
        print(f"\n  {n_cycle}/{len(CASES)} of the sweep's cycling cases reproduce with the "
              f"UNTOUCHED as136.f at {tag.split()[0]}")


if __name__ == "__main__":
    main()
