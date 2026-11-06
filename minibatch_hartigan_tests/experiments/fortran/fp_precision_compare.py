"""fp_precision_compare.py — does float32 change Hartigan's decisions, or only its rounding?

THE QUESTION. Mini-batch Hartigan in float32 fails to certify convergence on a small number
of grid cells where float64 never does (28 cells / 114 inits vs 0). That could be an
implementation artefact of the mini-batch drivers, or it could be that at float32 the
algorithm itself no longer reaches a true fixed point. ONLINE Hartigan is the clean place to
settle it: AS-136 is the original, it is deterministic given (data, init), and it has no
sampling, no tolerance and no early-stop heuristic to confound the answer.

THE DESIGN. For each regime and trial:
  * generate the dataset and the initial centres with the STUDY'S OWN seeds
    (`dataset_seed` / `init_seed` from bestof_init_benchmark) — so these are the same
    datasets the big test uses, and the seed is recorded in every output row;
  * run AS-136 in float64 and in float32 — same source, same order of operations, cast at
    the boundary;
  * compare the two runs DECISION BY DECISION, not just at the end. Every committed transfer
    is recorded as (stage, point, from, to); the first index at which the two sequences
    disagree is reported, with both sides' records.

THE VERDICT COLUMNS. `dp_fp_movers` / `sp_fp_movers` count, IN FLOAT64, how many points would
still move out of the returned partition under Hartigan's own criterion. Zero means a true
fixed point. This is the answer to the actual question:
    sp_fp_movers == 0 everywhere  -> float32 reaches genuine fixed points; the mini-batch
                                     failures are implementation-level.
    sp_fp_movers >  0 anywhere    -> float32 stops short of a fixed point in the ORIGINAL
                                     algorithm; this is fundamental, not a driver bug.
Both are computed twice: once with the centroids the run returned, and once with float64
means recomputed from the labels (a fixed point is a property of the partition, so the
second is the stricter reading).

The live-set guard is OFF (`--uselive 0`). It is an optimisation that lets OPTRA skip
clusters it believes cannot win, and it is already known to stop short of a true fixed point
sometimes (fp_movers > 0 with it ON, exactly 0 with it OFF). Leaving it on would give the
experiment a second possible explanation for any non-fixed-point result.

Run:
  KMEANS_SMOKE=1 python fp_precision_compare.py --workers 4
  python fp_precision_compare.py --workers 32 --trials 10 --csv fp_precision.csv
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"                       # hard set, not setdefault

_HERE = os.path.dirname(os.path.abspath(__file__))
_MB = os.path.join(_HERE, "..")
for _p in (_HERE, _MB, os.path.join(_MB, "..", "..", "batch_hartigan_tests", "experiments")):
    sys.path.insert(0, _p)

import argparse
import csv
import multiprocessing as mp
import time

import numpy as np

import bestof_init_benchmark as B
from fp_hartigan_ctypes import run_kmns_dp, run_kmns_sp

COLUMNS = [
    "d", "N", "K", "trial", "dataset_seed", "init_seed", "uselive", "max_iters",
    # did they do the same thing?
    "trace_identical", "n_transfers_dp", "n_transfers_sp", "first_divergence",
    "div_stage_dp", "div_point_dp", "div_from_dp", "div_to_dp",
    "div_stage_sp", "div_point_sp", "div_from_sp", "div_to_sp",
    "trace_overflow_dp", "trace_overflow_sp",
    # did they end up in the same place?
    "labels_identical", "n_label_diffs", "n_iters_dp", "n_iters_sp",
    "ifault_dp", "ifault_sp", "wss_dp", "wss_sp", "wss_rel_diff",
    # THE VERDICT: would-movers evaluated in float64 out of each returned partition
    "fp_movers_dp", "fp_movers_sp", "fp_movers_dp_exact", "fp_movers_sp_exact",
    "postinit_s_dp", "postinit_s_sp",
]


def _exact_means(data, labels, K):
    """float64 cluster means of a partition — a fixed point is a property of the partition,
    so this is the stricter centroid to test against."""
    c = np.zeros((K, data.shape[1]), dtype=np.float64)
    for l in range(K):
        m = labels == l
        if m.any():
            c[l] = data[m].mean(axis=0)
    return c


def compare_one(args):
    d, n, K, trial, max_iters, uselive = args
    ds_seed = B.dataset_seed(d, n, K, trial)
    i_seed = B.init_seed(d, n, K, trial, 0)
    data, _true = B.generate_dataset(n, d, K, ds_seed)
    from init_schemes import select_init_centroids
    init = select_init_centroids("random_centroid", data, K, i_seed)

    r64 = run_kmns_dp(data, init, max_iters=max_iters, uselive=uselive)
    r32 = run_kmns_sp(data, init, max_iters=max_iters, uselive=uselive)

    t64, t32 = r64["trace"], r32["trace"]
    ncommon = min(len(t64), len(t32))
    first = -1
    if ncommon:
        neq = np.nonzero((t64[:ncommon] != t32[:ncommon]).any(axis=1))[0]
        if neq.size:
            first = int(neq[0])
    if first < 0 and len(t64) != len(t32):
        first = ncommon                       # identical prefix, one ran longer
    identical = first < 0

    def rec(t, i):
        return (list(map(int, t[i])) if 0 <= i < len(t) else [-1, -1, -1, -1])
    dv64, dv32 = rec(t64, first), rec(t32, first)

    l64, l32 = r64["labels"], r32["labels"]
    ndiff = int((l64 != l32).sum())

    fp = B._fp_movers
    return dict(
        d=d, N=n, K=K, trial=trial, dataset_seed=int(ds_seed), init_seed=int(i_seed),
        uselive=uselive, max_iters=max_iters,
        trace_identical=int(identical), n_transfers_dp=len(t64), n_transfers_sp=len(t32),
        first_divergence=first,
        div_stage_dp=dv64[0], div_point_dp=dv64[1], div_from_dp=dv64[2], div_to_dp=dv64[3],
        div_stage_sp=dv32[0], div_point_sp=dv32[1], div_from_sp=dv32[2], div_to_sp=dv32[3],
        trace_overflow_dp=int(r64["trace_overflow"]), trace_overflow_sp=int(r32["trace_overflow"]),
        labels_identical=int(ndiff == 0), n_label_diffs=ndiff,
        n_iters_dp=r64["n_iters"], n_iters_sp=r32["n_iters"],
        ifault_dp=r64["ifault"], ifault_sp=r32["ifault"],
        wss_dp=r64["wss"], wss_sp=r32["wss"],
        wss_rel_diff=abs(r32["wss"] - r64["wss"]) / max(abs(r64["wss"]), 1e-300),
        fp_movers_dp=fp(data, r64["centroids"], l64, K),
        fp_movers_sp=fp(data, r32["centroids"], l32, K),
        fp_movers_dp_exact=fp(data, _exact_means(data, l64, K), l64, K),
        fp_movers_sp_exact=fp(data, _exact_means(data, l32, K), l32, K),
        postinit_s_dp=r64["postinit_s"], postinit_s_sp=r32["postinit_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dims", default="")
    ap.add_argument("--n-values", default="")
    ap.add_argument("--k-values", default="")
    ap.add_argument("--max-iters", type=int, default=10000)
    ap.add_argument("--uselive", type=int, default=0,
                    help="0 = no live-set guard (reaches true fixed points). See docstring.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--csv", default=os.path.join(_HERE, "fp_precision.csv"))
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")] if args.dims else B.D_TO_TEST
    nvals = ([int(x) for x in args.n_values.split(",")] if args.n_values
             else [int(x) for x in B.N_TO_TEST])
    kvals = [int(x) for x in args.k_values.split(",")] if args.k_values else B.K_TO_TEST

    tasks = [(d, n, K, t, args.max_iters, args.uselive)
             for d in dims for n in nvals for K in kvals for t in range(args.trials)]
    if args.num_shards > 1:
        k = args.shard % args.num_shards
        tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == k]

    print(f"fp precision compare | AS-136 float64 vs float32 | uselive={args.uselive} "
          f"max_iters={args.max_iters}\n  dims={dims} K={kvals}\n  N={nvals}\n"
          f"  {args.trials} trials/regime -> {len(tasks)} runs, workers={args.workers}, "
          f"csv={args.csv}", flush=True)

    t0 = time.perf_counter()
    done = 0
    n_div = n_spfp = n_dpfp = 0
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        if args.workers > 1:
            pool = mp.get_context("fork").Pool(args.workers)
            it = pool.imap_unordered(compare_one, tasks)
        else:
            it = map(compare_one, tasks)
        for row in it:
            w.writerow(row)
            f.flush()
            done += 1
            n_div += (row["trace_identical"] == 0)
            n_spfp += (row["fp_movers_sp_exact"] > 0)
            n_dpfp += (row["fp_movers_dp_exact"] > 0)
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.perf_counter()-t0:.0f}s) | "
                      f"traces diverged {n_div} | float32 non-fixed-point {n_spfp} | "
                      f"float64 non-fixed-point {n_dpfp}", flush=True)
        if args.workers > 1:
            pool.close()
            pool.join()

    print(f"\n=== VERDICT over {done} runs ({time.perf_counter()-t0:.0f}s) ===")
    print(f"  runs where the decision sequences DIVERGED : {n_div}/{done}")
    print(f"  runs where FLOAT32 ended off a fixed point : {n_spfp}/{done}")
    print(f"  runs where FLOAT64 ended off a fixed point : {n_dpfp}/{done}")
    print("  (fixed point = zero would-movers under Hartigan's criterion, evaluated in "
          "float64 from the partition's exact means)")
    print(f"\nDONE — {args.csv}", flush=True)


if __name__ == "__main__":
    main()
