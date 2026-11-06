"""
run_simulation_jax.py — sweep harness for the high-dimensional Batch-Hartigan failure-mode
study (JAX/GPU port). Restructures the numpy v2 grid:

  * SNR: the redundant 10x10 (intra x inter) variance grid collapses to the 4 anti-diagonal
    points {(1,10),(4,7),(7,4),(10,1)} (fixed total variance 11, swept separability ratio).
  * dim: {100, 500, 1000, 5000, 10000} replaces [2..15].
  * K [2..15], cluster_size (19 values), trials=1000, max_iter=500 are unchanged.

Grid = 14 x 5 x 4 x 19 = 5,320 regimes x 1000 = 5.32M trials. The CSV schema is byte-for-byte
the numpy study's (`CSV_FIELDS`), with both variance columns filled from each pair, so the
existing plotting code reads it unchanged.

Single process, sequential over regimes (one GPU can't be forked). Within a regime the 1000
trials run in GPU batches of B_safe(n,d) (memory-capped) via the compacted batched driver.
`--shard k --num-shards N` runs the disjoint regimes with regime_index % N == k (round-robin
so the expensive d=10000 regimes spread evenly across shards); deterministic per-coordinate
seeding makes the sharded result byte-identical to a single run.

    python run_simulation_jax.py [--shard k --num-shards N] [--csv PATH]
                                 [--dims ...] [--trials 1000] [--max-iter 500]
"""
import argparse
import csv
import math
import os
import time

import numpy as np

import batch_hartigan_jax as bhj

# ── Restructured grid ────────────────────────────────────────────────────────
AMOUNT_OF_CLUSTERS_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DIM_VALUES = [100, 500, 1000, 5000, 10000]
SNR_PAIRS = [(1, 10), (4, 7), (7, 4), (10, 1)]          # (intracluster_variance, intercluster_variance)
CLUSTER_SIZE_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100,
                       150, 200, 250, 300]

TRIALS_PER_REGIME = 1000
MAX_ITER = 500

# CSV schema — identical to run_simulation.py:53-62 in this directory (comparability).
CSV_FIELDS = [
    "amount_of_clusters", "dim", "intracluster_variance", "intercluster_variance",
    "cluster_size",
    "trials",
    "zero_count", "loop_count", "max_iter_count",
    "mean_iterations", "max_iterations",
    "mean_loop_size", "max_loop_size",
    "mean_iters_before_loop", "max_iters_before_loop",
    "iterations_histogram", "loop_size_histogram",
]

# Per-batch device-memory budget for the (B,n,d) float64 DATA array (bytes). Kept well below
# the ~22 GB JAX preallocates on the 32 GB Ada because the PEAK is ~2-3x the data array: the
# Triton float64 GEMM autotuner allocates a scratch buffer ~= the operands, and compaction's
# gather transiently holds a second copy. Budgeting the data at ~4 GB keeps peak ~12 GB with
# margin (a 12 GB data budget OOM'd: 11.9 GB data + 11.1 GB GEMM scratch > 22 GB).
DATA_BUDGET_BYTES = 4e9


def b_safe(n, dim, cap=1000, budget=DATA_BUDGET_BYTES):
    """Largest batch B whose (B,n,dim) float64 data array fits the budget (peak ~2-3x this)."""
    return max(1, min(cap, int(budget // (8 * n * dim))))


def build_regimes_jax():
    """Regime tuples (regime_index, K, dim, intra, inter, cluster_size). Loop order
    dim -> K -> cluster_size -> SNR so the 4 SNR regimes share one (n, dim, K) shape and reuse
    the same compiled pass. regime_index is 0-based and stable (drives sharding)."""
    regimes = []
    idx = 0
    for dim in DIM_VALUES:
        for K in AMOUNT_OF_CLUSTERS_VALUES:
            for cluster_size in CLUSTER_SIZE_VALUES:
                for (intra, inter) in SNR_PAIRS:
                    regimes.append((idx, K, dim, intra, inter, cluster_size))
                    idx += 1
    return regimes


def run_regime_jax(params, trials_per_regime=TRIALS_PER_REGIME, max_iter=MAX_ITER):
    """Run all trials of one regime and aggregate exactly as numpy run_simulation.run_regime
    (:106-190): the iterations histogram, zero/loop/max_iter counts, loop-size histogram, and
    the iters-before-loop transient."""
    regime_idx, K, dim, intra, inter, cluster_size = params
    n = cluster_size * K
    B = b_safe(n, dim)

    zero_count = 0
    loop_count = 0
    max_iter_count = 0
    iterations = np.zeros(max_iter + 1, dtype=int)      # bins 0..max_iter
    loop_size_counts = {}
    iters_before_loop_sum = 0
    iters_before_loop_max = 0

    start = time.monotonic()
    for chunk_start in range(0, trials_per_regime, B):
        tids = list(range(chunk_start, min(chunk_start + B, trials_per_regime)))
        data_b, init_b = bhj.generate_batch(dim, K, intra, inter, cluster_size, tids)
        for (_assigns, _cent, counter, loop, loop_size, zero) in \
                bhj.batch_hartigan_jax_batched(data_b, init_b, K, max_iter=max_iter):
            iterations[counter] += 1
            if zero:
                zero_count += 1
            elif loop:
                loop_count += 1
                loop_size_counts[loop_size] = loop_size_counts.get(loop_size, 0) + 1
                ibl = counter - loop_size
                iters_before_loop_sum += ibl
                iters_before_loop_max = max(iters_before_loop_max, ibl)
            elif counter == max_iter:
                max_iter_count += 1
    elapsed = time.monotonic() - start

    hist_str = ",".join(f"{i}:{c}" for i, c in enumerate(iterations) if c > 0)
    loop_hist_str = ",".join(
        f"{s}:{c}" for s, c in sorted(loop_size_counts.items()) if s > 0)
    total_iters = sum(i * int(c) for i, c in enumerate(iterations))
    total_trials_done = int(iterations.sum())
    mean_iters = total_iters / total_trials_done if total_trials_done > 0 else 0.0
    max_iters = max((i for i, c in enumerate(iterations) if c > 0), default=0)
    loop_size_sum = sum(s * c for s, c in loop_size_counts.items())
    loop_size_max = max(loop_size_counts.keys(), default=0)
    mean_loop_size = loop_size_sum / loop_count if loop_count > 0 else 0.0
    mean_iters_before_loop = (iters_before_loop_sum / loop_count
                              if loop_count > 0 else 0.0)

    return {
        "amount_of_clusters": K,
        "dim": dim,
        "intracluster_variance": intra,
        "intercluster_variance": inter,
        "cluster_size": cluster_size,
        "trials": total_trials_done,
        "zero_count": zero_count,
        "loop_count": loop_count,
        "max_iter_count": max_iter_count,
        "mean_iterations": mean_iters,
        "max_iterations": max_iters,
        "mean_loop_size": mean_loop_size,
        "max_loop_size": loop_size_max,
        "mean_iters_before_loop": mean_iters_before_loop,
        "max_iters_before_loop": iters_before_loop_max,
        "iterations_histogram": hist_str,
        "loop_size_histogram": loop_hist_str,
        "_regime_idx": regime_idx,
        "_batch": B,
        "_elapsed": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0, help="this shard's index k (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards N (regimes % N == k)")
    ap.add_argument("--csv", default=None, help="output CSV (default results_jax[_shard{k}].csv)")
    ap.add_argument("--dims", default=None, help="comma-list to override DIM_VALUES (smoke)")
    ap.add_argument("--clusters", default=None, help="comma-list to override K values (smoke)")
    ap.add_argument("--cluster-sizes", default=None, help="comma-list to override cluster sizes (smoke)")
    ap.add_argument("--trials", type=int, default=TRIALS_PER_REGIME)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    args = ap.parse_args()

    # optional grid overrides (smoke / calibration only).
    if args.dims:
        DIM_VALUES[:] = [int(x) for x in args.dims.split(",")]
    if args.clusters:
        AMOUNT_OF_CLUSTERS_VALUES[:] = [int(x) for x in args.clusters.split(",")]
    if args.cluster_sizes:
        CLUSTER_SIZE_VALUES[:] = [int(x) for x in args.cluster_sizes.split(",")]

    N = max(1, args.num_shards)
    k = args.shard % N
    csv_path = args.csv or (f"results_jax_shard{k}.csv" if N > 1 else "results_jax.csv")

    regimes = build_regimes_jax()
    mine = [r for r in regimes if r[0] % N == k]
    print(f"run_simulation_jax | shard {k}/{N} | {len(mine)}/{len(regimes)} regimes | "
          f"trials={args.trials} max_iter={args.max_iter} | -> {csv_path}", flush=True)

    t0 = time.monotonic()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, params in enumerate(mine):
            row = run_regime_jax(params, trials_per_regime=args.trials, max_iter=args.max_iter)
            writer.writerow(row)
            f.flush()
            if len(mine) <= 60 or i % 25 == 0 or i == len(mine) - 1:   # every regime on small/calib runs
                done = i + 1
                rate = done / max(time.monotonic() - t0, 1e-9)
                eta = (len(mine) - done) / max(rate, 1e-9)
                print(f"  [{done}/{len(mine)}] regime_idx={params[0]} "
                      f"K={params[1]} d={params[2]} snr=({params[3]},{params[4]}) "
                      f"cs={params[5]} B={row['_batch']} "
                      f"zero={row['zero_count']} loop={row['loop_count']} "
                      f"maxit={row['max_iter_count']} {row['_elapsed']:.2f}s "
                      f"| ETA {eta/60:.1f} min", flush=True)
    print(f"DONE shard {k}/{N}: {len(mine)} regimes in {(time.monotonic()-t0)/60:.1f} min "
          f"-> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
