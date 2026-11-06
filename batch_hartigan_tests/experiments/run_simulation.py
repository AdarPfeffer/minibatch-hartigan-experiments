"""
Simulation harness for the base Batch-Hartigan convergence study (v2).

Sweeps a 5-parameter grid (amount_of_clusters x dim x intracluster_variance x
cluster_size x intercluster_variance), runs many trials per regime, and writes a
CSV + verbose console output using multiprocessing.

This module is import-friendly: the notebook (run_base_v2.ipynb) drives the smoke
test and the full run by calling `build_regimes`, `run_sweep`, and
`estimate_full_run` — the multiprocessing worker (`run_regime`) lives here at
module scope so it stays picklable under both 'fork' and 'spawn' start methods.

Notes vs an earlier, unpublished version of this study (not part of this repository):
  * Ground-truth data generation (generate_ground_truth_data) instead of the
    single-blob generator — labels are discarded here but the data now has real
    cluster structure.
  * intracluster_variance (was sigsq) and intercluster_variance (was
    center_spread) are each swept 1..10.
  * compare_batch_hart is vectorized (bit-equivalent to the reference loop).
  * Per regime we record: zero/loop/max-iter counts, the iteration histogram and
    its mean/max, the loop-size histogram and its mean/max, and the mean/max
    number of iterations before a detected loop begins (counter - loop_size).
  * On a real cycle the algorithm stops immediately (see batch_hartigan), so
    looping trials record their (small) detection iteration rather than max_iter.
"""

import csv
import functools
import multiprocessing
import os
import random
import time
from datetime import timedelta

import numpy as np

from batch_hartigan import batch_hartigan, create_rng, generate_ground_truth_data
# Legacy single-blob generator (no ground-truth labels) is no longer used by the
# big test; ground-truth generation is better even though labels are discarded.
# from batch_hartigan import generate_data  # <-- intentionally not imported

# ── Default parameter grid (v2) ──────────────────────────────────────────────
AMOUNT_OF_CLUSTERS_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DIM_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
INTRACLUSTER_VARIANCE_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CLUSTER_SIZE_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100,
                       150, 200, 250, 300]
INTERCLUSTER_VARIANCE_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

TRIALS_PER_REGIME = 1000
MAX_ITER = 500  # per-trial iteration budget (MAX_ITER + 1 histogram bins: 0..500)

CSV_FIELDS = [
    "amount_of_clusters", "dim", "intracluster_variance", "intercluster_variance",
    "cluster_size",
    "trials",          # trials actually completed (== trials_per_regime unless some raised)
    "zero_count", "loop_count", "max_iter_count",
    "mean_iterations", "max_iterations",
    "mean_loop_size", "max_loop_size",                 # over this regime's looping trials
    "mean_iters_before_loop", "max_iters_before_loop",  # transient length before the cycle
    "iterations_histogram", "loop_size_histogram",
]


def build_regimes(amount_of_clusters_values=AMOUNT_OF_CLUSTERS_VALUES,
                  dim_values=DIM_VALUES,
                  intracluster_variance_values=INTRACLUSTER_VARIANCE_VALUES,
                  cluster_size_values=CLUSTER_SIZE_VALUES,
                  intercluster_variance_values=INTERCLUSTER_VARIANCE_VALUES):
    """Build the list of regime tuples for a sweep.

    Each regime is (regime_index, amount_of_clusters, dim, intracluster_variance,
    cluster_size, intercluster_variance). Defaults give the full v2 grid; pass
    smaller lists for a smoke test.
    """
    regimes = []
    idx = 1
    for amount_of_clusters in amount_of_clusters_values:
        for dim in dim_values:
            for intracluster_variance in intracluster_variance_values:
                for cluster_size in cluster_size_values:
                    for intercluster_variance in intercluster_variance_values:
                        regimes.append((idx, amount_of_clusters, dim,
                                        intracluster_variance, cluster_size,
                                        intercluster_variance))
                        idx += 1
    return regimes


def run_regime(params, trials_per_regime=TRIALS_PER_REGIME, max_iter=MAX_ITER):
    """Run all trials for a single parameter regime.

    Args:
        params:            (regime_index, amount_of_clusters, dim,
                           intracluster_variance, cluster_size,
                           intercluster_variance)
        trials_per_regime: number of independent trials for this regime
        max_iter:          per-trial iteration budget

    Returns:
        dict with all params, counts, histograms, and timing info.
    """
    (regime_idx, amount_of_clusters, dim, intracluster_variance, cluster_size,
     intercluster_variance) = params

    zero_count = 0
    loop_count = 0
    max_iter_count = 0                              # trials that exhausted the budget
    iterations = np.zeros(max_iter + 1, dtype=int)  # bins 0..max_iter
    loop_size_counts = {}                           # loop_size -> #trials
    iters_before_loop_sum = 0                       # sum over looping trials
    iters_before_loop_max = 0

    start = time.monotonic()

    for _ in range(trials_per_regime):
        try:
            rng, _seed = create_rng()
            # Ground-truth generation; labels are discarded for the base study.
            centroids, data, init_centers, _labels = generate_ground_truth_data(
                rng, cluster_size, amount_of_clusters, dim,
                intracluster_variance, intercluster_variance)
            assigns, centroids, counter, loop, loop_size, zero = batch_hartigan(
                data, init_centers, centroids, amount_of_clusters, max_iter=max_iter
            )

            iterations[counter] += 1
            if zero:
                zero_count += 1
            elif loop:
                loop_count += 1
                loop_size_counts[loop_size] = loop_size_counts.get(loop_size, 0) + 1
                ibl = counter - loop_size           # iterations before the loop begins
                iters_before_loop_sum += ibl
                iters_before_loop_max = max(iters_before_loop_max, ibl)
            elif counter == max_iter:
                max_iter_count += 1                 # budget exhausted, no convergence/loop

        except Exception as exc:
            # One bad trial must not kill the entire regime
            print(f"  [WARNING] Regime {regime_idx}: trial failed — {exc}")

    elapsed = time.monotonic() - start

    # Iteration histogram string: "1:499,2:368,..." (non-zero bins only)
    hist_str = ",".join(f"{i}:{c}" for i, c in enumerate(iterations) if c > 0)

    # Loop-size histogram string: only real loops (size > 0), e.g. "2:14,3:5"
    loop_hist_str = ",".join(
        f"{s}:{c}" for s, c in sorted(loop_size_counts.items()) if s > 0)

    total_iters = sum(i * int(c) for i, c in enumerate(iterations))
    total_trials_done = int(iterations.sum())
    mean_iters = total_iters / total_trials_done if total_trials_done > 0 else 0.0
    max_iters = max((i for i, c in enumerate(iterations) if c > 0), default=0)

    # Per-regime loop-size aggregates (over this regime's looping trials).
    loop_size_sum = sum(s * c for s, c in loop_size_counts.items())
    loop_size_max = max(loop_size_counts.keys(), default=0)
    mean_loop_size = loop_size_sum / loop_count if loop_count > 0 else 0.0
    mean_iters_before_loop = (iters_before_loop_sum / loop_count
                              if loop_count > 0 else 0.0)

    return {
        "regime_idx": regime_idx,
        "amount_of_clusters": amount_of_clusters,
        "dim": dim,
        "intracluster_variance": intracluster_variance,
        "cluster_size": cluster_size,
        "intercluster_variance": intercluster_variance,
        "trials": total_trials_done,   # trials that actually completed (no exception)
        "zero_count": zero_count,
        "loop_count": loop_count,
        "max_iter_count": max_iter_count,
        "mean_iterations": mean_iters,
        "max_iterations": max_iters,
        "mean_loop_size": mean_loop_size,                 # per-regime, 0.0 if no loops
        "max_loop_size": loop_size_max,                   # per-regime, 0 if no loops
        "mean_iters_before_loop": mean_iters_before_loop,  # per-regime, 0.0 if no loops
        "max_iters_before_loop": iters_before_loop_max,    # per-regime, 0 if no loops
        "iterations_histogram": hist_str,
        "loop_size_histogram": loop_hist_str,
        # carried to the grand summary (not CSV columns):
        "iterations_array": iterations,
        "loop_size_sum": loop_size_sum,
        "loop_size_max": loop_size_max,
        "iters_before_loop_sum": iters_before_loop_sum,
        "iters_before_loop_max": iters_before_loop_max,
        "elapsed": elapsed,
    }


def run_sweep(regimes=None, *, trials_per_regime=TRIALS_PER_REGIME,
              max_iter=MAX_ITER, csv_path="results_v2.csv", n_workers=None,
              verbose=True):
    """Run a sweep over `regimes` in parallel and write results to `csv_path`.

    Returns a dict of grand-total statistics (also printed when verbose).
    """
    if regimes is None:
        regimes = build_regimes()
    total_regimes = len(regimes)
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Grand-total accumulators
    grand_zero = 0
    grand_loop = 0
    grand_max_iter = 0       # trials that exhausted the budget
    grand_trials = 0         # trials that actually completed
    grand_requested = 0      # trials we asked for (completed + failed)
    grand_iterations = np.zeros(max_iter + 1, dtype=int)
    grand_loop_size_sum = 0
    grand_loop_size_max = 0
    grand_ibl_sum = 0        # iterations-before-loop, summed over looping trials
    grand_ibl_max = 0
    completed = 0
    overall_start = time.monotonic()

    # functools.partial of a module-level function stays picklable under 'spawn'.
    worker = functools.partial(
        run_regime, trials_per_regime=trials_per_regime, max_iter=max_iter)

    if verbose:
        print(f"Total regimes: {total_regimes}")
        print(f"Trials per regime: {trials_per_regime}")
        print(f"Total trials: {total_regimes * trials_per_regime:,}")
        print(f"Max iterations per trial: {max_iter}")
        print(f"Workers: {n_workers}")
        print(f"Output CSV: {os.path.abspath(csv_path)}")
        print()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with multiprocessing.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(worker, regimes):
                completed += 1

                grand_zero += result["zero_count"]
                grand_loop += result["loop_count"]
                grand_max_iter += result["max_iter_count"]
                grand_trials += result["trials"]          # only completed trials
                grand_requested += trials_per_regime
                grand_iterations += result["iterations_array"]
                grand_loop_size_sum += result["loop_size_sum"]
                grand_loop_size_max = max(grand_loop_size_max, result["loop_size_max"])
                grand_ibl_sum += result["iters_before_loop_sum"]
                grand_ibl_max = max(grand_ibl_max, result["iters_before_loop_max"])

                writer.writerow({k: result[k] for k in CSV_FIELDS})
                f.flush()

                if verbose:
                    elapsed_total = time.monotonic() - overall_start
                    avg_per_regime = elapsed_total / completed
                    remaining = avg_per_regime * (total_regimes - completed)

                    data_size = result["cluster_size"] * result["amount_of_clusters"]
                    done = result["trials"]
                    denom = done if done > 0 else 1
                    zp = result["zero_count"] / denom * 100
                    lp = result["loop_count"] / denom * 100

                    print(f"--- Regime [{completed}/{total_regimes}] ---")
                    print(f"  Params: amount_of_clusters={result['amount_of_clusters']}, "
                          f"dim={result['dim']}, "
                          f"intracluster_variance={result['intracluster_variance']}, "
                          f"cluster_size={result['cluster_size']}, "
                          f"intercluster_variance={result['intercluster_variance']} "
                          f"(data_size={data_size})")
                    if done < trials_per_regime:
                        print(f"  [!] only {done}/{trials_per_regime} trials completed "
                              f"({trials_per_regime - done} raised exceptions)")
                    print(f"  Zero clusters: {result['zero_count']}/{done} ({zp:.1f}%)")
                    print(f"  Infinite loops: {result['loop_count']}/{done} ({lp:.1f}%)")
                    if result["max_iter_count"]:
                        print(f"  [!] Hit max_iter ({max_iter}) without converging: "
                              f"{result['max_iter_count']}/{done}")
                    print(f"  Iteration histogram: {result['iterations_histogram']}")
                    if result["loop_count"]:
                        print(f"  Loop size: mean {result['mean_loop_size']:.2f}, "
                              f"max {result['max_loop_size']}  "
                              f"(histogram: {result['loop_size_histogram']})")
                        print(f"  Iters before loop: mean "
                              f"{result['mean_iters_before_loop']:.2f}, "
                              f"max {result['max_iters_before_loop']}")
                    print(f"  Mean iterations: {result['mean_iterations']:.1f} | "
                          f"Max iterations: {result['max_iterations']}")
                    print(f"  Elapsed: {timedelta(seconds=int(elapsed_total))} | "
                          f"Est. remaining: ~{timedelta(seconds=int(remaining))}")
                    print()

    total_time = time.monotonic() - overall_start
    grand_total_iters = sum(i * int(c) for i, c in enumerate(grand_iterations))
    grand_mean = grand_total_iters / grand_trials if grand_trials > 0 else 0.0
    grand_hist_str = ",".join(
        f"{i}:{c}" for i, c in enumerate(grand_iterations) if c > 0)
    gzp = grand_zero / grand_trials * 100 if grand_trials > 0 else 0.0
    glp = grand_loop / grand_trials * 100 if grand_trials > 0 else 0.0
    gmp = grand_max_iter / grand_trials * 100 if grand_trials > 0 else 0.0
    grand_mean_loop_size = grand_loop_size_sum / grand_loop if grand_loop > 0 else 0.0
    grand_mean_ibl = grand_ibl_sum / grand_loop if grand_loop > 0 else 0.0

    grand_failed = grand_requested - grand_trials

    summary = {
        "total_regimes": total_regimes,
        "total_trials": grand_trials,          # completed
        "total_requested": grand_requested,
        "total_failed": grand_failed,
        "total_zero": grand_zero,
        "total_loop": grand_loop,
        "total_max_iter": grand_max_iter,
        "zero_pct": gzp,
        "loop_pct": glp,
        "max_iter_pct": gmp,
        "mean_iterations": grand_mean,
        "mean_loop_size": grand_mean_loop_size,
        "max_loop_size": grand_loop_size_max,
        "mean_iters_before_loop": grand_mean_ibl,
        "max_iters_before_loop": grand_ibl_max,
        "iterations_histogram": grand_hist_str,
        "total_time_seconds": total_time,
        "csv_path": os.path.abspath(csv_path),
    }

    if verbose:
        print("=" * 48)
        print("                 GRAND TOTALS")
        print("=" * 48)
        print(f"Total regimes: {total_regimes}")
        print(f"Total trials completed: {grand_trials:,} / {grand_requested:,} requested")
        if grand_failed:
            print(f"  [!] {grand_failed:,} trials raised exceptions and were excluded")
        print(f"Total zero clusters: {grand_zero:,} ({gzp:.1f}%)")
        print(f"Total infinite loops: {grand_loop:,} ({glp:.1f}%)")
        print(f"Total hit max_iter ({max_iter}): {grand_max_iter:,} ({gmp:.2f}%)"
              + ("   <-- raise MAX_ITER if non-zero" if grand_max_iter else ""))
        print(f"Overall iteration histogram: {grand_hist_str}")
        print(f"Overall mean iterations: {grand_mean:.1f}")
        print(f"Mean loop size: {grand_mean_loop_size:.2f}  (over looping trials only)")
        print(f"Max loop size: {grand_loop_size_max}")
        print(f"Mean iters before loop: {grand_mean_ibl:.2f}  (over looping trials only)")
        print(f"Max iters before loop: {grand_ibl_max}")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Results saved to: {os.path.abspath(csv_path)}")
        print("=" * 48)

    return summary


def estimate_full_run(sample_size=40, trials_per_sample=20, seed=0,
                      regimes=None, full_trials_per_regime=TRIALS_PER_REGIME,
                      max_iter=MAX_ITER, n_workers=None):
    """Estimate wall-clock for the full sweep by timing a uniform random sample of
    regimes in-process and extrapolating.

    Because regimes are sampled uniformly from the grid, the mean per-trial time
    is an unbiased estimate of the grid-average per-trial cost. The parallel
    estimate assumes near-linear scaling across workers (optimistic).

    Returns a dict of estimates (does not write any CSV).
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    if regimes is None:
        regimes = build_regimes()
    total_regimes = len(regimes)

    rng = random.Random(seed)
    sample = rng.sample(regimes, min(sample_size, total_regimes))

    per_trial_times = []
    worst_per_trial = 0.0
    worst_params = None
    sample_start = time.monotonic()
    for params in sample:
        r = run_regime(params, trials_per_regime=trials_per_sample, max_iter=max_iter)
        per_trial = r["elapsed"] / trials_per_sample
        per_trial_times.append(per_trial)
        if per_trial > worst_per_trial:
            worst_per_trial = per_trial
            worst_params = params
    sample_wall = time.monotonic() - sample_start

    mean_per_trial = sum(per_trial_times) / len(per_trial_times)
    total_trials = total_regimes * full_trials_per_regime
    serial_seconds = mean_per_trial * total_trials
    parallel_seconds = serial_seconds / n_workers

    return {
        "sample_size": len(sample),
        "trials_per_sample": trials_per_sample,
        "mean_per_trial_s": mean_per_trial,
        "worst_per_trial_s": worst_per_trial,
        "worst_params": worst_params,
        "total_regimes": total_regimes,
        "full_trials_per_regime": full_trials_per_regime,
        "total_trials": total_trials,
        "n_workers": n_workers,
        "est_serial_seconds": serial_seconds,
        "est_parallel_seconds": parallel_seconds,
        "sample_wall_seconds": sample_wall,
    }


def main():
    """Run the full v2 sweep -> ../results/data/results_v2.csv.

    The shipped copy of that file is gzipped (results_v2.csv.gz); this writes the
    uncompressed .csv beside it and never touches the .gz.
    """
    run_sweep(csv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "results", "data", "results_v2.csv"))


if __name__ == "__main__":
    main()
