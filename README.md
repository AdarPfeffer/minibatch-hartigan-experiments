# Batch and Mini-Batch Hartigan k-means: code and data

Code, raw results and figure/table scripts for the paper's numerical experiments:

* **Batch Hartigan failure modes** (paper Section 5.1) — how often Hartigan's criterion applied to
  the whole dataset at once ends in an empty cluster or an infinite loop.
* **Mini-Batch Hartigan runtime and quality** (Section 5.2 and the Appendix) — certified
  Mini-Batch Hartigan on CPU (Numba) and GPU (JAX) against Online Hartigan (Numba) and the
  original Fortran 77 implementation (AS 136), and Lloyd's algorithm for quality.
* **Precision** (Section 3.3.3) — AS 136 in float64 vs float32.

Every result file the paper draws on is shipped, gzipped, next to the script that plots or
tabulates it, and the Slurm job that produced it is in the matching `experiments/jobs/`.

## Layout

```
batch_hartigan_tests/
  experiments/            batch_hartigan.py, run_simulation.py, run_base_v2.ipynb      (low d, NumPy)
                          batch_hartigan_jax.py, run_simulation_jax.py, merge_jax_shards.py,
                          audit_jax_vs_numpy.py                                          (high d, JAX)
    jobs/                 run_simulation_jax.job
  results/
    data/                 results_v2.csv.gz (low d), results_jax.csv.gz (high d)
    figures/              main_{zero,loop}_rate.svg, failure_{zero,loop}_rate_jax.png
    main_failure_plots.ipynb, make_failure_plots_jax.py

minibatch_hartigan_tests/
  experiments/            bestof_init_benchmark.py   the runtime/quality benchmark driver
                          mini_batch.py, streamed_hartigan.py          Numba Mini-Batch / Online
                          jax_minibatch_hartigan.py, _lloyd.py,
                          _common_functions_for_minibatch.py           JAX Mini-Batch / Lloyd
                          init_schemes.py, merge_bestof_csv.py
    fortran/              AS 136 + ctypes wrappers + the float64/float32 study (see its README)
    tests/                correctness tests and the pre-flight gates the jobs run
    jobs/                 the Slurm jobs behind every shipped data file
  results/
    data/random_centroid/     bestof_cpu_full, bestof_jaxtime_rc_29208159, bestof_lloyd_23611965
    data/random_partition/    bestof_rpart_cpu_26508322, bestof_jaxtime_rp_29208159,
                              bestof_rpart_lloyd_29098778                       (all .csv.gz)
    runtime/              the four runtime figures
    quality/              the three random-centroid quality tables (.tex)
    random_partition/     the random-partition runtime figure and quality table
    precision/            float64 vs float32 AS 136 data + control log (see its README)
    analysis.ipynb, make_quality_tables_tex.py
```

## Paper → files

| Paper | Produced by | File |
|---|---|---|
| Fig. `Figure 1: Empty Clusters` | `batch_hartigan_tests/results/main_failure_plots.ipynb` | `batch_hartigan_tests/results/figures/main_zero_rate.svg` |
| Fig. `fig:empty-cluster-high-d` | `batch_hartigan_tests/results/make_failure_plots_jax.py` | `batch_hartigan_tests/results/figures/failure_zero_rate_jax.png` |
| Fig. `fig:infinite-loops-low-d` | `main_failure_plots.ipynb` | `batch_hartigan_tests/results/figures/main_loop_rate.svg` |
| Fig. `fig:infinite-loops-high-d` | `make_failure_plots_jax.py` | `batch_hartigan_tests/results/figures/failure_loop_rate_jax.png` |
| Fig. `fig:runtime-to-convergence` | `minibatch_hartigan_tests/results/analysis.ipynb` | `minibatch_hartigan_tests/results/runtime/runtime_to_convergence_f64.png` |
| Fig. `fig:runtime-single-iteration` | `analysis.ipynb` | `minibatch_hartigan_tests/results/runtime/runtime_per_iteration_f64.png` |
| Tab. `tab:quality_mean_of_50` | `minibatch_hartigan_tests/results/make_quality_tables_tex.py` | `minibatch_hartigan_tests/results/quality/quality_mean_of_50.tex` |
| Tab. `tab:quality_bestof50` | `make_quality_tables_tex.py` | `minibatch_hartigan_tests/results/quality/quality_bestof50.tex` |
| Fig. `fig:iteration-counts` (Appendix) | `analysis.ipynb` | `minibatch_hartigan_tests/results/runtime/iterations_to_convergence_f64.png` |
| Fig. `fig:heatmap-runtime` (Appendix) | `analysis.ipynb` | `minibatch_hartigan_tests/results/runtime/jax_speedup_heatmap_vs_as136.png` |
| Tab. `tab:quality_by_batch_fraction_mean` (Appendix) | `make_quality_tables_tex.py` | `minibatch_hartigan_tests/results/quality/quality_by_batch_fraction_mean.tex` |
| Fig. `fig:random-partition-runtime` (Appendix) | `analysis.ipynb` | `minibatch_hartigan_tests/results/random_partition/runtime_to_convergence_random_partition.png` |
| Tab. `tab:quality_mean_of_50_rp` (Appendix) | `make_quality_tables_tex.py` | `minibatch_hartigan_tests/results/random_partition/quality_mean_of_50_rp.tex` |
| Section 3.3.3 (Precision) | `minibatch_hartigan_tests/experiments/fortran/fp_precision_compare.py`, `fp_untouched_check.py` | `minibatch_hartigan_tests/results/precision/` |

The two low-dimensional failure figures are shipped as SVG (the format the notebook writes); the
paper uses PNG renderings of them.

## Data

| File | Rows | Contents | Job → hardware |
|---|---|---|---|
| `batch_hartigan_tests/results/data/results_v2.csv.gz` | 372,400 | one row per regime (K × d × σ² × b × τ² = 14 × 14 × 10 × 19 × 10), 1,000 datasets each | `batch_hartigan_tests/experiments/run_simulation.py` (no Slurm job; launched from `run_base_v2.ipynb`) |
| `batch_hartigan_tests/results/data/results_jax.csv.gz` | 5,320 | one row per regime (14 K × 5 d × 4 (σ², τ²) × 19 b), 1,000 datasets each | `run_simulation_jax.job` (4 shards, merged with `merge_jax_shards.py`) |
| `.../random_centroid/bestof_cpu_full.csv.gz` | 104,400 | `numba_online_ls`, `fortran_ls`, `fortran_nols` (13,500 each), `numba_mb_cert` (63,900) | `run_bestof_cpu.job` + `run_bestof_cpu_finish.job` (merged) → AMD EPYC |
| `.../random_centroid/bestof_jaxtime_rc_29208159.csv.gz` | 50,400 | `jax_mb_cert`, float64, grid batch fractions | `run_bestof_jaxtime_h200.job` task 0 → NVIDIA H200 |
| `.../random_centroid/bestof_lloyd_23611965.csv.gz` | 13,500 | `jax_lloyd`, float64 (quality only) | `run_bestof_lloyd.job` → NVIDIA RTX 5000 Ada |
| `.../random_partition/bestof_rpart_cpu_26508322.csv.gz` | 63,900 | `numba_online_ls` (13,500), `numba_mb_cert` (50,400) | `run_bestof_rpart_cpu.job` → AMD EPYC |
| `.../random_partition/bestof_jaxtime_rp_29208159.csv.gz` | 50,400 | `jax_mb_cert`, float64 | `run_bestof_jaxtime_h200.job` task 1 → NVIDIA H200 |
| `.../random_partition/bestof_rpart_lloyd_29098778.csv.gz` | 13,500 | `jax_lloyd`, float64 (quality only) | `run_bestof_rpart_lloyd.job` → NVIDIA RTX 5000 Ada |

Every shipped `.csv.gz` decompresses to exactly the file the analysis was run on. The four JAX
files (`bestof_jaxtime_*`, `bestof_*lloyd_*`) are the job's part files concatenated under one header,
so their rows are byte-for-byte what the jobs wrote; `bestof_cpu_full` was merged with
`merge_bestof_csv.py` and `results_jax` with `merge_jax_shards.py` (both through pandas, so e.g. a
proportion of `-1` reads `-1.0` there). Each Mini-Batch Hartigan row is one dataset at one batch
fraction (`proportion`; −1 for algorithms without one) and summarizes its 50 starting partitions:
`nmi_*`, `sse_*`, `nmi_at_min_sse` (quality), `conv_*_s` (time to convergence, including
initialization), `singleit_*_s` (post-initialization time per iteration or epoch), `n_iters_*`,
`certified_frac`, `empty_frac`, `cert_budget`, `capped_frac`. pandas reads the `.csv.gz` files
directly.

## Reproducing the figures and tables from the shipped data

```bash
pip install -r requirements.txt

cd batch_hartigan_tests/results
jupyter nbconvert --to notebook --execute --inplace main_failure_plots.ipynb   # figures/main_*.svg
python make_failure_plots_jax.py                                               # figures/failure_*_jax.png
cd ../../minibatch_hartigan_tests/results
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb             # runtime/, random_partition/
python make_quality_tables_tex.py --check                                      # quality/, random_partition/
```

These overwrite the shipped figures and tables in place. With the package versions in
`requirements.txt`, the PNGs and `.tex` tables come out byte-identical.

## Re-running the experiments

All Mini-Batch, Lloyd and high-dimensional Batch Hartigan data were produced on a Slurm cluster.
The jobs in `*/experiments/jobs/` are submitted from the repository root
(`sbatch minibatch_hartigan_tests/experiments/jobs/<job>.job`). They assume the repository is at
`$HOME/kmeans-paper-repo` (override with `REPO=...`), a conda environment named
`kmeans-jax-gpu`, and the cluster's own partitions and modules — edit those lines for another
cluster. The Mini-Batch and precision jobs write fresh outputs next to the shipped ones as plain
`.csv` parts, and `--resume` only looks at those parts, never at the shipped `.csv.gz` files.
`run_simulation_jax.job` writes its shard CSVs into `batch_hartigan_tests/experiments/`; the merge
command in its header writes `batch_hartigan_tests/results/data/results_jax.csv`.

Local checks (CPU only, minutes):

```bash
cd batch_hartigan_tests/experiments && python audit_jax_vs_numpy.py && cd ../..   # NumPy vs JAX parity gate
python minibatch_hartigan_tests/experiments/tests/test_live_sets.py
python minibatch_hartigan_tests/experiments/tests/test_jax_hartigan.py
python minibatch_hartigan_tests/experiments/tests/test_fp64_precision_isolation.py
JAX_ENABLE_X64=1 python minibatch_hartigan_tests/experiments/tests/gate_jax_compile_in_clock.py random_centroid --allow-cpu
KMEANS_SMOKE=1 python minibatch_hartigan_tests/experiments/bestof_init_benchmark.py \
    --backend-group cpu --impls numba_online_ls,numba_mb_cert --csv smoke.csv
```

`run_base_v2.ipynb` runs the low-dimensional study's unit tests, a smoke sweep, and (behind a flag)
the full sweep.

**Determinism.**
* Mini-Batch Hartigan study: datasets, starting partitions and batch draws are seeded per
  `(d, N, K, dataset, start, batch fraction)`, so any cell can be re-run on its own.
* JAX: the shipped rows come from a JAX version in which `jax_threefry_partitionable` defaults to
  `True`. On older JAX (e.g. 0.4.30) set `JAX_THREEFRY_PARTITIONABLE=1` to get the same random
  stream. With it, re-running regimes of the high-dimensional Batch Hartigan sweep on CPU
  reproduces their `results_jax.csv` rows exactly.
* The low-dimensional Batch Hartigan sweep draws its per-trial seeds from NumPy's unseeded
  global generator, so a re-run reproduces the grid and the rates up to Monte-Carlo noise, not
  the exact counts.

**Fortran.** The `fortran_ls` / `fortran_nols` implementations and the precision study need
`gfortran` (`make lib`, `make lib-fp`, `make lib-ref` in `minibatch_hartigan_tests/experiments/fortran/`).
