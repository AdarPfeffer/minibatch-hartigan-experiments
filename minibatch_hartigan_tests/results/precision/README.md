# Precision evidence: AS 136 in float64 vs float32 (paper Section 3.3.3, "Precision")

The paper states that Hartigan's Fortran 77 implementation (AS 136) can fail to terminate in
single precision but not in double precision, and that all implementations are therefore run in
float64. These two files back that statement.

| File | What it is | Produced by |
|---|---|---|
| `fp_precision_21724342.csv.gz` | 13,500 paired float64/float32 AS 136 runs, 37 columns | `experiments/jobs/run_fp_precision.job` (Slurm job 21724342) → `experiments/fortran/fp_precision_compare.py` |
| `fp-untouched-21912818.out` | log of the untouched-source control | `experiments/jobs/run_fp_untouched.job` (Slurm job 21912818) → `experiments/fortran/fp_untouched_check.py` |

## The sweep (`fp_precision_21724342.csv.gz`)

Same grid, datasets and starting centroids as the runtime study (`bestof_init_benchmark.py`'s
`dataset_seed` / `init_seed`, `init_id` 0, random centroid): $d \in \{100, 1000, 10000\}$ × 15 values
of $N$ × $K \in \{4, 8, 16\}$ × 100 trials. Each trial runs AS 136 twice, once compiled in double
precision and once in single precision (`fp_as136_dp.f` / `fp_as136_sp.f`, both generated from
`as136.f` by `gen_fp_variants.py`, and checked to differ only in floating-point width by
`fp_audit.py`). Live-set guard off (`uselive=0`), pass budget 10,000. Every accepted transfer is
recorded, so the two runs are compared decision by decision.

What the file shows (all recomputable with pandas):

* **float64:** all 13,500 runs terminate (`ifault_dp == 0`), in at most 15 passes, at a true fixed
  point (`fp_movers_dp_exact == 0` in every row).
* **float32:** 10 runs never terminate: `ifault_sp == 2` after the full 10,000 passes. All 10 are at
  $d = 10{,}000$. Their float64 twins terminate in 2–4 passes. They cost 605× to 2,100× the float64
  run (median 1,498×). The other 13,490 float32 runs terminate in at most 15 passes.
* **Divergence:** the float32 decision sequence departs from float64 in 1,459 of 13,500 runs (10.8%;
  3.4% / 6.0% / 23.1% at $d$ = 100 / 1,000 / 10,000), and the final labels differ in 1,060.
* No trace overflowed in either precision.

## The untouched-source control (`fp-untouched-21912818.out`)

Removes the generator and the instrumentation from the evidence chain. It compiles the
**unmodified** `as136.f` (md5 `65128ab861af399bfe20e91b78ae2839`, recorded in the log and equal to
the file shipped here) twice — normally, and with `-freal-8-real-4`, which demotes every
`double precision` to 4-byte `real` at compile time — and replays the 10 non-terminating seeds plus
3 controls. With the live-set guard both off and on (`uselive=0` and `uselive=1`), all 10 again run
the full 10,000 passes (`ifault=2`) in float32 and terminate in 2–4 passes in float64, and the 3
controls terminate in both.

In this copy of the log, the compute node's hostname (line 2) and the user and group in two
`ls -l` lines (lines 10–11) are replaced by placeholders. Nothing else is changed.

## Scope

These files record pass counts, termination codes, trace lengths and the first divergence of each
run, not the full transfer sequences. They establish non-termination in float32 and termination at
fixed points in float64. They do not record the periodic structure of the non-terminating runs.

`as136.f` is John Burkardt's FORTRAN77 translation of AS 136 (Hartigan & Wong, 1979), with local
hooks for timing and for switching the live-set guard; see `experiments/fortran/README.md`.
