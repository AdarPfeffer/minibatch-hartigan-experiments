# Fortran AS 136 (Hartigan–Wong): the CPU baseline, and the float32 control

`as136.f` is Algorithm AS 136 (Hartigan & Wong 1979, *Applied Statistics* 28(1):100–108): online
Hartigan with optimal-transfer and quick-transfer passes and live sets. The file is John
Burkardt's 2008 FORTRAN77 translation, with local hooks added for timing and for switching the
live-set guard on and off (common blocks `/hgn_out/` and `/hgn_cfg/`, set and read through
`hartigan_fortran_ext.f`). These two files are shipped byte-identical to the ones that produced
the data.

## Build (needs `gfortran`)

```bash
make lib        # libas136.so            used by ../bestof_init_benchmark.py (impls fortran_ls, fortran_nols)
make lib-fp     # libfp136.so            used by fp_precision_compare.py
make fp-audit   #                        checks the generated float64/float32 sources differ only in width
make lib-ref    # libas136_ref{64,32}.so used by fp_untouched_check.py
```

The compile flags in the `Makefile` are exactly those recorded in the Slurm logs. The shared
libraries are build products and are not committed.

## Files

| File | Role |
|---|---|
| `as136.f`, `hartigan_fortran_ext.f` | AS 136 and the hook setters/getters |
| `hartigan_fortran_ctypes.py` | `run_kmns(...)`: ctypes wrapper used by the runtime benchmark |
| `gen_fp_variants.py` | generates `fp_as136_dp.f` / `fp_as136_sp.f` from `as136.f` (renamed units, one trace call per committed transfer; the `_sp` file demotes `double precision` to `real`) |
| `fp_audit.py` | inverts the precision transform on `fp_as136_sp.f` and requires an exact match with `fp_as136_dp.f` |
| `fp_trace.f` | records every accepted transfer (stage, point, from, to) for the float64/float32 comparison |
| `fp_hartigan_ctypes.py` | ctypes wrapper for the two generated variants |
| `fp_precision_compare.py` | the 13,500-run float64 vs float32 sweep → `../../results/precision/` |
| `fp_untouched_check.py` | replays the non-terminating seeds on the **unmodified** `as136.f`, compiled with and without `-freal-8-real-4` |

Slurm jobs: `../jobs/run_bestof_cpu*.job` (runtime benchmark, `make lib`),
`../jobs/run_fp_precision.job` and `../jobs/run_fp_untouched.job` (precision evidence).

## Fixed-point audit (`fp_movers`) and a known deviation in this `as136.f`

This translation deviates from the 1979 original in the optimal-transfer live-set guard: `optra`
tests `live(l2)` — the *running best candidate*, mutated in-loop — where the original tests the
per-candidate `LIVE(L)`. The variant can skip candidate clusters the original would examine, so
its `ifault=0` is a weaker claim than the original's: in principle a run can stop with points
still wanting to move. The runtime benchmark therefore audits every returned state, untimed:
`fp_movers` counts the points that would still move under the Hartigan criterion (own cost
`d²·n/(n−1)` against candidate cost `d²·n/(n+1)`, singletons pinned). A genuine fixed point has
`fp_movers = 0`. `fortran_ls` runs AS 136 with the live-set guard (`uselive=1`) and is the baseline
in the paper's runtime figures; `fortran_nols` runs it without (`uselive=0`). The precision study
uses `uselive=0`, and its untouched-source control repeats the non-terminating cases with both
settings.
