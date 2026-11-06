"""merge_bestof_csv.py — combine the GPU-half and CPU-half best-of-init CSVs into one.

    python merge_bestof_csv.py OUT.csv IN1.csv IN2.csv [IN3.csv ...]
    python merge_bestof_csv.py OUT.csv --all NEW*.csv --online-only PREFIX*.csv

`--online-only` marks inputs whose MINI-BATCH rows must be discarded: pre-fix data taken before
the mini-batch empty-cluster fix, where the online/fortran rows are still valid but every
mini-batch row is not. This cannot be handled by de-duplication — pre-fix mini-batch rows carry
proportions (e.g. 0.95) that no longer exist post-fix, so they would survive as extra rows with
no post-fix counterpart. They must be dropped by IMPL.

The GPU job writes the jax/cuda impls (all dims); the CPU job(s) write the numba/fortran
impls (possibly split by dimension). Each (init_scheme, impl, d, N, K, proportion,
dataset_id) cell is produced by exactly ONE job, so a straight concat + de-dup + sort
yields the single tidy results table. Prints a coverage report and flags duplicates.
"""
import sys

import pandas as pd

KEY = ["init_scheme", "impl", "d", "N", "K", "proportion", "dataset_id"]


ONLINE_IMPLS = {"numba_online_ls", "jax_online_tol", "cuda_online",
                "fortran_ls", "fortran_nols"}


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, rest = sys.argv[1], sys.argv[2:]
    all_ins, online_ins, bucket = [], [], "all"
    for a in rest:
        if a == "--all":
            bucket = "all"
        elif a == "--online-only":
            bucket = "online"
        else:
            (all_ins if bucket == "all" else online_ins).append(a)
    if not all_ins and not online_ins:
        raise SystemExit(__doc__)

    frames = []
    for p in all_ins:
        df = pd.read_csv(p)
        print(f"  {p}: {len(df)} rows, impls={sorted(df.impl.unique())}")
        frames.append(df)
    for p in online_ins:
        df = pd.read_csv(p)
        before = len(df)
        df = df[df.impl.isin(ONLINE_IMPLS)]
        print(f"  {p}: {len(df)} rows kept of {before} (online-only; dropped "
              f"{before - len(df)} mini-batch rows), impls={sorted(df.impl.unique())}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    dups = df.duplicated(subset=KEY, keep=False)
    if dups.any():
        print(f"WARNING: {int(dups.sum())} rows share a cell key across inputs "
              f"(a cell should be produced by exactly one job) — keeping the first.")
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="first")
    df = df.sort_values(KEY).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\nmerged {len(all_ins)+len(online_ins)} files -> {out}: {len(df)} rows "
          f"({before - len(df)} duplicate(s) dropped)")
    print(f"coverage: impls={sorted(df.impl.unique())}  "
          f"d={sorted(df.d.unique())}  N={len(df.N.unique())} sizes  "
          f"K={sorted(df.K.unique())}")


if __name__ == "__main__":
    main()
