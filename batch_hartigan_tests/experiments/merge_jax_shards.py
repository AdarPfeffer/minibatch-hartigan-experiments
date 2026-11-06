"""merge_jax_shards.py — combine the per-shard best-of failure-mode CSVs into one.

    python merge_jax_shards.py [OUT.csv] [IN1.csv IN2.csv ...]

With no arguments, merges ./results_jax_shard*.csv -> ./results_jax.csv. Each
(amount_of_clusters, dim, intracluster_variance, intercluster_variance, cluster_size) regime
is produced by exactly ONE shard (disjoint regime_index % N), so a straight concat + de-dup +
sort yields the single tidy table. Prints a coverage report and flags duplicates/gaps.
"""
import glob
import sys

import pandas as pd

KEY = ["amount_of_clusters", "dim", "intracluster_variance", "intercluster_variance",
       "cluster_size"]
EXPECTED = 14 * 5 * 4 * 19          # full grid = 5,320 regimes


def main():
    args = sys.argv[1:]
    if args and args[0].endswith(".csv") and (len(args) == 1 or args[1].endswith(".csv")):
        out = args[0]
        ins = args[1:] or sorted(glob.glob("results_jax_shard*.csv"))
    else:
        out = "results_jax.csv"
        ins = sorted(glob.glob("results_jax_shard*.csv"))
    if not ins:
        raise SystemExit("no input CSVs found (results_jax_shard*.csv) — pass them explicitly")

    frames = []
    for p in ins:
        df = pd.read_csv(p)
        print(f"  {p}: {len(df)} regimes")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    dups = df.duplicated(subset=KEY, keep=False)
    if dups.any():
        print(f"WARNING: {int(dups.sum())} rows share a regime key across shards "
              f"(each regime should be produced by exactly one shard) — keeping the first.")
    before = len(df)
    df = df.drop_duplicates(subset=KEY, keep="first").sort_values(KEY).reset_index(drop=True)
    df.to_csv(out, index=False)

    print(f"\nmerged {len(ins)} shard(s) -> {out}: {len(df)} regimes "
          f"({before - len(df)} duplicate(s) dropped)")
    dims = sorted(int(x) for x in df.dim.unique())
    Ks = sorted(int(x) for x in df.amount_of_clusters.unique())
    snr = sorted({(int(a), int(b))
                  for a, b in zip(df.intracluster_variance, df.intercluster_variance)})
    print(f"coverage: dim={dims}  K={Ks}  cluster_size={len(df.cluster_size.unique())} vals  "
          f"SNR_pairs={snr}")
    if len(df) != EXPECTED:
        print(f"NOTE: {len(df)} regimes != full grid {EXPECTED} — a shard may be incomplete "
              f"(check each shard's .out ended with 'DONE') or you merged a partial/smoke run.")


if __name__ == "__main__":
    main()
