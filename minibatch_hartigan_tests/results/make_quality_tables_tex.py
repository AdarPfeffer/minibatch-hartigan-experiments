#!/usr/bin/env python3
"""Emit the paper's four clustering-quality tables as LaTeX (booktabs) source.

    python make_quality_tables_tex.py            # writes the four .tex files below
    python make_quality_tables_tex.py --check    # also prints the numbers quoted in the text

    quality/quality_mean_of_50.tex                  Table  tab:quality_mean_of_50
    quality/quality_bestof50.tex                    Table  tab:quality_bestof50
    quality/quality_by_batch_fraction_mean.tex      Table  tab:quality_by_batch_fraction_mean
    random_partition/quality_mean_of_50_rp.tex      Table  tab:quality_mean_of_50_rp

Preamble the output needs: \\usepackage{booktabs,float}.

DATA. All float64, all paired on (d, n, K, dataset_id): the same datasets and the same 50
starting partitions per dataset for every algorithm, so every comparison is per dataset.
Three algorithms: `numba_online_ls` (Online Hartigan, CPU), `jax_mb_cert` (certified
Mini-Batch Hartigan, JAX on GPU) at batch fraction p = 0.75, and `jax_lloyd` (Lloyd's
algorithm, JAX). Random-centroid tables read data/random_centroid/, the random-partition
table reads data/random_partition/.

  mean_of_50          per dataset, the NMI averaged over all 50 starting partitions
  bestof50            per dataset, the NMI of the starting partition with the smallest SSE
  by_batch_fraction   Mini-Batch mean-of-50 NMI at each p in {0.25, 0.5, 0.75, 0.9}
  mean_of_50_rp       mean_of_50 under random-partition initialization

WHAT THE +- IS. Every cell is the mean over the 100 datasets in that regime +- 1 standard
deviation ACROSS THOSE DATASETS (ddof=1). It is not the standard error of the mean (10x smaller)
and not `nmi_std` (the spread between starting partitions within one dataset).

The "MB - O" column is formed PER DATASET (Mini-Batch minus Online) and only then aggregated,
so its +- is the spread of the paired difference.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_RC = os.path.join(HERE, "data", "random_centroid")
DATA_RP = os.path.join(HERE, "data", "random_partition")
KEY = ["init_scheme", "impl", "d", "N", "K", "proportion", "dataset_id"]

# ---- the regimes the tables show --------------------------------------------------------------
NS_TABLE = [565, 2378, 10000]      # three sizes spanning the grid, for legibility
P_TABLE = 0.75                     # Mini-Batch fraction for the algorithm-comparison tables
GRID_PROPS = [0.25, 0.5, 0.75, 0.9]
KEYD = ["d", "N", "K", "dataset_id"]
IDX3 = ["d", "N", "K"]

ALG_ON, ALG_MB, ALG_LL = "online", "minibatch", "lloyd"
DIFF = "difference"
HEAD = {
    ALG_ON: "Online",
    ALG_MB: r"Mini-Batch ($p = %g$)" % P_TABLE,
    ALG_LL: "Lloyd",
    DIFF:   r"MB $-$ O (Mean Difference)",
}

# Captions exactly as in the paper.
CAPTIONS = {
    "quality_mean_of_50": (
        "Mean NMI of 5000 trials per regime for selected regimes. Each mean value is given within "
        "one standard deviation of the mean NMI for each dataset in the regime. Last column shows "
        "the mean difference of the mean NMI of the 100 datasets run per regime, within on "
        "standard deviation of the differences. We report Mini-Batch's statistics only at "
        "$p = 0.75$, but there is no measurable or consistent difference between proportions. "
        "Online and Mini-Batch experience little disagreement, with the mean difference never "
        "exceeding 0.01"),
    "quality_bestof50": (
        "Average of the NMIs yielded by the best starting partition for each dataset in a regime. "
        "The best starting partition is the starting partition which yields the smallest SSE upon "
        "convergence. Errors are reported as plus-or-minus one standard deviation in the NMI of "
        "the best starting partitions for each regime. We also show the mean of differences "
        "between Mini-Batch and Online's best starting partition for each dataset in a regime, "
        "within one standard deviation of the differences. Mini-Batch, Online, and Lloyd's "
        "algorithm experience few significant disagreements, often landing on identical mean "
        "NMIs. All regimes, datasets, starting partitions, and samples selected each iteration "
        "for Mini-Batch are  identical to the runtime test."),
    "quality_by_batch_fraction_mean": (
        "Mini-Batch clustering quality across batch proportions.  Each cell is the mean NMI over "
        "the 5000 trials the particular regime was tested on. These values are reported within "
        "one standard deviation of the datasets' mean NMIs. No row varies by more than 0.009 "
        "across $p$."),
    "quality_mean_of_50_rp": (
        "Clustering quality under random-partition initialization for select regimes. Each cell "
        "is the mean NMI of all 5000 trials per regime plus-or-minus one standard deviation in "
        "the datasets' mean NMIs. The final column is the mean difference of mean NMIs per "
        "dataset between Mini-Batch and Online, within one standard deviation. All "
        "implementations ran on the same grid of starting partitions. Mini-Batch continues to "
        "perform at the same level as Online, with differences rarely exceeding 0.01 NMI."),
}


# ---- data -------------------------------------------------------------------------------------
def _read(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    return pd.read_csv(path)            # pandas infers gzip from the .csv.gz suffix


def load_quality(data_dir, cpu_name, jax_name, lloyd_name, scheme):
    """CPU table (numba_online_ls) + H200 float64 JAX Mini-Batch + JAX Lloyd, one init scheme."""
    cpu = _read(os.path.join(data_dir, cpu_name))
    h64 = _read(os.path.join(data_dir, jax_name)).drop_duplicates(subset=KEY)
    h64["impl"] = "jax_mb_cert_f64"
    ll = _read(os.path.join(data_dir, lloyd_name)).drop_duplicates(subset=KEY)
    Q = pd.concat([cpu, h64, ll], ignore_index=True)
    assert (Q.init_scheme == scheme).all(), scheme
    report = (f"{scheme}: cpu {len(cpu):,} rows | jax_mb_cert_f64 {len(h64):,} rows, "
              f"{h64.groupby(KEYD).ngroups:,}/13,500 cells | jax_lloyd {len(ll):,} rows, "
              f"{ll.groupby(KEYD).ngroups:,}/13,500 cells")
    return Q, report


def percell(Q, impl, prop, col):
    """One value per (d, N, K, dataset_id) for this impl (prop None = not proportion-swept)."""
    g = Q[Q.impl == impl]
    g = g[g.proportion < 0] if prop is None else g[np.isclose(g.proportion, prop)]
    s = g[g.N.isin(NS_TABLE)].set_index(KEYD)[col]
    return s[~s.index.duplicated()]


def mean_sd(series_by_label, full_idx):
    """{label: per-dataset Series} -> (mean frame, sd frame), both indexed (d, N, K)."""
    m = pd.DataFrame({k: v.groupby(level=IDX3).mean()
                      for k, v in series_by_label.items()}).reindex(full_idx).dropna(how="all")
    sd = pd.DataFrame({k: v.groupby(level=IDX3).std(ddof=1)
                       for k, v in series_by_label.items()}).reindex(m.index)
    return m, sd


# ---- LaTeX ------------------------------------------------------------------------------------
def _num(x):
    """Integer with a LaTeX-safe thousands separator, for use inside math mode."""
    return f"{int(x):,}".replace(",", "{,}")


def _cell(v, e, fmt):
    if not np.isfinite(v):
        return "--"
    body = fmt.format(v)
    if np.isfinite(e):
        body += r" \pm " + f"{e:.4f}"
    return f"${body}$"


def latex_table(m, sd, cols, fmts, caption, label):
    """A (d,N,K)-indexed frame as a booktabs tabular, one row per regime."""
    ds = sorted({i[0] for i in m.index})
    out = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{%s}" % caption, r"\label{%s}" % label,
           r"\begin{tabular}{rrr%s}" % ("c" * len(cols)), r"\toprule",
           " & ".join([r"$d$", r"$n$", r"$K$"] + [HEAD.get(c, c) for c in cols]) + r" \\",
           r"\midrule"]
    # The d and n labels are printed on the first row of their block and left blank afterwards.
    for di, d in enumerate(ds):
        sub_d = m[m.index.get_level_values("d") == d]
        ns = sorted({i[1] for i in sub_d.index})
        for ni, n in enumerate(ns):
            sub_n = sub_d[sub_d.index.get_level_values("N") == n]
            ks = sorted({i[2] for i in sub_n.index})
            for ki, k in enumerate(ks):
                row, erow = m.loc[(d, n, k)], sd.loc[(d, n, k)]
                cells = [_cell(row[c], erow[c], fmts[c]) for c in cols]
                dlab = ("$%s$" % _num(d)) if (ni == 0 and ki == 0) else ""
                nlab = ("$%s$" % _num(n)) if ki == 0 else ""
                out.append("  %s & %s & $%d$ & %s \\\\" % (dlab, nlab, k, " & ".join(cells)))
            if ni != len(ns) - 1:
                out.append(r"  \cmidrule(l){2-%d}" % (3 + len(cols)))
        out.append(r"\midrule" if di != len(ds) - 1 else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


# ---- the four tables --------------------------------------------------------------------------
def comparison_table(Q, col, name):
    """Online vs Mini-Batch (p = 0.75) vs Lloyd, plus the paired MB - O difference."""
    DS, KS = sorted(Q.d.unique()), sorted(Q.K.unique())
    full_idx = pd.MultiIndex.from_product([DS, NS_TABLE, KS], names=IDX3)
    ser = {ALG_ON: percell(Q, "numba_online_ls", None, col),
           ALG_MB: percell(Q, "jax_mb_cert_f64", P_TABLE, col),
           ALG_LL: percell(Q, "jax_lloyd", None, col)}
    ser[DIFF] = (ser[ALG_MB] - ser[ALG_ON]).dropna()          # PAIRED, per dataset
    m, sd = mean_sd(ser, full_idx)
    cols = [ALG_ON, ALG_MB, ALG_LL, DIFF]
    fmts = {**{a: "{:.4f}" for a in (ALG_ON, ALG_MB, ALG_LL)}, DIFF: "{:+.4f}"}
    return latex_table(m, sd, cols, fmts, CAPTIONS[name], "tab:" + name), m


def by_batch_fraction_table(Q):
    """Mini-Batch mean-of-50 NMI across batch fractions (Lloyd and Online have none)."""
    DS, KS = sorted(Q.d.unique()), sorted(Q.K.unique())
    full_idx = pd.MultiIndex.from_product([DS, NS_TABLE, KS], names=IDX3)
    ser = {f"$p = {p:g}$": percell(Q, "jax_mb_cert_f64", p, "nmi_mean") for p in GRID_PROPS}
    m, sd = mean_sd(ser, full_idx)
    name = "quality_by_batch_fraction_mean"
    return latex_table(m, sd, list(ser), {c: "{:.4f}" for c in ser},
                       CAPTIONS[name], "tab:" + name), m


def check(m1, m2, mp, mrp):
    """Recompute the numbers the paper's text quotes about these tables."""
    d1 = m1[DIFF]
    print(f"  mean_of_50     MB - O: {d1.min():+.4f} to {d1.max():+.4f}; "
          f"largest |mean difference| {d1.abs().max():.4f} (text: 0.0073)")
    ll1 = (m1[ALG_LL] - m1[ALG_ON])
    print(f"                 Lloyd below Online in {int((ll1 < 0).sum())}/{len(ll1)} regimes")
    d2 = m2[DIFF]
    print(f"  bestof50       MB - O: {d2.min():+.4f} to {d2.max():+.4f}")
    spread = mp.max(axis=1) - mp.min(axis=1)
    print(f"  by_batch_frac  largest spread across p within a row: {spread.max():.4f} "
          f"(caption: no row varies by more than 0.009)")
    drp = mrp[DIFF]
    i = mrp[ALG_LL].idxmin()
    print(f"  mean_of_50_rp  MB - O: {drp.min():+.4f} to {drp.max():+.4f}, "
          f"|difference| < 0.01 in {int((drp.abs() < 0.01).sum())}/{len(drp)} regimes")
    print(f"                 lowest Lloyd mean NMI {mrp.loc[i, ALG_LL]:.4f} at (d, n, K) = "
          f"{tuple(int(x) for x in i)}, "
          f"where Mini-Batch scored {mrp.loc[i, ALG_MB]:.4f} (text: 0.0829 / 0.9714)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "quality"),
                    help="directory for the three random-centroid tables (default: quality/)")
    ap.add_argument("--out-rp", default=os.path.join(HERE, "random_partition"),
                    help="directory for the random-partition table (default: random_partition/)")
    ap.add_argument("--check", action="store_true",
                    help="print the numbers the paper's text quotes, recomputed from the data")
    args = ap.parse_args()

    Q, report = load_quality(DATA_RC, "bestof_cpu_full.csv.gz",
                             "bestof_jaxtime_rc_29208159.csv.gz",
                             "bestof_lloyd_23611965.csv.gz", "random_centroid")
    print(report)
    QRP, report_rp = load_quality(DATA_RP, "bestof_rpart_cpu_26508322.csv.gz",
                                  "bestof_jaxtime_rp_29208159.csv.gz",
                                  "bestof_rpart_lloyd_29098778.csv.gz", "random_partition")
    print(report_rp)

    t1, m1 = comparison_table(Q, "nmi_mean", "quality_mean_of_50")
    t2, m2 = comparison_table(Q, "nmi_at_min_sse", "quality_bestof50")
    t3, mp = by_batch_fraction_table(Q)
    t4, mrp = comparison_table(QRP, "nmi_mean", "quality_mean_of_50_rp")
    if args.check:
        print("\nnumbers quoted in the paper, recomputed:")
        check(m1, m2, mp, mrp)

    header = ("% Generated by make_quality_tables_tex.py — do not edit by hand.\n"
              "% Preamble: \\usepackage{booktabs,float}\n")
    for out_dir, name, body in [(args.out, "quality_mean_of_50", t1),
                                (args.out, "quality_bestof50", t2),
                                (args.out, "quality_by_batch_fraction_mean", t3),
                                (args.out_rp, "quality_mean_of_50_rp", t4)]:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, name + ".tex")
        with open(p, "w") as f:
            f.write(header + body + "\n")
        print(f"wrote {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
