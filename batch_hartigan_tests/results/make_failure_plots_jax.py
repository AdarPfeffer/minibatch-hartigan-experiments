"""
make_failure_plots_jax.py — reproduce the Batch-Hartigan failure-mode figures for the
high-dimensional JAX sweep (data/results_jax.csv.gz), using the SAME formula and layout as the
original study (main_failure_plots.ipynb): 4 SNR rows x 5 dim columns of cividis heatmaps,
one pixel = one regime = 1000 trials, metric = 100 * count / trials, fixed 0-100% scale.

Only the dim axis differs from the original: DIMS = [100, 500, 1000, 5000, 10000].

    python make_failure_plots_jax.py         # prints analysis + writes figures/*.png
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "results_jax.csv.gz")

# Three mutually-exclusive failure modes -> rates as percentages (identical to the original).
df["total_fail"] = 100 * (df.loop_count + df.zero_count + df.max_iter_count) / df.trials
df["loop_rate"] = 100 * df.loop_count / df.trials
df["zero_rate"] = 100 * df.zero_count / df.trials

CS_VALS = sorted(df.cluster_size.unique())            # 19 values 2..300
K_VALS = sorted(df.amount_of_clusters.unique())       # 14 values 2..15
DIMS = [100, 500, 1000, 5000, 10000]                  # panel COLUMNS (high-d restructure)
RATIO_STEPS = [(1, 10), (4, 7), (7, 4), (10, 1)]      # panel ROWS (intra, inter), top->bottom


def panel_matrix(metric, dim, intra, inter):
    """One (dim, intra, inter) slice as a (14, 19) matrix: rows = K asc, cols = cluster size asc."""
    sub = df[(df.dim == dim)
             & (df.intracluster_variance == intra)
             & (df.intercluster_variance == inter)]
    P = sub.pivot_table(index="amount_of_clusters", columns="cluster_size", values=metric)
    return P.loc[K_VALS, CS_VALS].values


def plot_failure_grid(metric, cbar_label, fig_fname=None):
    fig, ax = plt.subplots(
        len(RATIO_STEPS), len(DIMS),
        figsize=(5 * len(DIMS), 4 * len(RATIO_STEPS)),
        sharex=True, sharey=True, layout="compressed",
    )
    for i, (a, b) in enumerate(RATIO_STEPS):
        for j, d in enumerate(DIMS):
            im = ax[i, j].imshow(
                panel_matrix(metric, d, a, b),
                origin="lower", cmap="cividis",           # the blue->yellow map, as before
                vmin=0, vmax=100, aspect="auto",          # FIXED 0-100 scale, comparable across figs
            )
    for j, d in enumerate(DIMS):
        ax[0, j].set_title(f"d = {d}")
    ax[-1, 0].set_xticks(range(len(CS_VALS)), CS_VALS, rotation=90, fontsize=13)
    for j in range(1, len(DIMS)):
        ax[-1, j].tick_params(labelbottom=False)
    for i, (a, b) in enumerate(RATIO_STEPS):
        # (a, b) = (intra sigma^2, inter tau^2). Label the SNR = tau^2/sigma^2 = inter/intra
        # (signal/noise), the paper's convention — NOT intra/inter. Top row = most separable.
        ax[i, 0].set_ylabel(rf"$\tau^2/\sigma^2$ {b}/{a}" + "\nK")
        ax[i, 0].set_yticks(range(0, len(K_VALS), 2), K_VALS[::2], fontsize=14)
    ax[-1, 0].set_xlabel("Cluster size")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label, rotation=270, labelpad=30)
    if fig_fname is not None:
        pathlib.Path(fig_fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_fname, bbox_inches="tight", dpi=200)
    return fig, ax


def analyze():
    print("=" * 78)
    print("ANALYSIS — results_jax.csv")
    print("=" * 78)
    print(f"rows: {len(df)} (expected 14*5*4*19 = {14*5*4*19})")
    print(f"dims: {sorted(int(x) for x in df.dim.unique())}")
    print(f"K:    {sorted(int(x) for x in df.amount_of_clusters.unique())}")
    print(f"cs:   {len(CS_VALS)} values {CS_VALS}")
    snr = sorted({(int(a), int(b)) for a, b in
                  zip(df.intracluster_variance, df.intercluster_variance)})
    print(f"SNR (intra,inter): {snr}")
    print(f"trials: min={df.trials.min()} max={df.trials.max()} "
          f"(all==1000? {bool((df.trials == 1000).all())})")
    print(f"max_iter_count: total={int(df.max_iter_count.sum())} over "
          f"{int(df.trials.sum())} trials  (rate {df.max_iter_count.sum()/df.trials.sum():.2e})")
    print(f"max loop_size seen (max_loop_size col): {int(df.max_loop_size.max())}")

    print("\n-- marginal failure rates by SNR = tau^2/sigma^2 = inter/intra --")
    for (a, b) in RATIO_STEPS:
        s = df[(df.intracluster_variance == a) & (df.intercluster_variance == b)]
        print(f"  tau^2/sigma^2 {b}/{a} (SNR={b/a:.2f}): "
              f"loop={s.loop_rate.mean():5.1f}%  zero={s.zero_rate.mean():5.1f}%  "
              f"total={s.total_fail.mean():5.1f}%")

    print("\n-- marginal failure rates by dim --")
    for d in DIMS:
        s = df[df.dim == d]
        print(f"  d={d:>5}: loop={s.loop_rate.mean():5.1f}%  zero={s.zero_rate.mean():5.1f}%  "
              f"total={s.total_fail.mean():5.1f}%")

    print("\n-- loop_rate (%) by dim x SNR (does looping persist in high d?) --")
    print("        " + "".join(f"d={d:>6}" for d in DIMS))
    for (a, b) in RATIO_STEPS:
        row = []
        for d in DIMS:
            s = df[(df.dim == d) & (df.intracluster_variance == a) & (df.intercluster_variance == b)]
            row.append(f"{s.loop_rate.mean():7.1f}")
        print(f"  {b:>2}/{a:<2} " + "".join(row))   # tau^2/sigma^2 = inter/intra

    print("\n-- zero_rate (%) by dim x SNR --")
    print("        " + "".join(f"d={d:>6}" for d in DIMS))
    for (a, b) in RATIO_STEPS:
        row = []
        for d in DIMS:
            s = df[(df.dim == d) & (df.intracluster_variance == a) & (df.intercluster_variance == b)]
            row.append(f"{s.zero_rate.mean():7.1f}")
        print(f"  {b:>2}/{a:<2} " + "".join(row))   # tau^2/sigma^2 = inter/intra

    hi = df.loc[df.loop_rate.idxmax()]
    print(f"\n-- peak loop regime: d={int(hi.dim)} K={int(hi.amount_of_clusters)} "
          f"cs={int(hi.cluster_size)} snr={int(hi.intracluster_variance)}/{int(hi.intercluster_variance)}"
          f" -> loop_rate={hi.loop_rate:.1f}%")
    hz = df.loc[df.zero_rate.idxmax()]
    print(f"-- peak zero regime: d={int(hz.dim)} K={int(hz.amount_of_clusters)} "
          f"cs={int(hz.cluster_size)} snr={int(hz.intracluster_variance)}/{int(hz.intercluster_variance)}"
          f" -> zero_rate={hz.zero_rate:.1f}%")


if __name__ == "__main__":
    analyze()
    plot_failure_grid("loop_rate", "Infinite-loop rate (%)", HERE / "figures" / "failure_loop_rate_jax.png")
    plot_failure_grid("zero_rate", "Empty-cluster rate (%)", HERE / "figures" / "failure_zero_rate_jax.png")
    print("\nwrote figures/failure_{loop_rate,zero_rate}_jax.png")
