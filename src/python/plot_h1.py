"""Publication-quality H1 figure: HD-EMA learning curves with shaded variance bands.

Implements the "Option A" chart for Hypothesis 1: for every eval episode, plot the
mean HD-EMA across the 5 seeds as a solid line and draw a translucent band around it
(standard error by default) so the reader can see both the trend and its spread.

It reads the per-arm aggregate files that analyze.py already writes:

    <results-dir>/agg_champion.csv
    <results-dir>/agg_selfplay.csv     (Win-Max opponent)
    <results-dir>/agg_teacher.csv      (TD-Max opponent)

Each agg_*.csv has columns: episode, n_seeds, ..., hdiff_ema_mean, hdiff_ema_std, ...

Usage:
    python plot_h1.py                                   # uses defaults below
    python plot_h1.py --results-dir ../2026-07-31_13-12-49
    python plot_h1.py --band ci95                       # 95% CI instead of standard error
    python plot_h1.py --band std                        # +/- 1 std (raw seed spread)
    python plot_h1.py --metric health_diff              # unsmoothed health diff instead
"""

import argparse
import csv
import math
import os

# ---- paper terminology + fixed colors (match the H1 figure legend) --------------------
# arm tag in the CSV filenames -> (legend label, line color)
ARM_STYLE = {
    "champion": ("Champion FSM", "#1f77b4"),  # blue
    "selfplay": ("Win-Max FSM", "#ff7f0e"),   # orange
    "teacher":  ("TD-Max FSM", "#2ca02c"),    # green
}
# draw order so the TD-Max line sits on top
ARM_ORDER = ["champion", "selfplay", "teacher"]

# two-sided 95% t-multipliers for small samples (df = n_seeds - 1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t95(df):
    return _T95.get(df, 1.96)  # large-sample fallback


def read_agg(path, metric):
    """Return (episodes, means, half_widths_std) reading one agg_<arm>.csv.

    half_widths_std is the raw per-episode std across seeds; the band type
    (se / std / ci95) is applied later so a single read supports all modes.
    """
    eps, means, stds, ns = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eps.append(int(row["episode"]))
            means.append(float(row[f"{metric}_mean"]))
            stds.append(float(row[f"{metric}_std"]))
            ns.append(int(row["n_seeds"]))
    return eps, means, stds, ns


def band_halfwidth(std, n, mode):
    """Convert a per-episode std across seeds into the plotted band half-width."""
    if n <= 1:
        return 0.0
    if mode == "std":
        return std
    se = std / math.sqrt(n)
    if mode == "se":
        return se
    if mode == "ci95":
        return _t95(n - 1) * se
    raise ValueError(f"unknown band mode {mode!r}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "..", "2026-07-31_13-12-49"),
                    help="directory holding agg_<arm>.csv files")
    ap.add_argument("--metric", default="hdiff_ema",
                    help="metric column prefix to plot (default: hdiff_ema)")
    ap.add_argument("--band", default="se", choices=["se", "std", "ci95"],
                    help="shaded band: standard error (default), raw std, or 95%% CI")
    ap.add_argument("--out", default=None, help="output PNG path (default: <results-dir>/h1_hdema.png)")
    ap.add_argument("--title", default="Learning Performance Under Different Training Opponents")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"matplotlib is required: pip install matplotlib  ({e})")

    band_label = {"se": "shaded band: +/-1 standard error",
                  "std": "shaded band: +/-1 std across seeds",
                  "ci95": "shaded band: 95% confidence interval"}[args.band]

    plt.figure(figsize=(8, 5))
    plotted = 0
    for arm in ARM_ORDER:
        path = os.path.join(args.results_dir, f"agg_{arm}.csv")
        if not os.path.exists(path):
            print(f"(skip, not found) {path}")
            continue
        label, color = ARM_STYLE[arm]
        eps, means, stds, ns = read_agg(path, args.metric)
        halves = [band_halfwidth(s, n, args.band) for s, n in zip(stds, ns)]
        lo = [m - h for m, h in zip(means, halves)]
        hi = [m + h for m, h in zip(means, halves)]
        plt.fill_between(eps, lo, hi, alpha=0.18, color=color, linewidth=0)
        plt.plot(eps, means, label=label, color=color, linewidth=2.5)
        plotted += 1

    if not plotted:
        raise SystemExit(f"No agg_<arm>.csv files found in {args.results_dir!r}. "
                         "Run analyze.py first to generate them.")

    plt.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    plt.xlabel("Training episode")
    plt.ylabel("HD-EMA (health difference, mean across 5 seeds)")
    plt.title(args.title)
    plt.legend(title=band_label, loc="lower right")
    plt.grid(True, alpha=0.3)

    out = args.out or os.path.join(args.results_dir, f"h1_{args.metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}  (metric={args.metric}, band={args.band})")


if __name__ == "__main__":
    main()
