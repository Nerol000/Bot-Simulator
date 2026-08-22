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

# ---- paper terminology + fixed colors (match the figure legends) ----------------------
# code arm tag (from the CSV filenames) -> (legend label, line color). Aligns with the
# --opponent names in train_tabular.py so the figure legend uses the PAPER's opponent names.
#   H1 compares champion vs teacher (TD-Max); H2 compares champion vs improve.
ARM_STYLE = {
    "champion": ("Champion FSM", "#1f77b4"),    # blue  - win-optimized (H1 & H2 baseline)
    "teacher":  ("TD-Max FSM", "#2ca02c"),      # green - maximize learner TD error (H1)
    "td_error": ("TD-Max FSM", "#2ca02c"),      # green - alias of teacher
    "improve":  ("Improvement FSM", "#d62728"), # red   - maximize learner improvement (H2)
    # extra arms that may exist in a sweep but are not part of H1/H2:
    "win_max":  ("Win-Max FSM", "#ff7f0e"),     # orange- alias of champion (not used separately)
    "selfplay": ("Self-Play FSM", "#9467bd"),   # purple- frozen self-snapshots
}
# preferred draw order (baseline first, hypothesis arm on top); unknown arms appended.
ARM_ORDER = ["selfplay", "win_max", "champion", "teacher", "td_error", "improve"]

# named figure presets: which arms belong on each hypothesis chart.
FIGURES = {
    "h1": ["champion", "teacher"],   # Champion vs TD-Max
    "h2": ["champion", "improve"],   # Champion vs Improve
}

# fallback colors for arm tags not in ARM_STYLE (so a new arm still plots, just unlabeled-nicely)
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

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
    ap.add_argument("--figure", default="h1", choices=["h1", "h2", "all"],
                    help="which comparison to plot: h1 (champion vs TD-Max), "
                         "h2 (champion vs improve), or all (every arm found)")
    ap.add_argument("--metric", default="hdiff_ema",
                    help="metric column prefix to plot (default: hdiff_ema)")
    ap.add_argument("--band", default="se", choices=["se", "std", "ci95"],
                    help="shaded band: standard error (default), raw std, or 95%% CI")
    ap.add_argument("--out", default=None, help="output PNG path (default: <results-dir>/<figure>_<metric>.png)")
    ap.add_argument("--title", default=None, help="chart title (default depends on --figure)")
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

    default_titles = {
        "h1": "H1: Learning Performance -- Champion vs TD-Max Opponent",
        "h2": "H2: Learning Performance -- Champion vs Improvement Opponent",
        "all": "Learning Performance Under Different Training Opponents",
    }
    title = args.title or default_titles[args.figure]

    # Discover which arms this sweep actually produced (agg_<arm>.csv), so no arm is silently
    # dropped and no missing arm is silently expected. Order known arms by ARM_ORDER, then append
    # any unrecognized arm alphabetically so a brand-new opponent still shows up on the chart.
    import glob
    found = {}
    for p in glob.glob(os.path.join(args.results_dir, "agg_*.csv")):
        arm = os.path.splitext(os.path.basename(p))[0][len("agg_"):]
        found[arm] = p
    if not found:
        raise SystemExit(f"No agg_<arm>.csv files found in {args.results_dir!r}. "
                         "Run analyze.py first to generate them.")

    # Restrict to the arms for the requested figure (h1/h2); 'all' keeps everything found.
    if args.figure == "all":
        wanted = set(found)
    else:
        wanted = set(FIGURES[args.figure])
        missing = [a for a in FIGURES[args.figure] if a not in found]
        if missing:
            raise SystemExit(f"--figure {args.figure} needs arms {FIGURES[args.figure]} but "
                             f"{missing} are missing from {args.results_dir!r}. "
                             f"Available: {sorted(found)}. Re-run the sweep including {missing}.")
    ordered = ([a for a in ARM_ORDER if a in found and a in wanted]
               + sorted(a for a in found if a not in ARM_ORDER and a in wanted))
    print(f"[{args.figure}] plotting arms: " + ", ".join(ordered))
    # collapse teacher/td_error alias so the same opponent isn't drawn twice
    seen_labels = set()

    plt.figure(figsize=(8, 5))
    plotted = 0
    n_seen = 0
    fb = 0
    for arm in ordered:
        path = found[arm]
        if arm in ARM_STYLE:
            label, color = ARM_STYLE[arm]
        else:
            label, color = arm, _FALLBACK_COLORS[fb % len(_FALLBACK_COLORS)]
            fb += 1
            print(f"  (note) arm '{arm}' has no paper label/color; plotting with raw name.")
        if label in seen_labels:
            print(f"  (skip duplicate label) '{arm}' -> '{label}' already plotted.")
            continue
        seen_labels.add(label)
        eps, means, stds, ns = read_agg(path, args.metric)
        if ns:
            n_seen = max(n_seen, max(ns))
        halves = [band_halfwidth(s, n, args.band) for s, n in zip(stds, ns)]
        lo = [m - h for m, h in zip(means, halves)]
        hi = [m + h for m, h in zip(means, halves)]
        plt.fill_between(eps, lo, hi, alpha=0.18, color=color, linewidth=0)
        plt.plot(eps, means, label=label, color=color, linewidth=2.5)
        plotted += 1

    plt.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    plt.xlabel("Training episode")
    plt.ylabel(f"HD-EMA (health difference, mean across {n_seen} seeds)")
    plt.title(title)
    plt.legend(title=band_label, loc="lower right")
    plt.grid(True, alpha=0.3)

    out = args.out or os.path.join(args.results_dir, f"{args.figure}_{args.metric}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}  (figure={args.figure}, metric={args.metric}, band={args.band})")


if __name__ == "__main__":
    main()
