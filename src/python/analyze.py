"""Aggregate and summarize the tabular H1/H2 sweep results.

Reads every runs/<arm>_s<seed>_ep<N>_metrics.csv produced by train_tabular.py, groups the runs
by ARM (champion / teacher / selfplay / ...), averages across SEEDS at each eval episode, and
emits the two experiment deliverables:

  1) A per-arm SUMMARY TABLE (final-value mean +/- std across seeds) -> stdout + summary.csv
  2) Per-arm AGGREGATED LEARNING CURVES (mean +/- std at each episode) -> agg_<arm>.csv,
     and, if matplotlib is available, health_diff.png / avg_td.png plots.
  3) A single COMBINED long-format table (one row per arm x episode, all metrics averaged
     across seeds) -> combined.csv, for easy brain-type vs episode comparison in one file.

The metrics CSV columns are: episode, avg_td, health_diff, win_rate, dmg_ratio, avg_steps.

  H2 (learner improvement): compare arms on the `health_diff` curve + its final value.
  H1 (training signal):     compare arms on `avg_td` (teacher should sit above champion).

Usage:
    python analyze.py                      # reads ./runs, writes ./runs/analysis
    python analyze.py --runs-dir runs --out-dir runs/analysis
    python analyze.py --metric health_diff # which metric the console curve prints
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

# tag format from train_tabular.py: <arm>_s<seed>_ep<episodes>_metrics.csv
_NAME_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)_ep(?P<episodes>\d+)_metrics\.csv$")

METRIC_COLS = ["avg_td", "health_diff", "hdiff_ema", "hdiff_best", "win_rate", "dmg_ratio", "avg_steps"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    """Sample standard deviation (ddof=1); 0 for a single value."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def load_runs(runs_dir):
    """Return {arm: {seed: {episode: {metric: value}}}} parsed from *_metrics.csv."""
    runs = defaultdict(lambda: defaultdict(dict))
    paths = sorted(glob.glob(os.path.join(runs_dir, "*_metrics.csv")))
    if not paths:
        raise SystemExit(f"No *_metrics.csv found in {runs_dir!r}. Point --runs-dir at the sweep output.")
    for path in paths:
        m = _NAME_RE.match(os.path.basename(path))
        if not m:
            print(f"  (skip, unrecognized name) {os.path.basename(path)}")
            continue
        arm, seed = m["arm"], int(m["seed"])
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ep = int(row["episode"])
                # Tolerate older metrics CSVs that predate the smoothed columns: default to the
                # raw health_diff for hdiff_ema/hdiff_best, 0.0 for anything else missing.
                def _col(c):
                    if c in row and row[c] != "":
                        return float(row[c])
                    if c in ("hdiff_ema", "hdiff_best") and row.get("health_diff", "") != "":
                        return float(row["health_diff"])
                    return 0.0
                runs[arm][seed][ep] = {c: _col(c) for c in METRIC_COLS}
    return runs


def aggregate_arm(seed_map):
    """Given {seed: {episode: {metric: value}}}, return sorted list of per-episode aggregates:
    [{episode, n_seeds, <metric>_mean, <metric>_std, ...}] over episodes common to all seeds."""
    seeds = list(seed_map)
    # episodes present in EVERY seed (so mean/std compare like with like)
    common = set.intersection(*(set(seed_map[s]) for s in seeds)) if seeds else set()
    rows = []
    for ep in sorted(common):
        agg = {"episode": ep, "n_seeds": len(seeds)}
        for c in METRIC_COLS:
            vals = [seed_map[s][ep][c] for s in seeds]
            agg[f"{c}_mean"] = _mean(vals)
            agg[f"{c}_std"] = _std(vals)
        rows.append(agg)
    return rows


def write_agg_csv(path, rows):
    if not rows:
        return
    cols = ["episode", "n_seeds"] + [f"{c}_{stat}" for c in METRIC_COLS for stat in ("mean", "std")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})


def write_combined_csv(path, agg_by_arm):
    """One tidy long-format table for ALL arms: a row per (arm, episode) with every metric
    averaged across seeds. Sorted by arm then episode so you can eyeball or pivot on
    'brain type x episode' in Excel/pandas without touching the per-arm files."""
    cols = ["arm", "episode", "n_seeds"] + [f"{c}_{stat}" for c in METRIC_COLS for stat in ("mean", "std")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm in sorted(agg_by_arm):
            for r in agg_by_arm[arm]:
                row = {"arm": arm}
                row.update({k: r[k] for k in cols if k != "arm"})
                w.writerow(row)
    print(f"Wrote {path}")


def print_summary(runs, agg_by_arm, out_dir):
    """Final-value (last common episode) mean +/- std per arm -> console + summary.csv."""
    header = f"{'arm':<12}{'seeds':>6}{'final_ep':>10}{'health_diff':>22}{'avg_td':>18}{'win_rate':>16}"
    print("\n=== SUMMARY (final eval, mean +/- std across seeds) ===")
    print(header)
    print("-" * len(header))
    summary_rows = []
    for arm in sorted(agg_by_arm):
        rows = agg_by_arm[arm]
        if not rows:
            continue
        last = rows[-1]
        n = last["n_seeds"]
        ep = last["episode"]
        hd = f"{last['health_diff_mean']:+.3f} +/- {last['health_diff_std']:.3f}"
        td = f"{last['avg_td_mean']:.3f} +/- {last['avg_td_std']:.3f}"
        wr = f"{last['win_rate_mean']*100:4.0f}% +/- {last['win_rate_std']*100:3.0f}"
        print(f"{arm:<12}{n:>6}{ep:>10}{hd:>22}{td:>18}{wr:>16}")
        summary_rows.append({
            "arm": arm, "n_seeds": n, "final_episode": ep,
            "health_diff_mean": last["health_diff_mean"], "health_diff_std": last["health_diff_std"],
            "avg_td_mean": last["avg_td_mean"], "avg_td_std": last["avg_td_std"],
            "win_rate_mean": last["win_rate_mean"], "win_rate_std": last["win_rate_std"],
            "dmg_ratio_mean": last["dmg_ratio_mean"], "dmg_ratio_std": last["dmg_ratio_std"],
        })
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {path}")


def print_console_curve(agg_by_arm, metric):
    """A tiny text learning-curve: metric_mean per arm at each shared episode (no deps)."""
    print(f"\n=== {metric} learning curve (mean across seeds) ===")
    arms = sorted(agg_by_arm)
    print("episode".ljust(10) + "".join(a[:11].ljust(12) for a in arms))
    # union of episodes, arms may differ slightly in length
    eps = sorted({r["episode"] for a in arms for r in agg_by_arm[a]})
    lut = {a: {r["episode"]: r for r in agg_by_arm[a]} for a in arms}
    for ep in eps:
        cells = []
        for a in arms:
            r = lut[a].get(ep)
            cells.append(("--".ljust(12)) if r is None else f"{r[f'{metric}_mean']:+.3f}".ljust(12))
        print(str(ep).ljust(10) + "".join(cells))


def maybe_plot(agg_by_arm, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib not available -> skipping PNG plots; agg_*.csv still written)")
        return
    for metric, ylabel in (("health_diff", "health diff (mean +/- std)"),
                           ("avg_td", "avg |TD error| (mean +/- std)")):
        plt.figure(figsize=(8, 5))
        for arm in sorted(agg_by_arm):
            rows = agg_by_arm[arm]
            if not rows:
                continue
            xs = [r["episode"] for r in rows]
            ms = [r[f"{metric}_mean"] for r in rows]
            ss = [r[f"{metric}_std"] for r in rows]
            lo = [m - s for m, s in zip(ms, ss)]
            hi = [m + s for m, s in zip(ms, ss)]
            line, = plt.plot(xs, ms, label=arm)
            plt.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
        plt.xlabel("training episode")
        plt.ylabel(ylabel)
        plt.title(f"{metric} vs training episode (per opponent arm)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"{metric}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs", help="directory holding *_metrics.csv")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <runs-dir>/analysis)")
    ap.add_argument("--metric", default="health_diff", choices=METRIC_COLS,
                    help="metric for the console text curve")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.runs_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.runs_dir)
    print(f"Loaded arms: " + ", ".join(f"{a}({len(runs[a])} seeds)" for a in sorted(runs)))

    agg_by_arm = {}
    for arm, seed_map in runs.items():
        rows = aggregate_arm(seed_map)
        agg_by_arm[arm] = rows
        write_agg_csv(os.path.join(out_dir, f"agg_{arm}.csv"), rows)

    write_combined_csv(os.path.join(out_dir, "combined.csv"), agg_by_arm)
    print_summary(runs, agg_by_arm, out_dir)
    print_console_curve(agg_by_arm, args.metric)
    maybe_plot(agg_by_arm, out_dir)
    print(f"\nDone. Aggregates + summary in: {out_dir}")


if __name__ == "__main__":
    main()
