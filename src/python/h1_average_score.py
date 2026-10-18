"""
H1 check, high-school version: "average score during training".

The regular summary only looks at the LAST eval. But the whole point of H1 is which
training partner makes a BETTER STUDENT while it is learning. So here we do something
simple: for each training run we take the AVERAGE of its HD-EMA score across all the
eval checkpoints (that is just "how well did this student fight, on average, during
training"). Then we compare the arms.

HD-EMA is the smoothed health-difference score:
   +1 = perfect win, 0 = even fight, -1 = destroyed.

We compute, per arm:
   - the average score for each seed (one number per training run)
   - the mean of those seed numbers, and the standard error (how much they wobble)
Then a simple t-test tells us if two arms are really different or just noise.

Run:
   python h1_average_score.py --results-dir runs\\2026-09-13_23-36-31
"""

import argparse
import csv
import glob
import math
import os
import re

# file name looks like:  champion_s0_ep50000_metrics.csv
NAME_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)_ep\d+_metrics\.csv$")

# nice names for the chart/print-out
ARM_LABEL = {
    "champion": "Champion (Win-Max)",
    "teacher": "TD-Max",
    "td_error": "TD-Max",
    "improve": "Improvement",
    "win_max": "Win-Max",
    "selfplay": "Self-Play",
}


def average_score_per_seed(results_dir, metric="hdiff_ema"):
    """For each arm, return a list with one 'average score' number per seed."""
    per_arm = {}
    for path in glob.glob(os.path.join(results_dir, "*_metrics.csv")):
        name = os.path.basename(path)
        m = NAME_RE.match(name)
        if not m:
            continue
        arm = m.group("arm")
        # read every eval row's score, then average them = this run's "average score"
        scores = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores.append(float(row[metric]))
        if scores:
            per_arm.setdefault(arm, []).append(sum(scores) / len(scores))
    return per_arm


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# --- simple two-group t-test (Welch). Same math analyze.py already uses. ---
def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for mm in range(1, itmax + 1):
        m2 = 2 * mm
        aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lb + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def ttest(a, b):
    """Return (t, df, p) comparing two lists of numbers. p < 0.05 = 'really different'."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = std(a) ** 2, std(b) ** 2
    sa, sb = va / na, vb / nb
    denom = sa + sb
    if denom <= 0:
        return float("nan"), float("nan"), float("nan")
    t = (mean(a) - mean(b)) / math.sqrt(denom)
    df = denom ** 2 / ((sa ** 2) / (na - 1) + (sb ** 2) / (nb - 1))
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return t, df, p


def label(arm):
    return ARM_LABEL.get(arm, arm)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results-dir", default=os.path.join(here, "runs"),
                    help="folder with the *_metrics.csv files")
    ap.add_argument("--metric", default="hdiff_ema",
                    help="which score column to average (default hdiff_ema)")
    args = ap.parse_args()

    per_arm = average_score_per_seed(args.results_dir, args.metric)
    if not per_arm:
        raise SystemExit(f"No *_metrics.csv found in {args.results_dir!r}")

    # print one line per arm: average score across seeds, +/- standard error
    print(f"\nAverage score during training ({args.metric}), across seeds:")
    print("-" * 55)
    order = [a for a in ["champion", "win_max", "teacher", "td_error", "improve", "selfplay"]
             if a in per_arm] + [a for a in per_arm if a not in ARM_LABEL]
    for arm in order:
        vals = per_arm[arm]
        se = std(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {label(arm):22s}  {mean(vals):+.3f}  +/- {se:.3f}   (n={len(vals)} seeds)")

    # the H1 comparisons we care about
    print("\nComparisons (is the difference real, or just noise?):")
    print("-" * 55)
    pairs = [("teacher", "champion"), ("improve", "champion"), ("improve", "teacher")]
    for a, b in pairs:
        if a in per_arm and b in per_arm:
            t, df, p = ttest(per_arm[a], per_arm[b])
            delta = mean(per_arm[a]) - mean(per_arm[b])
            verdict = "YES, really different (p<0.05)" if p < 0.05 else "no, could be noise"
            print(f"  {label(a):14s} vs {label(b):16s}  delta={delta:+.3f}  "
                  f"t={t:.2f}  df={df:.1f}  p={p:.4f}  -> {verdict}")
    print("\n(Higher average score = better student. H1 expects TD-Max > Champion.)\n")


if __name__ == "__main__":
    main()
