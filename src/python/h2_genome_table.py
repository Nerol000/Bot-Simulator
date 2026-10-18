"""
Build the H2 "evolved opponent genome parameters" table (Table 3).

Reads every *_genome.json file in a results folder, groups them by arm
(teacher = TD-Max, improve = Improvement), and for each genome parameter
computes mean +/- standard deviation across the 15 seeds. It then runs a
Welch two-sample t-test (TD-Max vs Improvement) for each parameter.

Usage:
    & $py h2_genome_table.py --results-dir runs\\2026-09-13_23-36-31

Only the numeric parameters are tested. A Bonferroni note is printed so you
can see which rows survive a corrected 0.05 / (number of parameters) threshold.
"""

import argparse
import glob
import json
import math
import os


# ----- tiny stats helpers (same Welch math as h1_average_score.py) -----
def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


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


# ----- parameters to show, in the same order as the old Table 3 -----
# (label shown in the table, key inside genome params)
PARAM_ROWS = [
    ("preferred_distance", "preferred_distance"),
    ("attack_prob", "attack_prob"),
    ("jump_prob", "jump_prob"),
    ("band (spacing tolerance)", "band"),
    ("retreat_prob", "retreat_prob"),
    ("strafe_prob", "strafe_prob"),
    ("pause_prob", "pause_prob"),
    ("stap_ticks (strafe timing)", "stap_ticks"),
]

ARM_LABEL = {"teacher": "TD-Max", "improve": "Improvement"}


def load_params(results_dir, arm):
    """Return dict: param_key -> list of values across seeds for one arm."""
    out = {}
    pattern = os.path.join(results_dir, f"{arm}_s*_genome.json")
    files = sorted(glob.glob(pattern))
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        params = data.get("params", {})
        for _, key in PARAM_ROWS:
            v = params.get(key)
            if isinstance(v, bool) or v is None:
                continue
            out.setdefault(key, []).append(float(v))
    return out, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="folder containing *_genome.json files")
    args = ap.parse_args()

    td, n_td = load_params(args.results_dir, "teacher")
    imp, n_imp = load_params(args.results_dir, "improve")

    if not td or not imp:
        raise SystemExit(f"No genome files found in {args.results_dir!r}")

    n_params = len(PARAM_ROWS)
    bonf = 0.05 / n_params

    print(f"\nEvolved opponent genome parameters "
          f"(TD-Max n={n_td} seeds, Improvement n={n_imp} seeds):")
    print("-" * 92)
    print(f"{'Genome parameter':28s} {'TD-Max':16s} {'Improvement':16s} "
          f"{'t':>6s} {'df':>6s} {'p':>8s}")
    print("-" * 92)

    for pretty, key in PARAM_ROWS:
        a = td.get(key, [])
        b = imp.get(key, [])
        if not a or not b:
            continue
        t, df, p = ttest(a, b)
        td_str = f"{mean(a):.2f} \u00b1 {std(a):.2f}"
        imp_str = f"{mean(b):.2f} \u00b1 {std(b):.2f}"
        star = " *" if p < bonf else ""
        print(f"{pretty:28s} {td_str:16s} {imp_str:16s} "
              f"{t:6.2f} {df:6.1f} {p:8.4f}{star}")

    print("-" * 92)
    print(f"* = significant at Bonferroni-corrected threshold "
          f"0.05 / {n_params} = {bonf:.4f}")
    print("Mean +/- standard deviation across seeds. "
          "Welch two-sample, two-tailed t-test (TD-Max vs Improvement).")


if __name__ == "__main__":
    main()
