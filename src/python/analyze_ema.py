"""One-off: re-run the H1/H2 summary + Welch t-tests on the SMOOTHED metric
(hdiff_ema) instead of the single noisy final eval. Reuses analyze.py."""
import sys
import math
import analyze as A


def per_seed(runs, metric):
    out = {}
    for arm, seed_map in runs.items():
        seeds = list(seed_map)
        if not seeds:
            continue
        common = set.intersection(*(set(seed_map[s]) for s in seeds))
        if not common:
            continue
        last_ep = max(common)
        out[arm] = [seed_map[s][last_ep][metric] for s in seeds]
    return out


def summary(runs, metric):
    vals = per_seed(runs, metric)
    print(f"\n=== SUMMARY ({metric}, final episode, mean +/- std across seeds) ===")
    print(f"{'arm':<12}{'n':>4}{'mean':>12}{'std':>10}")
    print("-" * 38)
    for arm in sorted(vals):
        xs = vals[arm]
        print(f"{arm:<12}{len(xs):>4}{A._mean(xs):>+12.4f}{A._std(xs):>10.4f}")
    return vals


def ttests(vals, metric):
    surprise = "teacher" if "teacher" in vals else ("td_error" if "td_error" in vals else None)
    pairs = []
    if "improve" in vals and "champion" in vals:
        pairs.append(("H2 core: improve vs champion", "improve", "champion"))
    if "improve" in vals and surprise:
        pairs.append(("proxy gap: improve vs " + surprise, "improve", surprise))
    if surprise and "champion" in vals:
        pairs.append(("orig H2: " + surprise + " vs champion", surprise, "champion"))
    print(f"\n=== SIGNIFICANCE (Welch's t-test on final {metric}) ===")
    hdr = f"{'comparison':<34}{'mean_a':>9}{'mean_b':>9}{'t':>8}{'df':>7}{'p':>10}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for label, a_name, b_name in pairs:
        a, b = vals[a_name], vals[b_name]
        t, df, p = A.welch_ttest(a, b)
        if math.isnan(p):
            v = "need >=2 seeds/arm"
        elif p < 0.01:
            v = "significant (p<0.01)"
        elif p < 0.05:
            v = "significant (p<0.05)"
        else:
            v = "NOT significant (trend only)"
        print(f"{label:<34}{A._mean(a):>+9.3f}{A._mean(b):>+9.3f}{t:>8.2f}{df:>7.1f}{p:>10.4f}  {v}")


if __name__ == "__main__":
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else r"..\2026-09-06_22-06-55"
    runs = A.load_runs(runs_dir)
    print("Loaded arms: " + ", ".join(f"{a}({len(s)} seeds)" for a, s in runs.items()))
    for metric in ("hdiff_ema", "hdiff_best", "avg_td"):
        vals = summary(runs, metric)
        ttests(vals, metric)
