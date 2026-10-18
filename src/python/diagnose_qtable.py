"""Ad-hoc diagnostic: decode a tabular Q-table's greedy policy per state and flag pathologies.

Prints, for each of the 48 states (charged x distance x direction), the greedy action, its Q,
and how many action-cells are ~unvisited (|q| < eps). Highlights the states that matter most for
"approach / attack" behavior so we can see WHY a table only retreats.

    python diagnose_qtable.py <table.csv> [<table2.csv> ...]
"""
import sys

ACTIONS = ["IDLE", "FORWARD", "SPRINT_FWD", "BACK", "STRAFE_L", "STRAFE_R",
           "ATTACK", "JUMP", "YAW_L", "YAW_R", "YAW_L_FINE", "YAW_R_FINE",
           "PITCH_UP", "PITCH_DOWN", "LOOK_AT"]
DIST = {0: "NEAR", 1: "MID", 2: "FAR"}
UNVISITED_EPS = 1e-3


def load(path):
    with open(path, encoding="utf-8") as f:
        ns, na = (int(x) for x in f.readline().split(","))
        return [[float(c) for c in f.readline().split(",")] for _ in range(ns)], ns, na


def decode(s):
    charged, rem = divmod(s, 24)
    dist, direction = divmod(rem, 8)
    return charged, dist, direction


def analyze(path):
    q, ns, na = load(path)
    print(f"\n=== {path}  ({ns} states x {na} actions) ===")
    total = sum(len(r) for r in q)
    neg = sum(1 for r in q for v in r if v < 0)
    unvisited = sum(1 for r in q for v in r if abs(v) < UNVISITED_EPS)
    print(f"cells: {total}  negative: {neg} ({neg/total:.0%})  "
          f"~unvisited(|q|<{UNVISITED_EPS}): {unvisited} ({unvisited/total:.0%})")

    # category tallies of the greedy action across all states
    from collections import Counter
    cat = Counter()
    retreat_states = []
    for s in range(ns):
        row = q[s]
        a = max(range(na), key=lambda i: row[i])
        cat[ACTIONS[a]] += 1
        charged, dist, direction = decode(s)
        if ACTIONS[a] in ("BACK",):
            retreat_states.append((s, charged, dist, direction, row[a]))
    print("greedy action distribution across 48 states:",
          dict(sorted(cat.items(), key=lambda kv: -kv[1])))

    # the key combat states: facing target (direction 0), charged
    print("key states (direction=FRONT):")
    for charged in (0, 1):
        for dist in (2, 1, 0):
            s = charged * 24 + dist * 8 + 0
            row = q[s]
            a = max(range(na), key=lambda i: row[i])
            best_attack = row[6]
            print(f"  s{s:2d} charged={charged} {DIST[dist]:4s} FRONT -> "
                  f"{ACTIONS[a]:11s} q={row[a]:+.4f}   ATTACK q={best_attack:+.4f}")
    if retreat_states:
        print(f"states whose greedy action is BACK (retreat): {len(retreat_states)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
