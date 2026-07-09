"""Train the tabular Q-learner against a chosen FSM opponent (Python port of Main.java's loop).

    python train_tabular.py --opponent champion     # vs win-seeking FSM
    python train_tabular.py --opponent teacher       # vs teaching FSM
    python train_tabular.py --opponent selfplay      # vs frozen snapshots of itself (self-play)
    python train_tabular.py --eval                   # greedy win-rate vs scripted w-tap

Progress is measured the same way as the neural trainer: greedy rollouts vs the fixed scripted
w-tap opponent (env.step_eval), so tabular and neural numbers are directly comparable. The
per-step |TD error| is logged too (the H1 secondary metric). Checkpoint: ./qtable.csv.
"""

import argparse
import os
import time
from collections import deque

import numpy as np

from environment import DuelEnv, NUM_ACTIONS, MAX_HEALTH
from core.tabular import TabularQLearner
from core.opponents import ParameterizedFSM, SnapshotOpponent

CKPT = "qtable.csv"

OPPONENTS = {
    "champion": ParameterizedFSM.champion,
    "teacher": ParameterizedFSM.teacher,
    "win_max": ParameterizedFSM.win_max,
    "td_error": ParameterizedFSM.td_error,
    "selfplay": SnapshotOpponent,
}


def evaluate(env, learner, episodes=50):
    """Greedy rollout of the learner vs the scripted w-tap opponent (fixed yardstick).

    Primary metric is `health_diff` (mean end-of-episode (my_health - opp_health)/MAX_HEALTH,
    range [-1, +1]): a CONTINUOUS skill measure that stays informative at every level, unlike
    win_rate which is 0 until the bot crosses a competence cliff then jumps. -1 = destroyed
    dealing nothing, 0 = even trade, +1 = flawless win. This is the H1 learning-curve metric."""
    wins = 0
    total_dealt = total_taken = 0.0
    total_steps = 0
    total_hdiff = 0.0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            s = env.state_index(env.bot1, env.bot2)
            a = learner.act(s, greedy=True)
            _, _, done, info = env.step_eval(a)
            total_dealt += info["dmg_dealt"]
            total_taken += info["dmg_taken"]
            total_steps += 1
        if env.bot2.health <= 0 < env.bot1.health:
            wins += 1
        total_hdiff += (max(0.0, env.bot1.health) - max(0.0, env.bot2.health)) / MAX_HEALTH
    n = episodes
    return {
        "health_diff": total_hdiff / n,
        "win_rate": wins / n,
        "dmg_ratio": total_dealt / max(total_taken, 1e-6),
        "avg_steps": total_steps / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", choices=list(OPPONENTS), default="champion")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    # discrete (default) = Java-style bucket reward that matches the 24-state resolution, so the
    # tabular bot can actually learn to attack. continuous = the fine-grained neural reward
    # (kept for comparison / ablation; the coarse table struggles to exploit it).
    ap.add_argument("--reward", choices=["discrete", "continuous"], default="discrete")
    # only affects --reward continuous; disable to stop over-punishing coarse-bucket aiming.
    ap.add_argument("--miss-penalty", type=float, default=None,
                    help="override continuous-reward MISS_PENALTY (e.g. 0 to disable)")
    # --opponent selfplay only: how often to freeze the learner into the snapshot pool, and how
    # many past selves to keep. The opponent samples one snapshot per episode from this pool.
    ap.add_argument("--snapshot-every", type=int, default=250,
                    help="[selfplay] episodes between freezing the learner into the snapshot pool")
    ap.add_argument("--pool-size", type=int, default=5,
                    help="[selfplay] number of past-self snapshots to keep and sample from")
    # Output naming: each run writes distinct files so a seed/opponent sweep never clobbers
    # itself. Default tag encodes the params -> e.g. runs/teacher_s3_ep20000_best.csv.
    ap.add_argument("--out-dir", default="runs",
                    help="directory for checkpoints and the metrics CSV")
    ap.add_argument("--tag", default=None,
                    help="filename stem for this run (default: <opponent>_s<seed>_ep<episodes>)")
    args = ap.parse_args()

    # time_penalty=0.0: the 24x15 table can't out-run the per-tick drain (it sinks learned cells
    # below unvisited 0.0 cells and inverts the greedy policy). Matches the Java tabular trainer.
    env_kwargs = dict(max_steps=args.max_steps, seed=args.seed, time_penalty=0.0,
                      reward_mode=args.reward)
    if args.miss_penalty is not None:
        env_kwargs["miss_penalty"] = args.miss_penalty
    env = DuelEnv(**env_kwargs)
    learner = TabularQLearner(env.NUM_STATES, NUM_ACTIONS, seed=args.seed)

    if args.eval:
        learner.load(CKPT)
        m = evaluate(env, learner, episodes=50)
        print(f"Eval vs w-tap (50): win={m['win_rate']:.0%}  dmg-ratio={m['dmg_ratio']:.2f}  "
              f"ttk={m['avg_steps']:.0f}t")
        return

    # Per-run output paths (distinct per opponent+seed+episodes so a sweep never overwrites itself).
    tag = args.tag or f"{args.opponent}_s{args.seed}_ep{args.episodes}"
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, f"{tag}.csv")
    ckpt_best = os.path.join(args.out_dir, f"{tag}_best.csv")
    metrics_path = os.path.join(args.out_dir, f"{tag}_metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write("episode,avg_td,health_diff,win_rate,dmg_ratio,avg_steps\n")
    print(f"[run] tag={tag}  opponent={args.opponent}  seed={args.seed}  episodes={args.episodes}")

    opponent = OPPONENTS[args.opponent](seed=args.seed)
    if isinstance(opponent, SnapshotOpponent):
        opponent.pool = deque(maxlen=args.pool_size)
    is_selfplay = isinstance(opponent, SnapshotOpponent)
    best_score = -1e9
    recent_td = []
    t0 = time.time()

    for ep in range(args.episodes):
        env.reset()
        opponent.reset()
        done = False
        while not done:
            s = env.state_index(env.bot1, env.bot2)
            a = learner.act(s)
            _, r, done, _ = env.step_against(a, opponent)
            s2 = env.state_index(env.bot1, env.bot2)
            td = learner.observe(s, a, r, s2, float(done))
            recent_td.append(td)
        learner.decay_epsilon()

        # Self-play: periodically freeze the current learner into the past-self pool.
        if is_selfplay and (ep + 1) % args.snapshot_every == 0:
            opponent.add_snapshot(learner.table)

        if (ep + 1) % args.log_every == 0:
            avg_td = np.mean(recent_td[-5000:])
            sps = (ep + 1) / (time.time() - t0)
            print(f"ep {ep+1:6d}  avg_td={avg_td:7.3f}  eps={learner.eps:.3f}  {sps:.1f} ep/s")

        if (ep + 1) % args.eval_every == 0:
            m = evaluate(env, learner)
            avg_td = float(np.mean(recent_td[-5000:])) if recent_td else 0.0
            print(f"  [eval vs w-tap] hdiff={m['health_diff']:+.3f}  win={m['win_rate']:.0%}  "
                  f"dmg-ratio={m['dmg_ratio']:.2f}  ttk={m['avg_steps']:.0f}t")
            with open(metrics_path, "a", encoding="utf-8") as mf:
                mf.write(f"{ep+1},{avg_td:.6f},{m['health_diff']:.6f},{m['win_rate']:.6f},"
                         f"{m['dmg_ratio']:.6f},{m['avg_steps']:.3f}\n")
            learner.save(ckpt)
            if m["health_diff"] > best_score:
                best_score = m["health_diff"]
                learner.save(ckpt_best)
                print(f"  new best health-diff {m['health_diff']:+.3f}")

    learner.save(ckpt)
    print(f"Done. Saved {ckpt}  (best: {ckpt_best}, metrics: {metrics_path})")


if __name__ == "__main__":
    main()
