"""Train the tabular Q-learner against a chosen FSM opponent (Python port of Main.java's loop).

    python train_tabular.py --opponent champion     # vs win-seeking FSM
    python train_tabular.py --opponent teacher       # vs teaching FSM
    python train_tabular.py --eval                   # greedy win-rate vs scripted s-tap

Progress is measured the same way as the neural trainer: greedy rollouts vs the fixed scripted
s-tap opponent (env.step_eval), so tabular and neural numbers are directly comparable. The
per-step |TD error| is logged too (the H1 secondary metric). Checkpoint: ./qtable.csv.
"""

import argparse
import time

import numpy as np

from environment import DuelEnv, NUM_ACTIONS
from core.tabular import TabularQLearner
from core.opponents import ParameterizedFSM

CKPT = "qtable.csv"

OPPONENTS = {
    "champion": ParameterizedFSM.champion,
    "teacher": ParameterizedFSM.teacher,
    "win_max": ParameterizedFSM.win_max,
    "td_error": ParameterizedFSM.td_error,
}


def evaluate(env, learner, episodes=50):
    """Greedy rollout of the learner vs the scripted s-tap opponent (fixed yardstick)."""
    wins = 0
    total_dealt = total_taken = 0.0
    total_steps = 0
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
    n = episodes
    return {
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
    args = ap.parse_args()

    # time_penalty=0.0: the 24x15 table can't out-run the per-tick drain (it sinks learned cells
    # below unvisited 0.0 cells and inverts the greedy policy). Matches the Java tabular trainer.
    env = DuelEnv(max_steps=args.max_steps, seed=args.seed, time_penalty=0.0)
    learner = TabularQLearner(env.NUM_STATES, NUM_ACTIONS, seed=args.seed)

    if args.eval:
        learner.load(CKPT)
        m = evaluate(env, learner, episodes=50)
        print(f"Eval vs s-tap (50): win={m['win_rate']:.0%}  dmg-ratio={m['dmg_ratio']:.2f}  "
              f"ttk={m['avg_steps']:.0f}t")
        return

    opponent = OPPONENTS[args.opponent](seed=args.seed)
    best_win = -1.0
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

        if (ep + 1) % args.log_every == 0:
            avg_td = np.mean(recent_td[-5000:])
            sps = (ep + 1) / (time.time() - t0)
            print(f"ep {ep+1:6d}  avg_td={avg_td:7.3f}  eps={learner.eps:.3f}  {sps:.1f} ep/s")

        if (ep + 1) % args.eval_every == 0:
            m = evaluate(env, learner)
            print(f"  [eval vs s-tap] win={m['win_rate']:.0%}  dmg-ratio={m['dmg_ratio']:.2f}  "
                  f"ttk={m['avg_steps']:.0f}t")
            learner.save(CKPT)
            if m["win_rate"] > best_win:
                best_win = m["win_rate"]
                learner.save(CKPT.replace(".csv", "_best.csv"))
                print(f"  new best win-rate {m['win_rate']:.0%}")

    learner.save(CKPT)
    print(f"Done. Saved {CKPT}")


if __name__ == "__main__":
    main()
