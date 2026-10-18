"""One-off: measure a saved Q-table's true greedy win-rate vs the scripted w-tap opponent.

    python eval_table.py <table.csv> [episodes]

Loads the table, runs `evaluate()` (the same greedy rollout the trainer uses), and prints the
win-rate / health-diff / damage-ratio. Use this as the deploy sanity check -- it's the ground
truth for a specific file, unbiased by the noisy per-checkpoint evals during training.
"""
import sys

from environment import DuelEnv, NUM_ACTIONS
from core.tabular import TabularQLearner
from train_tabular import evaluate

path = sys.argv[1]
episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200

env = DuelEnv(max_steps=1200, seed=123, time_penalty=0.0, reward_mode="discrete")
learner = TabularQLearner(env.NUM_STATES, NUM_ACTIONS, seed=123)
learner.load(path)
m = evaluate(env, learner, episodes=episodes)
print(f"{path}")
print(f"  win={m['win_rate']:.0%}  hdiff={m['health_diff']:+.3f}  "
      f"dmg-ratio={m['dmg_ratio']:.2f}  ttk={m['avg_steps']:.0f}t  ({episodes} eps)")
