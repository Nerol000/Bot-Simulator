"""Train the Double-DQN agent on the duel env.

    python train.py                 # train from scratch
    python train.py --resume        # continue from the last checkpoint
    python train.py --eval          # greedy rollout for a few episodes (no learning)

Checkpoints are written to ./checkpoints/qnet.pt (and qnet_best.pt for the best win-rate).
"""

import argparse
import os
import time

import numpy as np

from environment import DuelEnv, OBS_DIM, NUM_ACTIONS
from dqn import DQNAgent

CKPT_DIR = "checkpoints"
CKPT = os.path.join(CKPT_DIR, "qnet.pt")
CKPT_BEST = os.path.join(CKPT_DIR, "qnet_best.pt")


def evaluate(env, agent, episodes=20):
    wins = 0
    total = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, done, _ = env.step(a)
            ep_r += r
        total += ep_r
        if env.bot2.health <= 0 < env.bot1.health:
            wins += 1
    return wins / episodes, total / episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    env = DuelEnv(max_steps=args.max_steps, seed=args.seed)
    agent = DQNAgent(OBS_DIM, NUM_ACTIONS)

    if (args.resume or args.eval) and os.path.exists(CKPT):
        agent.load(CKPT)
        print(f"Loaded checkpoint {CKPT} (eps={agent.eps:.3f})")

    if args.eval:
        win, avg = evaluate(env, agent, episodes=50)
        print(f"Eval: win-rate={win:.2%}  avg-reward={avg:.2f}")
        return

    best_win = -1.0
    recent_rewards = []
    t0 = time.time()
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.act(obs)
            next_obs, r, done, _ = env.step(a)
            agent.remember(obs, a, r, next_obs, float(done))
            agent.learn()
            obs = next_obs
            ep_r += r
        agent.decay_epsilon()
        recent_rewards.append(ep_r)

        if (ep + 1) % args.log_every == 0:
            avg = np.mean(recent_rewards[-args.log_every:])
            sps = (ep + 1) / (time.time() - t0)
            print(f"ep {ep+1:6d}  avg_reward={avg:8.2f}  eps={agent.eps:.3f}  {sps:.1f} ep/s")

        if (ep + 1) % args.eval_every == 0:
            win, avg = evaluate(env, agent)
            print(f"  [eval] win-rate={win:.2%}  avg-reward={avg:.2f}")
            agent.save(CKPT)
            if win > best_win:
                best_win = win
                agent.save(CKPT_BEST)
                print(f"  new best win-rate {win:.2%} -> {CKPT_BEST}")

    agent.save(CKPT)
    print(f"Done. Saved {CKPT}")


if __name__ == "__main__":
    main()
