"""Train the Double-DQN agent by SELF-PLAY on the duel env.

Both bots are driven by the same (shared) policy and both of their transitions go into one replay
buffer, so the agent improves by playing itself — the opponent is always at its own skill level, so
there's always a learnable gradient (no cold-start wall like training vs the s-tap expert).

Progress is *measured* against the fixed scripted s-tap opponent via step_eval().

    python train.py                 # self-play training from scratch
    python train.py --resume        # continue from the last checkpoint
    python train.py --eval          # greedy rollout vs the scripted opponent (no learning)

Checkpoints: ./checkpoints/qnet.pt (latest) and qnet_best.pt (best eval win-rate).
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
    """Greedy rollout of bot1 against the scripted s-tap opponent (a fixed yardstick), with graded
    metrics so even lopsided results stay informative."""
    wins = 0
    total_reward = total_dealt = total_taken = win_hp = 0.0
    total_steps = 0
    for _ in range(episodes):
        obs1, _ = env.reset()
        done = False
        ep_r = dealt = taken = 0.0
        steps = 0
        while not done:
            a = agent.act(obs1, greedy=True)
            obs1, r, done, info = env.step_eval(a)
            ep_r += r
            dealt += info["dmg_dealt"]
            taken += info["dmg_taken"]
            steps += 1
        total_reward += ep_r
        total_dealt += dealt
        total_taken += taken
        total_steps += steps
        if env.bot2.health <= 0 < env.bot1.health:
            wins += 1
            win_hp += env.bot1.health
    n = episodes
    return {
        "win_rate": wins / n,
        "avg_reward": total_reward / n,
        "dmg_ratio": total_dealt / max(total_taken, 1e-6),   # >1 = out-damaging the opponent
        "avg_win_hp": (win_hp / wins) if wins else 0.0,       # margin of victory, /20
        "avg_steps": total_steps / n,                         # time-to-kill (lower = more decisive)
    }


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
        m = evaluate(env, agent, episodes=50)
        print(f"Eval vs s-tap (50): win={m['win_rate']:.0%}  dmg-ratio={m['dmg_ratio']:.2f}  "
              f"win-hp={m['avg_win_hp']:.1f}/20  ttk={m['avg_steps']:.0f}t  reward={m['avg_reward']:.1f}")
        return

    best_win = -1.0
    recent_rewards = []
    t0 = time.time()
    for ep in range(args.episodes):
        obs1, obs2 = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a1 = agent.act(obs1)
            a2 = agent.act(obs2)
            n1, n2, r1, r2, done, _ = env.step(a1, a2)
            # Pool BOTH bots' experience into the shared buffer -> both perspectives improve one net.
            agent.remember(obs1, a1, r1, n1, float(done))
            agent.remember(obs2, a2, r2, n2, float(done))
            agent.learn()
            obs1, obs2 = n1, n2
            ep_r += r1
        agent.decay_epsilon()
        recent_rewards.append(ep_r)

        if (ep + 1) % args.log_every == 0:
            avg = np.mean(recent_rewards[-args.log_every:])
            sps = (ep + 1) / (time.time() - t0)
            # NOTE: this is bot1's *self-play* reward (≈0 in a balanced match) — the real progress
            # signal is the periodic [eval] line below, not this.
            print(f"ep {ep+1:6d}  selfplay_reward={avg:7.2f}  eps={agent.eps:.3f}  {sps:.1f} ep/s")

        if (ep + 1) % args.eval_every == 0:
            m = evaluate(env, agent)
            print(f"  [eval vs s-tap] win={m['win_rate']:.0%}  dmg-ratio={m['dmg_ratio']:.2f}  "
                  f"win-hp={m['avg_win_hp']:.1f}/20  ttk={m['avg_steps']:.0f}t")
            agent.save(CKPT)
            if m["win_rate"] > best_win:
                best_win = m["win_rate"]
                agent.save(CKPT_BEST)
                print(f"  new best win-rate {m['win_rate']:.0%} -> {CKPT_BEST}")

    agent.save(CKPT)
    print(f"Done. Saved {CKPT}")


if __name__ == "__main__":
    main()
