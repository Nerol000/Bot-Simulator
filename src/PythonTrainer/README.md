# Python neural-net trainer (Double DQN, 360° yaw + pitch)

A neural-network alternative to the Java tabular trainer in `RL/Simulator`. It trains a
Double-DQN agent against the same s-tap opponent, but with a **continuous observation** and a
**full 360° yaw + pitch** action space (instead of the Java sim's 45°/90° turns and fixed pitch).

## Files
- `environment.py` — Python port of `RL/Simulator/bot_simulator/Environment.java` (movement,
  knockback, the eye→look **raytrace** attack gate now in 3D, bot2's s-tap AI, rewards). Gym-like
  `reset()` / `step(action)`.
- `dqn.py` — `QNetwork` (MLP), `ReplayBuffer`, and the `DQNAgent` (Double DQN, target net,
  epsilon-greedy).
- `train.py` — training loop, evaluation, checkpointing/resume.

## Setup & run
```bash
cd RL/PythonTrainer
pip install -r requirements.txt
python train.py                 # train (checkpoints -> ./checkpoints/qnet.pt)
python train.py --resume        # continue training
python train.py --eval          # greedy win-rate over 50 episodes
```

## Action space (14 discrete actions)
`idle, forward, sprint_forward, back, strafe_left, strafe_right, attack, jump,`
`yaw±15°, yaw±4°, pitch±8°`. Small turn deltas compose to **any** yaw (0–360) and pitch
(−90…90); the agent learns precise aim from the continuous state, so attacks (raytrace-gated)
require it to actually point at the target in 3D.

## Observation (13 floats)
`[dist/10, dy, sin/cos(rel_yaw), sin/cos(rel_pitch_err), vx, vz, attack_charge,
 self_hp, target_hp, on_ground, can_hit_now]`. Angles are sin/cos encoded so aim has no
wrap-around discontinuity.

## Reward
Damage dealt (+), damage taken (−0.75/HP), closing distance (+), **reducing aim error** (+, this
drives the 360°+pitch aiming), a small penalty for swinging when it can't connect, and ±20 on
kill/death.

## Fidelity note
The physics constants and the bot2 s-tap logic mirror `Environment.java`. The armor/Protection
damage reduction is approximated by a single `PROTECTION_MULT` constant (the Java sim derives it
from the `Equipment`); tune it in `environment.py` if duels feel too long/short.

## Deploying to the Minecraft mod
The Java mod (`RL/Bot`) currently loads a **tabular `qtable.csv`** via `LiveController`. This
trainer produces a **PyTorch network**, not a table, so deployment needs an inference bridge —
either export the net to ONNX/TorchScript and run it in Java, or add a small head that
re-discretizes/look-controls. That bridge isn't built yet; ask and I'll wire it up.
