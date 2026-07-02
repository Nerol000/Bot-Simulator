"""Export a trained Q-network to a plain JSON the Java mod can load (no ONNX / native deps).

    python export_weights.py [checkpoint] [out.json]

Defaults: checkpoints/qnet_best.pt -> policy.json. Copy the result to the mod at
RL/Bot/src/main/resources/assets/pvp_bot/policy.json (the mod loads it via NeuralPolicy).

Format:
    {"obs_dim": 13, "actions": 14,
     "layers": [{"w": [[...in], ...out], "b": [...out]}, ...]}   # ReLU between layers, none after last
"""

import json
import sys

import torch

from dqn import QNetwork
from environment import OBS_DIM, NUM_ACTIONS


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/qnet_best.pt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "policy.json"

    net = QNetwork(OBS_DIM, NUM_ACTIONS)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu")["q"])
    net.eval()

    layers = [
        {"w": m.weight.detach().tolist(), "b": m.bias.detach().tolist()}
        for m in net.net if isinstance(m, torch.nn.Linear)
    ]
    with open(out_path, "w") as f:
        json.dump({"obs_dim": OBS_DIM, "actions": NUM_ACTIONS, "layers": layers}, f)

    print(f"Wrote {out_path}  ({len(layers)} linear layers, obs_dim={OBS_DIM}, actions={NUM_ACTIONS})")
    print("Copy it to: RL/Bot/src/main/resources/assets/pvp_bot/policy.json")


if __name__ == "__main__":
    main()