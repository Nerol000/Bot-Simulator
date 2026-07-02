"""Double-DQN agent for the duel env: an MLP Q-network with a target network,
experience replay, and epsilon-greedy exploration. CPU-friendly."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),
            torch.as_tensor(a, dtype=torch.int64).unsqueeze(1),
            torch.as_tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.as_tensor(np.array(s2), dtype=torch.float32),
            torch.as_tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim, num_actions, *,
                 lr=1e-3, gamma=0.99, buffer_size=100_000, batch_size=128,
                 target_sync=1000, eps_start=1.0, eps_end=0.05, eps_decay=0.9995,
                 device=None):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(obs_dim, num_actions).to(self.device)
        self.target = QNetwork(obs_dim, num_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    def act(self, obs, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(t).argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(self.batch_size)
        s, a, r, s2, d = (x.to(self.device) for x in (s, a, r, s2, d))

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            # Double DQN: action chosen by online net, valued by target net.
            next_a = self.q(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, next_a)
            target = r + self.gamma * q_next * (1.0 - d)
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"q": self.q.state_dict(), "eps": self.eps,
                    "learn_steps": self.learn_steps}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["q"])
        self.eps = ckpt.get("eps", self.eps_end)
        self.learn_steps = ckpt.get("learn_steps", 0)