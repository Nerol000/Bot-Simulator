"""Tabular Q-learning learner (Python port of RL/Simulator's QTable.java + Agent.java).

Faithful to the Java version's update rule and CSV format, but generalized so it plugs into
the shared experiment loop:
  - state space  = the env's 24-cell discrete index (Environment.state_index)
  - action space = the env's 15-action space (environment.NUM_ACTIONS)
  - learn() returns |TD error| so H1 can log it per step.

The Java learner used a fixed epsilon=0.2; here epsilon decays (start->end) like the DQN
so both architectures share the same exploration schedule for a fair comparison. Set
eps_start == eps_end to recover the fixed-epsilon Java behavior.
"""

import random

from environment import NUM_ACTIONS  # 15-action space, shared with the neural learner


class QTable:
    """A [num_states][num_actions] table with the Java save/load CSV format.

    CSV: first line "numStates,numActions", then one row of comma-separated doubles per state.
    Matches RL/Simulator QTable.save()/load() and the mod's QTableLoader, so a trained table
    is loadable by both.
    """

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.q = [[0.0] * num_actions for _ in range(num_states)]

    def best_action(self, s):
        row = self.q[s]
        best_a, best_q = 0, row[0]
        for a in range(1, self.num_actions):
            if row[a] > best_q:
                best_q, best_a = row[a], a
        return best_a

    def max_q(self, s):
        return max(self.q[s])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self.num_states},{self.num_actions}\n")
            for s in range(self.num_states):
                f.write(",".join(repr(v) for v in self.q[s]) + "\n")

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            file_states, file_actions = int(header[0]), int(header[1])
            if file_states != self.num_states or file_actions != self.num_actions:
                raise ValueError(
                    f"QTable dimension mismatch: file {file_states}x{file_actions}, "
                    f"expected {self.num_states}x{self.num_actions}")
            for s in range(self.num_states):
                cells = f.readline().strip().split(",")
                self.q[s] = [float(c) for c in cells]


class TabularQLearner:
    """Q-learning agent with the shared learner API. Update rule mirrors Agent.learn():
    Q(s,a) += lr * (r + gamma * max_a' Q(s',a') * (1-done) - Q(s,a)).
    """

    def __init__(self, num_states, num_actions=NUM_ACTIONS, *,
                 lr=0.1, gamma=0.9, eps_start=1.0, eps_end=0.05, eps_decay=0.9995,
                 seed=0):
        self.table = QTable(num_states, num_actions)
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = random.Random(seed)

    def act(self, s, greedy=False):
        if (not greedy) and self.rng.random() < self.eps:
            return self.rng.randrange(self.num_actions)
        return self.table.best_action(s)

    def observe(self, s, a, r, s2, done):
        """One Q-update. Returns |TD error| (the H1 secondary metric)."""
        target = r + (0.0 if done else self.gamma * self.table.max_q(s2))
        td = target - self.table.q[s][a]
        self.table.q[s][a] += self.lr * td
        return abs(td)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path):
        self.table.save(path)

    def load(self, path):
        self.table.load(path)
