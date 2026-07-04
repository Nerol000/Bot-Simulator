"""Shared experiment core for the Minecraft-PvP RL trainers.

One physics environment (../environment.py), pluggable opponents, and interchangeable
learners (tabular Q + neural DQN) exposing the same tiny API so a single experiment loop
runs both. This is the substrate for the H1 (TD-error vs win-max teaching) and H2
(teacher vs champion FSM) studies, and for next year's evolutionary FSM search.

Learner API (both TabularQLearner and the DQN wrapper honor it):
    act(state, greedy=False) -> int          # choose an action index
    observe(s, a, r, s2, done) -> float      # learn from one transition, return |TD error|
    decay_epsilon() -> None
    save(path) / load(path)
"""

from .tabular import QTable, TabularQLearner
from .opponents import Opponent, ParameterizedFSM

__all__ = ["QTable", "TabularQLearner", "Opponent", "ParameterizedFSM"]
