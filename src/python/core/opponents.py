"""Pluggable opponents that drive bot2 by returning an action index (0..14) each tick.

The learner and the opponent therefore share the same 15-action space, so an opponent is
symmetric with a learner and can later be swapped for a QTableOpponent (self-play) without
touching the env.

ParameterizedFSM is the single opponent class used by both experiments; a preset is just a
parameter vector (a "genome"), which is exactly what next year's evolutionary / Bayesian
search will optimize:
  - H1: win_max() (aggressive, win-seeking) vs td_error() (varied, challenge-seeking)
  - H2: champion() (optimize win rate) vs teacher() (optimize learner improvement)
win_max/champion and td_error/teacher are aliases so both hypotheses share one implementation.
"""

import math
import random
from abc import ABC, abstractmethod

from environment import (
    REACH, IDLE, BACK, JUMP, STRAFE_LEFT, STRAFE_RIGHT, SPRINT_FORWARD, ATTACK,
    LOOK_AT_TARGET,
)


class Opponent(ABC):
    """Drives bot2 (`me`) against bot1 (`opp`) for one tick by returning an action index."""

    def reset(self):
        """Reset per-episode internal state. Called at the start of each episode."""

    @abstractmethod
    def act(self, env, me, opp) -> int:
        ...

    # --- behavior metrics (H2 FSM-behavior charts). Subclasses may override. ---
    def behavior_summary(self):
        return {}


class ParameterizedFSM(Opponent):
    """Distance-keeping fighter parameterized by the H2 behavior knobs. Aims first (snaps to
    target when mis-aimed), then chooses approach / retreat / strafe / attack / pause based on
    its preferred combat distance and per-tick action frequencies.

    Params:
      preferred_distance : combat distance it tries to hold (blocks)
      band               : tolerance around preferred_distance before it approaches/retreats
      attack_prob        : chance to swing when in reach and aimed
      retreat_prob       : chance to back off when closer than preferred
      strafe_prob        : chance to strafe when in-band
      jump_prob          : chance to jump on any tick
      pause_prob         : chance to begin an idle pause when in-band
      pause_ticks        : length of an idle pause
      aim_tol            : aim error (rad) above which it re-aims instead of acting
    """

    DEFAULTS = dict(
        preferred_distance=2.5, band=0.75, attack_prob=0.8, retreat_prob=0.1,
        strafe_prob=0.15, jump_prob=0.02, pause_prob=0.0, pause_ticks=0, aim_tol=0.25,
    )

    def __init__(self, name="fsm", seed=0, **params):
        self.name = name
        self.params = {**self.DEFAULTS, **params}
        self.rng = random.Random(seed)
        self._pause = 0
        self._counts = {"attack": 0, "retreat": 0, "strafe": 0, "approach": 0, "idle": 0}
        self._ticks = 0

    def reset(self):
        self._pause = 0
        self._counts = {k: 0 for k in self._counts}
        self._ticks = 0

    def act(self, env, me, opp) -> int:
        p = self.params
        self._ticks += 1

        # 1) Aim first: a swing only lands if the look ray clips the target.
        if env._aim_error(me, opp) > p["aim_tol"]:
            return LOOK_AT_TARGET

        # 2) Honor an in-progress pause.
        if self._pause > 0:
            self._pause -= 1
            self._counts["idle"] += 1
            return IDLE

        dist = math.hypot(opp.x - me.x, opp.z - me.z)
        rng = self.rng

        # 3) Occasional jump (movement variety / knockback dodging).
        if rng.random() < p["jump_prob"]:
            return JUMP

        # 4) Spacing control.
        if dist > p["preferred_distance"] + p["band"]:
            self._counts["approach"] += 1
            return SPRINT_FORWARD
        if dist < p["preferred_distance"] - p["band"] and rng.random() < p["retreat_prob"]:
            self._counts["retreat"] += 1
            return BACK

        # 5) In-band behavior: attack, strafe, or pause.
        if dist <= REACH and rng.random() < p["attack_prob"]:
            self._counts["attack"] += 1
            return ATTACK
        if rng.random() < p["strafe_prob"]:
            self._counts["strafe"] += 1
            return STRAFE_LEFT if rng.random() < 0.5 else STRAFE_RIGHT
        if p["pause_prob"] > 0.0 and rng.random() < p["pause_prob"]:
            self._pause = p["pause_ticks"]
            self._counts["idle"] += 1
            return IDLE

        self._counts["approach"] += 1
        return SPRINT_FORWARD if dist > REACH else IDLE

    def behavior_summary(self):
        t = max(self._ticks, 1)
        return {f"{k}_rate": v / t for k, v in self._counts.items()}

    # --- presets (genomes) ---
    @classmethod
    def champion(cls, seed=0):
        """Optimize for winning: press the attack, stay in reach, rarely retreat."""
        return cls(name="champion", seed=seed,
                   preferred_distance=2.3, band=0.6, attack_prob=0.92, retreat_prob=0.05,
                   strafe_prob=0.08, jump_prob=0.02, pause_prob=0.0, pause_ticks=0)

    @classmethod
    def teacher(cls, seed=0):
        """Optimize for teaching: vary spacing/timing, allow recovery, create diverse states."""
        return cls(name="teacher", seed=seed,
                   preferred_distance=4.0, band=1.0, attack_prob=0.45, retreat_prob=0.5,
                   strafe_prob=0.4, jump_prob=0.1, pause_prob=0.05, pause_ticks=6)

    # H1 aliases (same behaviors, hypothesis-specific names).
    win_max = champion
    td_error = teacher
