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

import copy
import math
import random
from abc import ABC, abstractmethod
from collections import deque

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
      wait_for_charge    : only swing when fully charged (charge >= 1.0) so hits land at full
                           strength while sprinting -> correct sprint-knockback timing
      stap_ticks         : post-hit backward tap length (the s-tap reset); 0 disables it
    """

    DEFAULTS = dict(
        preferred_distance=2.5, band=0.75, attack_prob=0.8, retreat_prob=0.1,
        strafe_prob=0.15, jump_prob=0.02, pause_prob=0.0, pause_ticks=0, aim_tol=0.25,
        wait_for_charge=False, stap_ticks=0,
    )

    def __init__(self, name="fsm", seed=0, **params):
        self.name = name
        self.params = {**self.DEFAULTS, **params}
        self.rng = random.Random(seed)
        self._pause = 0
        self._stap = 0
        self._counts = {"attack": 0, "retreat": 0, "strafe": 0, "approach": 0, "idle": 0}
        self._ticks = 0

    def reset(self):
        self._pause = 0
        self._stap = 0
        self._counts = {k: 0 for k in self._counts}
        self._ticks = 0

    def act(self, env, me, opp) -> int:
        p = self.params
        self._ticks += 1

        # 1) Aim first: a swing only lands if the look ray clips the target.
        if env._aim_error(me, opp) > p["aim_tol"]:
            return LOOK_AT_TARGET

        # 2) Post-hit s-tap: a brief backward tap resets sprint so the NEXT full-charge hit
        #    re-applies the sprint-knockback bonus (correct s-tap rhythm). Scheduled when it swings.
        if self._stap > 0:
            self._stap -= 1
            self._counts["retreat"] += 1
            return BACK

        # 3) Honor an in-progress pause.
        if self._pause > 0:
            self._pause -= 1
            self._counts["idle"] += 1
            return IDLE

        dist = math.hypot(opp.x - me.x, opp.z - me.z)
        rng = self.rng

        # 4) Occasional jump (movement variety / knockback dodging).
        if rng.random() < p["jump_prob"]:
            return JUMP

        # 5) Spacing control.
        if dist > p["preferred_distance"] + p["band"]:
            self._counts["approach"] += 1
            return SPRINT_FORWARD
        if dist < p["preferred_distance"] - p["band"] and rng.random() < p["retreat_prob"]:
            self._counts["retreat"] += 1
            return BACK

        # 6) In-band behavior: attack (only at full charge if wait_for_charge), strafe, or pause.
        ready = (me.charge >= 1.0) if p["wait_for_charge"] else True
        if dist <= REACH and ready and rng.random() < p["attack_prob"]:
            self._counts["attack"] += 1
            if p["stap_ticks"] > 0:
                self._stap = p["stap_ticks"]   # schedule the post-hit sprint-reset tap
            return ATTACK
        if dist <= REACH and not ready:
            # In reach but the attack is still recharging: hold with IDLE, which PRESERVES the
            # sprint flag (unlike strafe/back), so the eventual full-charge hit still lands the
            # sprint-knockback bonus. This is the core of correct hit timing.
            self._counts["idle"] += 1
            return IDLE
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
        """Optimize for WINNING: a clean s-tapper. It sprints in, waits for a FULL charge, lands a
        full-strength sprint-knockback hit, then s-taps (brief backpedal) to reset sprint and re-
        engage -- correct hit/knockback timing rather than weak spam. Low behavioral variance ->
        once the learner adapts, outcomes become predictable (lower sustained TD error). This is
        the H1 Win-Max arm and the H2 Champion arm."""
        return cls(name="champion", seed=seed,
                   preferred_distance=2.3, band=0.6, attack_prob=0.95, retreat_prob=0.05,
                   strafe_prob=0.06, jump_prob=0.02, pause_prob=0.0, pause_ticks=0,
                   wait_for_charge=True, stap_ticks=1)

    @classmethod
    def teacher(cls, seed=0):
        """Optimize for TEACHING (maximize the learner's TD error). Key design point from H1:
        TD error is large when the learner is *surprised* by big reward events (damage dealt or
        taken). So the teacher must stay ENGAGED (in reach, frequent damage exchanges) but
        UNPREDICTABLE (high variance in attack timing, strafe angle, and pauses) so the learner's
        Q-predictions are repeatedly wrong. A distant/retreating opponent produces almost no
        damage events -> near-zero TD (the failure mode we observed), which is the opposite of
        the goal. It keeps combat frequent but noisy, and allows the learner counter-openings
        (moderate retreat) so *both* sides deal damage -> larger, more varied TD signal.
        This is the H1 TD-Error arm and the H2 Teacher arm.

        Empirical check: a correct teacher should show a HIGHER avg_td during training than the
        champion at matched episodes (the avg_td the trainer already logs)."""
        return cls(name="teacher", seed=seed,
                   preferred_distance=2.5, band=0.5, attack_prob=0.6, retreat_prob=0.35,
                   strafe_prob=0.45, jump_prob=0.08, pause_prob=0.12, pause_ticks=4)

    # H1 aliases (same behaviors, hypothesis-specific names).
    win_max = champion
    td_error = teacher


class SnapshotOpponent(Opponent):
    """Self-play against FROZEN snapshots of the learner itself (prioritized fictitious
    self-play, tabular edition).

    The training loop periodically deep-copies the learner's Q-table into a bounded pool; each
    episode this opponent plays one snapshot sampled uniformly from that pool. The opponent
    therefore tracks the learner's OWN skill frontier: the gradient never vanishes (unlike a
    fixed opponent the learner has already beaten) and never walls the learner out (unlike a
    too-strong scripted expert at cold start).

    A pool of PAST selves (not just the single latest) is used on purpose -- pure latest-self
    mirror play tends to cycle (rock-paper-scissors chasing) and forget skills that beat older
    versions. Sampling from a window of recent snapshots curbs that.

    While the pool is still empty (cold start, before the first snapshot is taken) it falls back
    to `fallback` (a champion FSM by default) so two blank policies don't just flail at each
    other and learn nothing.

    Note: unlike the FSM arms, this is a MOVING, learner-dependent target -- two seeds face
    different opponents because their own histories differ. Keep the EVALUATION opponent fixed
    (the scripted w-tapper in step_eval) so hdiff / avg_td stay comparable across all arms.
    """

    def __init__(self, seed=0, pool_size=5, fallback=None):
        self.rng = random.Random(seed)
        self.pool = deque(maxlen=pool_size)
        self.fallback = fallback if fallback is not None else ParameterizedFSM.champion(seed=seed)
        self._active = None

    def add_snapshot(self, table):
        """Freeze a deep copy of the current learner Q-table into the pool."""
        self.pool.append(copy.deepcopy(table))

    def reset(self):
        self.fallback.reset()
        self._active = self.rng.choice(self.pool) if self.pool else None

    def act(self, env, me, opp) -> int:
        # Empty pool -> lean on the scripted fallback so cold start has a real opponent.
        if self._active is None:
            return self.fallback.act(env, me, opp)
        # Greedy w.r.t. the frozen snapshot, read from bot2's (=`me`) own perspective.
        return self._active.best_action(env.state_index(me, opp))

    def behavior_summary(self):
        return {"pool_size": len(self.pool)}
