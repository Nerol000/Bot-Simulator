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

AdaptiveTeacher is the TD-Error / Teacher arm proper: instead of a single hand-tuned "hard to
predict" genome (whose avg_td overlaps the champion's because what surprises the learner drifts
as it improves), it runs a non-stationary bandit over a BANK of ParameterizedFSM behavior modes
and each episode plays the mode that currently maximizes the learner's measured |TD error|.

ImprovementTeacher is the H2-proper arm: same population/UCB/evolution machinery as
AdaptiveTeacher, but it rewards genomes by the learner's DIRECTLY measured improvement (change in
eval health_diff per block) instead of TD error. AdaptiveTeacher uses surprise as a PROXY for
good teaching; ImprovementTeacher optimizes the teaching objective directly, so comparing the two
is the sharp form of H2.
"""

import copy
import json
import math
import random
from abc import ABC, abstractmethod
from collections import deque

from environment import (
    REACH, IDLE, BACK, JUMP, STRAFE_LEFT, STRAFE_RIGHT, SPRINT_FORWARD, FORWARD, ATTACK,
    LOOK_AT_TARGET, _ray_hits,
)

# Minimum spacing (blocks) the FSM keeps while strafing: a strafe run circle-strafes at range,
# and if the opponent closes inside this it reopens the gap (BACK) instead of stepping sideways.
STRAFE_MIN_SPACING = 3.0


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
      strafe_ticks       : strafe WIDTH -- the MEAN hold length (ticks) of one strafe run (a
                           sustained A or D tap). Each run's actual length is drawn from a Gaussian
                           centered here (sigma = strafe_ticks/3), so most runs land near this
                           width; direction is a 50/50 left/right flip. 1 = the old jittery single-
                           tick strafe that nets almost no displacement; larger = wider sidesteps.
                           Runs circle-strafe while keeping >= 3 block (STRAFE_MIN_SPACING) spacing.
      jump_prob          : chance to jump on any tick
      pause_prob         : chance to begin an idle pause when in-band
      pause_ticks        : length of an idle pause
      aim_tol            : aim error (rad) above which it re-aims instead of acting
      wait_for_charge    : only swing when fully charged (charge >= 1.0) so hits land at full
                           strength while sprinting -> correct sprint-knockback timing
      stap_ticks         : post-hit backward tap length (the s-tap reset); 0 disables it
      bait               : when in reach but still recharging, RETREAT out of reach to charge
                           safely instead of holding. Beats a stationary spam-clicker ("turtle"),
                           which chips anything sitting in its reach: bait charges outside the
                           turtle's reach, then lunges in for one full-charge sprint-knockback hit.
      jump_reset         : anti-combo DEFENSE, SELF-CALIBRATING. In MC, a jump on the tick a hit
                           lands lets the jump's upward velocity override the knockback pop while
                           the standard vx/2 + push math keeps only the residual horizontal
                           knockback (env JUMP branch) -- so a jump timed onto the hit blunts the
                           combo. The FSM can't read the hit directly (was_hit is cleared before
                           act()), so it INFERS each hit from its own health dropping and MEASURES
                           the opponent's cadence itself: once it sees 3+ hits at a CONSISTENT
                           interval it locks that period and jumps on the predicted hit ticks. If
                           the opponent varies its timing the intervals stop agreeing, the lock
                           drops, and it stops jumping -- so it only defends a PREDICTABLE combo and
                           pressures the learner to hit UNPREDICTABLY. No period is configured; the
                           frequency is discovered at runtime.
    """

    DEFAULTS = dict(
        preferred_distance=2.5, band=0.75, attack_prob=0.8, retreat_prob=0.1,
        strafe_prob=0.15, strafe_ticks=4, jump_prob=0.02, pause_prob=0.0, pause_ticks=0,
        aim_tol=0.25, wait_for_charge=False, stap_ticks=0, bait=False, jump_reset=False,
    )

    def __init__(self, name="fsm", seed=0, **params):
        self.name = name
        self.params = {**self.DEFAULTS, **params}
        self.rng = random.Random(seed)
        self._pause = 0
        self._stap = 0
        self._strafe = 0
        self._strafe_dir = STRAFE_LEFT
        self._last_health = None
        self._hit_ticks = deque(maxlen=6)
        self._detected_period = 0
        self._counts = {"attack": 0, "retreat": 0, "strafe": 0, "approach": 0, "idle": 0}
        self._ticks = 0

    def reset(self):
        self._pause = 0
        self._stap = 0
        self._strafe = 0
        self._strafe_dir = STRAFE_LEFT
        self._last_health = None
        self._hit_ticks = deque(maxlen=6)
        self._detected_period = 0
        self._counts = {k: 0 for k in self._counts}
        self._ticks = 0

    def act(self, env, me, opp) -> int:
        p = self.params
        self._ticks += 1

        # 0) Jump-reset (anti-combo DEFENSE), self-calibrating. A jump on the tick a hit lands lets
        #    the jump's upward velocity override the knockback pop while the standard vx/2 + push
        #    math keeps only the residual horizontal knockback (env JUMP branch), blunting the
        #    combo. The FSM can't read the hit directly (env clears was_hit before act()), so it
        #    INFERS each hit from its own health dropping -- damage is applied during the attacker's
        #    earlier action, so a drop seen this tick means the hit landed LAST tick (self._ticks-1).
        #    It MEASURES the opponent's cadence itself: once it sees 3+ hits at a consistent
        #    interval it locks that period, then jumps on the PREDICTED next hit tick (phase-anchored
        #    to the last observed hit) so the jump coincides with the incoming hit. If the opponent
        #    varies its timing the last two gaps stop agreeing (or it stops hitting for >2 periods),
        #    the lock drops, and the FSM stops jumping -- so it only defends a PREDICTABLE combo and
        #    pressures the learner to hit UNPREDICTABLY. This runs FIRST (before the aim/stap/pause
        #    early-returns) so hit tracking never misses a tick and the timed reset has priority.
        if p["jump_reset"]:
            if self._last_health is not None and me.health < self._last_health - 1e-9:
                self._hit_ticks.append(self._ticks - 1)   # the hit actually landed last tick
                ht = self._hit_ticks
                if len(ht) >= 3:
                    i1, i2 = ht[-2] - ht[-3], ht[-1] - ht[-2]
                    # 3+ hits at the SAME frequency (last two gaps agree within a tick) -> lock it.
                    self._detected_period = round((i1 + i2) / 2) if (i2 > 0 and abs(i1 - i2) <= 1) else 0
            self._last_health = me.health
            if self._detected_period > 0 and self._hit_ticks:
                since = self._ticks - self._hit_ticks[-1]
                if since > 2 * self._detected_period:
                    self._detected_period = 0             # combo stopped -> drop the lock
                elif me.on_ground and since > 0 and since % self._detected_period == 0:
                    self._counts["idle"] += 1             # jump onto the predicted hit tick
                    return JUMP

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
        # Whether a swing THIS tick would actually connect. The hitbox is a non-rotating AABB, so
        # the reach in blocks depends on the approach angle -- a diagonal look ray clips the box's
        # corner and connects from FARTHER (center-dist) than a head-on face hit. A single center-
        # distance threshold (dist <= REACH) can't express that: it under-reaches on diagonals and
        # over-reaches head-on. So the swing gate reuses the env's OWN ray-vs-AABB test
        # (_ray_hits, the exact predicate _perform_attack uses to resolve a hit) -- the FSM swings
        # exactly when the mod would register the hit. Step 1 has already aimed at the target, so
        # the ray points the right way.
        can_hit = _ray_hits(me, opp)
        # A hit is "available" this tick when the swing would connect, aimed (step 1 guaranteed it),
        # and -- if wait_for_charge -- fully charged. Computed once so the active-strafe swing (3a)
        # and the main punish (step 4) share the same readiness gate.
        ready = (me.charge >= 1.0) if p["wait_for_charge"] else True

        # 3a) Active strafe run: a committed sustained A/D tap (a Gaussian-length run started in
        #     step 7). Movement is STICKY (the env only frictions velocity ~0.91/tick and ATTACK
        #     leaves velocity untouched), so the lateral momentum carries even on a tick we DON'T
        #     re-tap -- the bot can swing mid-run and still drift sideways, circle-strafing AND
        #     attacking. Sprint-strafe: strafing now PRESERVES the sprint flag (env), so if the run
        #     was entered sprinting a mid-strafe swing lands the FULL sprint-knockback -- a real
        #     sprint hit while circle-strafing. Preserving combos: when a full-charge hit is
        #     available at reach it takes the SAME ATTACK path as the main punish (scheduling the
        #     s-tap), so the hit still lands its knockback and the s-tap / re-engage rhythm keeps
        #     running through the strafe; a swing at reach also knocks the opponent back, which
        #     reopens spacing. Otherwise it keeps >= 3 block spacing: inside 3 blocks it reopens
        #     with BACK, else it re-taps the strafe direction. The run count decrements every tick.
        if self._strafe > 0:
            self._strafe -= 1
            if can_hit and ready and rng.random() < p["attack_prob"]:
                self._counts["attack"] += 1
                if p["stap_ticks"] > 0:
                    self._stap = p["stap_ticks"]   # keep the combo's post-hit sprint-reset tap
                return ATTACK
            if dist < STRAFE_MIN_SPACING:
                self._counts["retreat"] += 1
                return BACK
            self._counts["strafe"] += 1
            return self._strafe_dir

        # 4) Punish at the REACH edge FIRST. If a hit can land right now (in reach, charged when
        #    wait_for_charge, already aimed by step 1), swing before any spacing/jump decision.
        #    A landed hit knocks an approacher back to the reach boundary; the OLD spacing-first
        #    order then re-APPROACHED from there instead of swinging, so a stationary spam-clicker
        #    (the "turtle") chipped the FSM for free and was never punished. Swinging whenever the
        #    hit WOULD connect (can_hit) lands the full-charge sprint-knockback punish that actually
        #    beats the turtle -- and closes the old dead zone where the turtle's longer ray-hit
        #    range let it chip from just outside the FSM's center-distance swing gate.
        if can_hit and ready and rng.random() < p["attack_prob"]:
            self._counts["attack"] += 1
            if p["stap_ticks"] > 0:
                self._stap = p["stap_ticks"]   # schedule the post-hit sprint-reset tap
            return ATTACK
        if can_hit and not ready:
            if p["bait"]:
                # Recharging inside the turtle's reach is a losing trade (it chips every tick).
                # Back out of reach to charge safely, then lunge in for the punish next cycle.
                self._counts["retreat"] += 1
                return BACK
            # Otherwise hold with IDLE, which PRESERVES the sprint flag (like strafe, unlike BACK)
            # so the eventual full-charge hit still lands the sprint-knockback bonus (correct hit
            # timing) without drifting off the spot.
            self._counts["idle"] += 1
            return IDLE

        # 5) Occasional jump (movement variety / knockback dodging).
        if rng.random() < p["jump_prob"]:
            return JUMP

        # 6) Spacing control (out of reach): approach, or retreat if closer than preferred.
        if dist > p["preferred_distance"] + p["band"]:
            self._counts["approach"] += 1
            return SPRINT_FORWARD
        if dist < p["preferred_distance"] - p["band"] and rng.random() < p["retreat_prob"]:
            self._counts["retreat"] += 1
            return BACK

        # 7) In-band variety: START a strafe run or a brief pause (the unpredictable spacing the
        #    teacher wants). A/D direction is a 50/50 coin flip, and the run LENGTH is sampled from a
        #    Gaussian centered on strafe_ticks (sigma = strafe_ticks/3) so most runs land near that
        #    target width. The run is then carried out by the active-strafe block (3a), which keeps
        #    >= 3 block spacing throughout. Once a run is committed step 7 is unreachable (3a returns
        #    first) until it finishes.
        if rng.random() < p["strafe_prob"]:
            self._strafe_dir = STRAFE_LEFT if rng.random() < 0.5 else STRAFE_RIGHT   # 50/50 A vs D
            sigma = max(1.0, p["strafe_ticks"] / 3.0)
            self._strafe = max(1, round(rng.gauss(p["strafe_ticks"], sigma)))
            self._strafe -= 1                              # this tick is the run's first step
            if dist < STRAFE_MIN_SPACING:
                self._counts["retreat"] += 1
                return BACK
            self._counts["strafe"] += 1
            return self._strafe_dir
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
                   wait_for_charge=True, stap_ticks=1, jump_reset=True)

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


class AdaptiveTeacher(Opponent):
    """A teacher that ACTUALLY maximizes the learner's TD error instead of hoping one hand-tuned
    FSM preset does. The static ParameterizedFSM.teacher only *guesses* at a "hard to predict"
    genome, so its avg_td ends up overlapping the champion's: what surprises the learner keeps
    changing as it improves, and no fixed genome tracks that moving target.

    This maximizes TD as a CONTINUOUS optimization, not a 6-way multiple choice. It drives bot2
    with a POPULATION of FSM genomes and runs two coupled processes whose only objective is the
    learner's measured |TD error|:

      1) Selection -- a non-stationary discounted-UCB bandit picks, each EPISODE, the genome that
         currently maximizes the learner's mean per-tick |TD| (stale genomes get re-examined as
         the learner adapts and old surprises wear off).
      2) Evolution -- every `evolve_every` episodes the worst non-anchor genome is EVICTED and
         replaced by a Gaussian mutation of a tournament-best genome. This hill-climbs the genome
         space, so the teacher reaches high-TD genomes BETWEEN and OUTSIDE the hand presets that a
         fixed bank can't represent. (A pure 6-mode bandit tops out ~+38% over the champion; the
         search keeps climbing past that.)

    The 6 hand genomes (pressure / evasive / erratic / counter / feint / punish) seed the
    population as protected ANCHORS -- never evicted -- so proven behaviors (esp. the anti-turtle
    `punish` counter) and diversity are preserved while the extra slots evolve freely.

    One genome per game (not intra-episode switching) is deliberate: switching mid-game shreds the
    FSM's own combat rhythm (s-tap / charge timing) into a weaker, MORE learnable opponent, which
    lets the learner converge and DROPS its TD -- the opposite of the goal (we measured teacher
    avg_td falling BELOW the champion's when switching every 25 ticks). Whole-episode credit keeps
    each genome's coherent pressure, which is what actually sustains high learner TD.

    The trainer feeds the signal via feedback(td); reset() commits the reward, evolves, and
    re-selects. Because the reward IS the learner's TD, disengaged/distant genomes (near-zero TD,
    the failure mode we observed) are out-competed and bred out automatically. This is the H1
    TD-Error arm and the H2 Teacher arm."""

    # Hand genomes = protected ANCHORS seeding the population (the bandit+evolution take it from
    # here). Diverse ways to surprise the learner; `punish` is the anti-turtle counter.
    SEED_MODES = {
        "pressure": dict(preferred_distance=2.2, band=0.5, attack_prob=0.9, retreat_prob=0.05,
                         strafe_prob=0.2, jump_prob=0.05, pause_prob=0.0, pause_ticks=0,
                         wait_for_charge=True, stap_ticks=1, jump_reset=True),
        "evasive":  dict(preferred_distance=3.2, band=0.6, attack_prob=0.5, retreat_prob=0.5,
                         strafe_prob=0.6, jump_prob=0.10, pause_prob=0.05, pause_ticks=3),
        "erratic":  dict(preferred_distance=2.5, band=0.4, attack_prob=0.55, retreat_prob=0.35,
                         strafe_prob=0.5, jump_prob=0.12, pause_prob=0.15, pause_ticks=4),
        "counter":  dict(preferred_distance=2.6, band=0.5, attack_prob=0.7, retreat_prob=0.4,
                         strafe_prob=0.3, jump_prob=0.06, pause_prob=0.08, pause_ticks=5),
        "feint":    dict(preferred_distance=2.4, band=0.45, attack_prob=0.75, retreat_prob=0.2,
                         strafe_prob=0.25, jump_prob=0.04, pause_prob=0.3, pause_ticks=6),
        "punish":   dict(preferred_distance=3.2, band=0.35, attack_prob=0.95, retreat_prob=0.6,
                         strafe_prob=0.3, jump_prob=0.06, pause_prob=0.1, pause_ticks=3,
                         wait_for_charge=True, stap_ticks=2, bait=True, jump_reset=True),
    }

    # Search space for evolution: continuous knobs as (lo, hi), integer knobs as (lo, hi, "int"),
    # and boolean knobs listed separately (mutated by an occasional flip).
    PARAM_SPACE = {
        "preferred_distance": (1.5, 4.0), "band": (0.2, 1.0), "attack_prob": (0.3, 1.0),
        "retreat_prob": (0.0, 0.8), "strafe_prob": (0.0, 0.8), "jump_prob": (0.0, 0.2),
        "pause_prob": (0.0, 0.4), "pause_ticks": (0, 8, "int"), "stap_ticks": (0, 3, "int"),
        "strafe_ticks": (1, 8, "int"),
    }
    BOOL_PARAMS = ("wait_for_charge", "bait", "jump_reset")

    def __init__(self, name="teacher", seed=0, ucb_c=0.7, ema_alpha=0.15, count_discount=0.99,
                 pop_size=12, evolve_every=40, mutate_scale=0.2, flip_prob=0.1, min_evals=3):
        self.name = name
        self.rng = random.Random(seed)
        self.ucb_c = ucb_c            # UCB exploration weight over genomes
        self.ema_alpha = ema_alpha    # step size for each genome's TD-value estimate (non-stationary)
        self.count_discount = count_discount  # decays visit counts so genomes get re-tried over time
        self.pop_size = max(pop_size, len(self.SEED_MODES))
        self.evolve_every = evolve_every   # committed episodes between evolution steps
        self.mutate_scale = mutate_scale   # Gaussian std as a fraction of each param's range
        self.flip_prob = flip_prob         # per-bool chance to flip on mutation
        self.min_evals = min_evals         # min count before a genome can breed or be evicted
        self._fsm = ParameterizedFSM(name="teacher_fsm", seed=seed)

        # Population: id -> genome (full param dict). Anchors are the seed modes, kept forever.
        self._genome = {}
        self._anchor = set()
        self._value = {}
        self._count = {}
        for gid, params in self.SEED_MODES.items():
            self._genome[gid] = {**ParameterizedFSM.DEFAULTS, **params}
            self._anchor.add(gid)
            self._value[gid] = 0.0
            self._count[gid] = 0.0
        # Fill remaining slots with random genomes to explore beyond the hand presets.
        self._next_id = 0
        while len(self._genome) < self.pop_size:
            self._add_genome(self._random_genome())

        self._active = next(iter(self._genome))
        self._ep_td_sum = 0.0
        self._ep_ticks = 0
        self._commits = 0

    # ------------------------------------------------------------- genome ops
    def _add_genome(self, genome):
        gid = f"evo{self._next_id}"
        self._next_id += 1
        self._genome[gid] = genome
        self._value[gid] = 0.0
        self._count[gid] = 0.0
        return gid

    def _random_genome(self):
        g = dict(ParameterizedFSM.DEFAULTS)
        for k, spec in self.PARAM_SPACE.items():
            lo, hi = spec[0], spec[1]
            v = self.rng.uniform(lo, hi)
            g[k] = int(round(v)) if len(spec) == 3 else v
        for k in self.BOOL_PARAMS:
            g[k] = self.rng.random() < 0.5
        return g

    def _mutate(self, parent):
        g = dict(parent)
        for k, spec in self.PARAM_SPACE.items():
            lo, hi = spec[0], spec[1]
            std = (hi - lo) * self.mutate_scale
            v = g[k] + self.rng.gauss(0.0, std)
            v = min(hi, max(lo, v))                       # clamp to bounds
            g[k] = int(round(v)) if len(spec) == 3 else v
        for k in self.BOOL_PARAMS:
            if self.rng.random() < self.flip_prob:
                g[k] = not g[k]
        return g

    def _select(self):
        # Try every genome once before trusting the value estimates.
        ids = list(self._genome)
        for gid in ids:
            if self._count[gid] == 0.0:
                return gid
        log_total = math.log(sum(self._count.values()) + 1.0)
        best_id, best_score = ids[0], -1e18
        for gid in ids:
            score = self._value[gid] + self.ucb_c * math.sqrt(log_total / self._count[gid])
            if score > best_score:
                best_score, best_id = score, gid
        return best_id

    def _evolve(self):
        """Replace the worst-performing EVOLVED (non-anchor) genome with a mutation of a
        tournament-best genome. Only considers genomes with >= min_evals so noisy one-off
        estimates don't drive selection/eviction."""
        rated = [g for g in self._genome if self._count[g] >= self.min_evals]
        if len(rated) < 2:
            return
        # Parent: best of a small random tournament (favors high-TD genomes, keeps some diversity).
        k = min(3, len(rated))
        parent = max(self.rng.sample(rated, k), key=lambda g: self._value[g])
        # Victim: worst evolved genome (anchors are protected). Nothing evictable -> skip.
        evolvable = [g for g in rated if g not in self._anchor]
        if not evolvable:
            return
        victim = min(evolvable, key=lambda g: self._value[g])
        # Don't evict a genome that's better than the parent (can happen with tournament noise).
        if self._value[victim] >= self._value[parent]:
            return
        del self._genome[victim], self._value[victim], self._count[victim]
        child = self._mutate(self._genome[parent])
        gid = self._add_genome(child)
        # Warm-start the child near its parent so UCB doesn't force a blind re-eval from zero.
        self._value[gid] = self._value[parent]

    # ------------------------------------------------------------- bandit loop
    def _commit_episode(self):
        """Fold the just-finished episode's mean TD into the active genome's stats, then maybe
        evolve. Whole-episode (not intra-episode) credit on purpose: switching genome mid-game
        breaks the FSM's own combat rhythm into a weaker, MORE learnable opponent, dropping TD
        (measured). One genome per game preserves coherent pressure, which sustains higher TD."""
        if self._ep_ticks == 0:
            return
        reward = self._ep_td_sum / self._ep_ticks   # mean per-tick |TD error| under this genome
        for gid in self._genome:                      # decay all counts -> non-stationary bandit
            self._count[gid] *= self.count_discount
        self._count[self._active] += 1.0
        self._value[self._active] += self.ema_alpha * (reward - self._value[self._active])
        self._commits += 1
        if self._commits % self.evolve_every == 0:
            self._evolve()

    def feedback(self, td):
        """Called by the trainer after every learner update with that step's |TD error|."""
        self._ep_td_sum += td
        self._ep_ticks += 1

    def reset(self):
        # Commit the finished episode (may evolve), then pick the genome maximizing learner TD.
        self._commit_episode()
        self._ep_td_sum = 0.0
        self._ep_ticks = 0
        self._active = self._select()
        self._fsm.params = {**ParameterizedFSM.DEFAULTS, **self._genome[self._active]}
        self._fsm.reset()

    def act(self, env, me, opp) -> int:
        return self._fsm.act(env, me, opp)

    def behavior_summary(self):
        best = max(self._genome, key=lambda g: self._value[g])
        summary = {
            "active_genome": self._active,
            "pop_size": len(self._genome),
            "best_td_value": self._value[best],
            "best_genome": best,
        }
        return {**self._fsm.behavior_summary(), **summary}

    def export_genome(self, path):
        """Write the best (highest-value) genome's FULL parameter dict to `path` as JSON so the live
        mod can load the ACTUAL converged behavior of this arm instead of a hand representative.
        The schema (arm / genome_id / value / params) is what GenomeLoader.java reads."""
        best = max(self._genome, key=lambda g: self._value[g])
        payload = {
            "arm": self.name,
            "genome_id": best,
            "value": self._value[best],
            "params": {**ParameterizedFSM.DEFAULTS, **self._genome[best]},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path


class ImprovementTeacher(AdaptiveTeacher):
    """H2-proper teacher: rewards genomes by the learner's measured IMPROVEMENT (change in eval
    health_diff) instead of its TD error.

    Reuses AdaptiveTeacher's entire machinery -- the genome POPULATION, the non-stationary
    discounted-UCB selection, and the mutate/evict EVOLUTION -- and swaps ONLY the reward signal:

        AdaptiveTeacher reward  = mean per-tick |TD error|   (surprise; the H1 mechanism, a PROXY
                                                              for good teaching)
        ImprovementTeacher reward = delta eval health_diff   (actual learner improvement; the H2
                                                              objective, measured DIRECTLY)

    This is the arm that turns H2 from "does the surprise-maximizing teacher ALSO improve the
    learner?" into a head-to-head "does directly optimizing improvement beat using surprise as a
    proxy?". Comparing ImprovementTeacher against AdaptiveTeacher is the sharp form of H2.

    Credit assignment -- why ONE genome per inter-eval BLOCK (not per episode):
        The improvement signal only exists at eval boundaries (one health_diff number every
        --eval-every episodes). If the genome switched every episode, ~eval_every genomes would
        be smeared into a single improvement delta and no genome could be credited. So a genome is
        held for the WHOLE block between two evals; the improvement over that block is that
        genome's reward. reset() (per episode) just starts a new game with the SAME genome;
        improvement_feedback(delta) at the block boundary commits the reward, evolves, and selects
        the next block's genome.

    Caveat (state this in the writeup): the improvement signal is SPARSE and noisy -- one number
    per eval, and health_diff has large cross-seed variance -- so this arm wants FREQUENT evals
    (small --eval-every), a SMALL population, and long runs / many seeds to rank genomes reliably.
    That sparsity is the fundamental cost of optimizing improvement directly, and is itself a
    finding worth reporting."""

    def __init__(self, name="improve", seed=0, **kw):
        # The sparse improvement signal can rank only a few genomes: keep the population small
        # (6 anchors + a couple of evolvable slots) and evolve more eagerly than the TD teacher.
        kw.setdefault("pop_size", 8)
        kw.setdefault("evolve_every", 6)
        kw.setdefault("min_evals", 2)
        super().__init__(name=name, seed=seed, **kw)
        self._activate(self._active)   # configure the FSM up front (no per-episode re-select)

    def _activate(self, gid):
        """Make `gid` the block's genome and load it into the FSM."""
        self._active = gid
        self._fsm.params = {**ParameterizedFSM.DEFAULTS, **self._genome[gid]}
        self._fsm.reset()

    def feedback(self, td):
        # TD error is NOT this teacher's objective -- ignore the per-step signal entirely.
        return

    def reset(self):
        # New game, SAME genome for the whole block. No commit/select here: that happens at the
        # block (eval) boundary in improvement_feedback().
        self._fsm.reset()

    def improvement_feedback(self, delta):
        """Reward the active genome by the learner's improvement (delta eval health_diff) over the
        block it just taught, then maybe evolve and select the next block's genome."""
        for gid in self._genome:                      # non-stationary bandit: decay all counts
            self._count[gid] *= self.count_discount
        self._count[self._active] += 1.0
        self._value[self._active] += self.ema_alpha * (delta - self._value[self._active])
        self._commits += 1
        if self._commits % self.evolve_every == 0:
            self._evolve()
        self._activate(self._select())

    def behavior_summary(self):
        best = max(self._genome, key=lambda g: self._value[g])
        return {
            **self._fsm.behavior_summary(),
            "active_genome": self._active,
            "pop_size": len(self._genome),
            "best_improve_value": self._value[best],
            "best_genome": best,
        }


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
    other and learn nothing. Crucially it ALSO mixes the fallback in on a fraction
    (`fallback_prob`) of episodes AFTER the pool fills: pure self-play collapses to health_diff=-1
    at eval because the pool fills with mutually-weak selves that never advance or swing at full
    charge, so the learner never experiences an aggressive sprint-attacker -- exactly what the
    fixed w-tap EVAL opponent is. Keeping a scripted anchor in the training mix preserves a
    gradient toward that eval distribution so the curve can actually climb instead of flat-lining.

    The anchor share is ANNEALED, not constant: it starts high (`fallback_start`, mostly scripted
    pressure) and decays to a floor (`fallback_end`) as training progresses -- call
    set_progress(frac in [0,1]) each episode. A CONSTANT share collapses anyway (measured): once
    the good aggressive policy appears, the still-large block of weak-self episodes -- combined
    with the annealed learning rate, which by then updates too slowly to re-correct -- drags the
    policy back to passivity and eval falls to -1. Front-loading the anchor teaches aggression
    while lr is still high, then hands off to self-play to refine it; the floor keeps a permanent
    gradient toward the eval distribution so gains don't erode.

    Note: unlike the FSM arms, this is a MOVING, learner-dependent target -- two seeds face
    different opponents because their own histories differ. Keep the EVALUATION opponent fixed
    (the scripted w-tapper in step_eval) so hdiff / avg_td stay comparable across all arms.
    """

    def __init__(self, seed=0, pool_size=5, fallback=None,
                 fallback_start=0.8, fallback_end=0.25):
        self.rng = random.Random(seed)
        self.pool = deque(maxlen=pool_size)
        self.fallback = fallback if fallback is not None else ParameterizedFSM.champion(seed=seed)
        # Annealed scripted-anchor share: fallback_start -> fallback_end over training progress.
        self.fallback_start = fallback_start
        self.fallback_end = fallback_end
        self.fallback_prob = fallback_start   # current share; updated via set_progress()
        self._active = None

    def set_progress(self, frac):
        """Linearly anneal the anchor share from fallback_start to fallback_end. `frac` is training
        progress in [0, 1] (episode / total_episodes), supplied by the trainer each episode."""
        frac = min(1.0, max(0.0, frac))
        self.fallback_prob = self.fallback_start + (self.fallback_end - self.fallback_start) * frac

    def add_snapshot(self, table):
        """Freeze a deep copy of the current learner Q-table into the pool."""
        self.pool.append(copy.deepcopy(table))

    def reset(self):
        self.fallback.reset()
        # Empty pool, or (with prob fallback_prob) a scheduled anchor episode -> use the scripted
        # fallback. _active=None marks "play the fallback this episode".
        if not self.pool or self.rng.random() < self.fallback_prob:
            self._active = None
        else:
            self._active = self.rng.choice(self.pool)

    def act(self, env, me, opp) -> int:
        # No snapshot selected this episode -> lean on the scripted anchor (cold start OR a
        # scheduled fallback episode) so there's always aggressive pressure to learn against.
        if self._active is None:
            return self.fallback.act(env, me, opp)
        # Greedy w.r.t. the frozen snapshot, read from bot2's (=`me`) own perspective.
        return self._active.best_action(env.state_index(me, opp))

    def behavior_summary(self):
        return {"pool_size": len(self.pool)}