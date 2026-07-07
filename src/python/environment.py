"""
Self-play duel environment for the neural-net trainer (Python port of the Java sim, extended).

During training, bot1 and bot2 are BOTH driven externally by the policy (shared-policy self-play),
so the opponent is always at the agent's own skill level — there is always a learnable gradient,
unlike training against the fixed s-tap expert (which gives a cold-start agent no foothold).
For *measuring* absolute progress, step_eval() pits the agent (bot1) against the scripted s-tap
opponent as a fixed yardstick.

Action space adds LOOK_AT_TARGET (snap exact aim) so hits are landable; full 360 yaw + pitch.

Gym-like, symmetric:
    reset() -> (obs1, obs2)
    step(a1, a2)  -> (obs1, obs2, r1, r2, done, info)   # self-play (training)
    step_eval(a1) -> (obs1, r1, done, info)             # vs scripted s-tap (evaluation)
"""

import math
import numpy as np

# --- MC-like constants (mirror Environment.java) ---
MOVEMENT_SPEED = 0.1
WALK_IMPULSE_PER_SPEED = 0.1943
SPRINT_MULTIPLIER = 1.3
WALK_IMPULSE = MOVEMENT_SPEED * WALK_IMPULSE_PER_SPEED

H_FRICTION = 0.91
GRAVITY = 0.08
V_DRAG = 0.98
JUMP_VELOCITY = 0.42

REACH = 3.0
PLAYER_WIDTH = 0.6
PLAYER_HEIGHT = 1.8
EYE_HEIGHT = 1.62

ATTACK_DAMAGE = 7.0
CHARGE_PER_TICK = 1.0 / 12.5
FULL_STRENGTH = 0.9
PROTECTION_MULT = 0.30
BASE_KNOCKBACK = 0.4
SPRINT_KNOCKBACK = 0.5

# scripted s-tap opponent (eval yardstick only)
S_TAP_MEAN = 3.0
S_TAP_STDDEV = 0.75
S_TAP_MAX = 5

MAX_HEALTH = 20.0

#--- reward weights ---
DMG_TAKEN_W = 0.5        # net-damage tilt (dealt counts full, taken half -> rewards aggression)
DIST_SHAPE_W = 0.02      # reward closing distance
AIM_SHAPE_W = 0.10       # reward reducing aim error (drives 360 + pitch aiming)
MISS_PENALTY = 0.10      # swung while CHARGED but mis-aimed (wasted a ready swing)
GOOD_SWING_W = 0.30      # charged + on-target swing -> reinforces attacking when it can land
TIME_PENALTY = 0.005     # small per-tick cost -> discourages the "both run away" stalemate
TERMINAL = 20.0

# --- discrete reward (tabular): mirrors RL/Simulator Main.computeReward, whose bucket-based
# shaping matches the 24-state resolution. The continuous reward above is finer-grained than the
# 24-state can perceive or act on, so tabular learners can't exploit it (they learn "don't
# attack" because coarse-aim swings look like misses); this discrete variant keys reward off the
# same distance/direction buckets the table indexes on. ---
D_TURN_FRONT = 0.0025    # turned to face target dead-on (direction bucket 0)
D_TURN_CLOSER = 0.0015   # turned nearer to front
D_TURN_PENALTY = 0.001   # small cost per turn (efficiency)
D_CLOSER = 0.01          # moved into a nearer distance bucket
D_MISS = 0.003           # ATTACK that couldn't connect
D_DMG_DEALT_W = 2.0
D_DMG_TAKEN_W = 1.0
D_KILL = 30.0
D_DEATH = 25.0

# --- Action space (15 discrete; LOOK_AT_TARGET snaps exact aim) ---
(IDLE, FORWARD, SPRINT_FORWARD, BACK, STRAFE_LEFT, STRAFE_RIGHT,
 ATTACK, JUMP, YAW_L, YAW_R, YAW_L_FINE, YAW_R_FINE, PITCH_UP, PITCH_DOWN, LOOK_AT_TARGET) = range(15)
NUM_ACTIONS = 15
YAW_STEP = 15.0
YAW_STEP_FINE = 4.0
PITCH_STEP = 8.0

OBS_DIM = 13


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _look_vec(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    return cp * math.cos(yaw), -math.sin(pitch), cp * math.sin(yaw)


def _ray_hits(att, dfn, reach=REACH):
    ex, ey, ez = att.x, att.y + EYE_HEIGHT, att.z
    lx, ly, lz = _look_vec(att.yaw, att.pitch)
    delta = (lx * reach, ly * reach, lz * reach)
    half = PLAYER_WIDTH / 2.0
    box = ((dfn.x - half, dfn.x + half),
           (dfn.y, dfn.y + PLAYER_HEIGHT),
           (dfn.z - half, dfn.z + half))
    origin = (ex, ey, ez)
    tmin, tmax = 0.0, 1.0
    for o, d, (lo, hi) in zip(origin, delta, box):
        if abs(d) < 1e-8:
            if o < lo or o > hi:
                return False
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


class Bot:
    __slots__ = ("x", "y", "z", "vx", "vy", "vz", "yaw", "pitch",
                 "health", "charge", "on_ground", "sprinting", "was_hit", "aim_lock")

    def reset(self, x, z, yaw):
        self.x, self.y, self.z = x, 0.0, z
        self.vx = self.vy = self.vz = 0.0
        self.yaw, self.pitch = yaw, 0.0
        self.health = MAX_HEALTH
        self.charge = 1.0
        self.on_ground = True
        self.sprinting = False
        self.was_hit = False
        self.aim_lock = False


class DuelEnv:
    def __init__(self, max_steps=1200, seed=0, time_penalty=TIME_PENALTY,
                 reward_mode="continuous", miss_penalty=MISS_PENALTY):
        self.max_steps = max_steps
        # Per-tick reward drain that discourages the "both bots run away forever" stalemate.
        # Kept at TIME_PENALTY for the neural learner; the tabular trainer passes 0.0, because
        # over ~1000 ticks the drain (~-5) sinks every learned Q-cell below the unvisited 0.0
        # cells, inverting the greedy argmax. The Java tabular trainer has no such term.
        self.time_penalty = time_penalty
        # Reward shaping used by the single-learner step_against(): "continuous" (the fine-grained
        # default, for the neural learner) or "discrete" (Java-style bucket reward, for tabular).
        self.reward_mode = reward_mode
        # Penalty for a charged-but-mis-aimed swing (continuous reward only). Exposed so the
        # tabular/coarse-state path can disable it (it over-punishes 45-degree-bucket aiming).
        self.miss_penalty = miss_penalty
        self.rng = np.random.default_rng(seed)
        self.bot1 = Bot()
        self.bot2 = Bot()
        self.reset()

    def reset(self):
        self.bot1.reset(0.0, 0.0, 0.0)
        self.bot2.reset(10.0, 0.0, 180.0)
        self.steps = 0
        self._stap_ticks = 0
        self._prev_dist = self._dist()
        self._prev_aim1 = self._aim_error(self.bot1, self.bot2)
        self._prev_aim2 = self._aim_error(self.bot2, self.bot1)
        self._prev_db1, self._prev_dir1 = self._state_buckets(self.bot1, self.bot2)
        return self._observe(self.bot1, self.bot2), self._observe(self.bot2, self.bot1)

    # ----------------------------------------------------- self-play step
    def step(self, a1, a2):
        self.bot1.was_hit = self.bot2.was_hit = False
        hp1, hp2 = self.bot1.health, self.bot2.health
        rdy1 = self.bot1.charge > FULL_STRENGTH
        rdy2 = self.bot2.charge > FULL_STRENGTH
        ch1 = rdy1 and _ray_hits(self.bot1, self.bot2)
        ch2 = rdy2 and _ray_hits(self.bot2, self.bot1)

        self._apply_action(self.bot1, self.bot2, a1)
        self._apply_action(self.bot2, self.bot1, a2)
        self._physics(self.bot1)
        self._physics(self.bot2)
        self._recharge(self.bot1)
        self._recharge(self.bot2)
        self.steps += 1

        d2 = max(0.0, hp2 - self.bot2.health)   # bot1 dealt this
        d1 = max(0.0, hp1 - self.bot1.health)   # bot2 dealt this
        done = self.bot1.health <= 0 or self.bot2.health <= 0 or self.steps >= self.max_steps

        dist = self._dist()
        aim1 = self._aim_error(self.bot1, self.bot2)
        aim2 = self._aim_error(self.bot2, self.bot1)
        r1 = self._reward(self.bot1, self.bot2, a1, d2, d1, ch1, rdy1, self._prev_dist, self._prev_aim1, dist, aim1)
        r2 = self._reward(self.bot2, self.bot1, a2, d1, d2, ch2, rdy2, self._prev_dist, self._prev_aim2, dist, aim2)
        self._prev_dist, self._prev_aim1, self._prev_aim2 = dist, aim1, aim2

        info = {"dmg_dealt": d2, "dmg_taken": d1}
        return self._observe(self.bot1, self.bot2), self._observe(self.bot2, self.bot1), r1, r2, done, info

    # ------------------------------------------- eval step (vs scripted s-tap)
    def step_eval(self, a1):
        self.bot1.was_hit = self.bot2.was_hit = False
        hp1, hp2 = self.bot1.health, self.bot2.health
        rdy1 = self.bot1.charge > FULL_STRENGTH
        ch1 = rdy1 and _ray_hits(self.bot1, self.bot2)

        self._apply_action(self.bot1, self.bot2, a1)
        self._opponent_ai()
        self._physics(self.bot1)
        self._physics(self.bot2)
        self._recharge(self.bot1)
        self._recharge(self.bot2)
        self.steps += 1

        d2 = max(0.0, hp2 - self.bot2.health)
        d1 = max(0.0, hp1 - self.bot1.health)
        done = self.bot1.health <= 0 or self.bot2.health <= 0 or self.steps >= self.max_steps

        dist = self._dist()
        aim1 = self._aim_error(self.bot1, self.bot2)
        r1 = self._reward(self.bot1, self.bot2, a1, d2, d1, ch1, rdy1, self._prev_dist, self._prev_aim1, dist, aim1)
        self._prev_dist, self._prev_aim1 = dist, aim1

        info = {"dmg_dealt": d2, "dmg_taken": d1}
        return self._observe(self.bot1, self.bot2), r1, done, info

    # ------------------------------------------- step vs a pluggable opponent
    def step_against(self, a1, opponent):
        """Advance one tick with bot1 driven by action index `a1` and bot2 driven by a
        pluggable `opponent` (an Opponent that returns an action index for bot2). Mirrors
        step_eval()'s single-learner return shape so tabular/neural learners share a loop.
        Used by the H1/H2 experiments (Win-Max / TD-Error / Champion / Teacher FSMs)."""
        self.bot1.was_hit = self.bot2.was_hit = False
        hp1, hp2 = self.bot1.health, self.bot2.health
        rdy1 = self.bot1.charge > FULL_STRENGTH
        ch1 = rdy1 and _ray_hits(self.bot1, self.bot2)

        a2 = opponent.act(self, self.bot2, self.bot1)
        self._apply_action(self.bot1, self.bot2, a1)
        self._apply_action(self.bot2, self.bot1, a2)
        self._physics(self.bot1)
        self._physics(self.bot2)
        self._recharge(self.bot1)
        self._recharge(self.bot2)
        self.steps += 1

        d2 = max(0.0, hp2 - self.bot2.health)
        d1 = max(0.0, hp1 - self.bot1.health)
        done = self.bot1.health <= 0 or self.bot2.health <= 0 or self.steps >= self.max_steps

        dist = self._dist()
        aim1 = self._aim_error(self.bot1, self.bot2)
        db, dirb = self._state_buckets(self.bot1, self.bot2)
        if self.reward_mode == "discrete":
            r1 = self._reward_discrete(self.bot1, self.bot2, a1, d2, d1, ch1,
                                       self._prev_db1, self._prev_dir1, db, dirb)
        else:
            r1 = self._reward(self.bot1, self.bot2, a1, d2, d1, ch1, rdy1,
                              self._prev_dist, self._prev_aim1, dist, aim1)
        self._prev_dist, self._prev_aim1 = dist, aim1
        self._prev_db1, self._prev_dir1 = db, dirb

        info = {"dmg_dealt": d2, "dmg_taken": d1, "opp_action": a2}
        return self._observe(self.bot1, self.bot2), r1, done, info

    # ------------------------------------------- discrete state (tabular port)
    def _state_buckets(self, me, opp):
        """(distance bucket 0..2, direction bucket 0..7) for the 24-state tabular index.
        Mirrors Environment.java computeDistanceBucket()/computeDirectionBucket()."""
        dx = opp.x - me.x
        dz = opp.z - me.z
        dist = math.hypot(dx, dz)
        if dist < 1.66666:
            dbucket = 0
        elif dist <= 3.0:
            dbucket = 1
        else:
            dbucket = 2
        bearing = math.degrees(math.atan2(dz, dx))
        relative = ((bearing - me.yaw) % 360 + 360 + 22.5) % 360
        dirbucket = int(relative / 45.0)  # 0..7
        return dbucket, dirbucket

    def state_index(self, me, opp):
        """Discrete tabular state: distance(3) x direction(8) x charged(2) -> 0..47.

        The charged bit (charge > FULL_STRENGTH) lets the agent SEE whether its next swing can
        actually deal damage, so it can LEARN to wait out the ~12-tick attack cooldown instead of
        spam-swinging. Swinging while uncharged is still allowed (it just whiffs) -- nothing is
        gated; the agent simply now has the state bit needed to prefer waiting. Without it,
        'near+aligned' looks identical whether charged or not, so attacking every tick is optimal
        (only ~1 in 12 swings lands). Extends the Java 24-state (distance x direction), which
        omitted charge."""
        dbucket, dirbucket = self._state_buckets(me, opp)
        charged = 1 if me.charge > FULL_STRENGTH else 0
        return charged * 24 + dbucket * 8 + dirbucket

    NUM_STATES = 48

    # ------------------------------------------------------------- actions
    def _apply_action(self, bot, opp, a):
        yaw = math.radians(bot.yaw)
        if a == FORWARD:
            bot.vx += math.cos(yaw) * WALK_IMPULSE
            bot.vz += math.sin(yaw) * WALK_IMPULSE
        elif a == SPRINT_FORWARD:
            bot.sprinting = True
            bot.vx += math.cos(yaw) * WALK_IMPULSE * SPRINT_MULTIPLIER
            bot.vz += math.sin(yaw) * WALK_IMPULSE * SPRINT_MULTIPLIER
        elif a == BACK:
            bot.sprinting = False
            bot.vx -= math.cos(yaw) * WALK_IMPULSE
            bot.vz -= math.sin(yaw) * WALK_IMPULSE
        elif a == STRAFE_LEFT:
            bot.sprinting = False
            bot.vx += math.cos(yaw - math.pi / 2) * WALK_IMPULSE
            bot.vz += math.sin(yaw - math.pi / 2) * WALK_IMPULSE
        elif a == STRAFE_RIGHT:
            bot.sprinting = False
            bot.vx += math.cos(yaw + math.pi / 2) * WALK_IMPULSE
            bot.vz += math.sin(yaw + math.pi / 2) * WALK_IMPULSE
        elif a == ATTACK:
            self._perform_attack(bot, opp)
        elif a == JUMP:
            if bot.on_ground:
                bot.vy = JUMP_VELOCITY
                bot.on_ground = False
        elif a == YAW_L:
            bot.aim_lock = False
            bot.yaw = _wrap_deg(bot.yaw - YAW_STEP)
        elif a == YAW_R:
            bot.aim_lock = False
            bot.yaw = _wrap_deg(bot.yaw + YAW_STEP)
        elif a == YAW_L_FINE:
            bot.aim_lock = False
            bot.yaw = _wrap_deg(bot.yaw - YAW_STEP_FINE)
        elif a == YAW_R_FINE:
            bot.aim_lock = False
            bot.yaw = _wrap_deg(bot.yaw + YAW_STEP_FINE)
        elif a == PITCH_UP:
            bot.aim_lock = False
            bot.pitch = max(-90.0, bot.pitch - PITCH_STEP)
        elif a == PITCH_DOWN:
            bot.aim_lock = False
            bot.pitch = min(90.0, bot.pitch + PITCH_STEP)
        elif a == LOOK_AT_TARGET:
            # Sticky: latch auto-tracking on. The bot now keeps its aim snapped onto the target
            # every tick (see the tail of this method) until it chooses a manual rotation action
            # (YAW_*/PITCH_*), which releases the lock and hands rotation back to the policy.
            # This lets a greedy tabular policy aim WITHOUT sequencing look->attack in one state:
            # it can LOOK in an approach state and the aim persists into the near+charged state
            # where it attacks.
            bot.aim_lock = True
            self._snap_aim(bot, opp)
        # IDLE: nothing

        # While latched, re-track the target every tick regardless of the (non-rotation) action
        # taken, so aim stays locked through movement/attack/idle ticks.
        if bot.aim_lock:
            self._snap_aim(bot, opp)

    def _snap_aim(self, bot, opp):
        dx, dz, dy = opp.x - bot.x, opp.z - bot.z, opp.y - bot.y
        bot.yaw = math.degrees(math.atan2(dz, dx))
        bot.pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))

    # ----------------------------------------------------------- combat
    def _perform_attack(self, att, dfn):
        # No charge floor: a sub-full-charge swing still lands (damage scales via 0.2+0.8*scale^2),
        # matching the deployed mod's ActionPack.attack() which always swings at an in-reach target.
        # The agent LEARNS to wait for charge (weak/whiffed swings waste the cooldown) rather than
        # being physically prevented from swinging early.
        if not _ray_hits(att, dfn):
            att.charge = 0.0            # a swing at air still consumes the charge
            return
        scale = max(0.0, min(1.0, att.charge))
        dmg = ATTACK_DAMAGE * (0.2 + 0.8 * scale * scale)
        # Crits require a FULL-strength swing (scale >= 1.0), matching the Java sim's
        # fullStrengthAttack gate -- a partially-charged swing deals damage but can't crit
        # or apply the sprint-knockback bonus.
        full_strength = scale >= 1.0
        if full_strength and (not att.on_ground) and att.vy < 0.0 and not att.sprinting:
            dmg *= 1.5
        taken = dmg * PROTECTION_MULT
        if taken > 0.0:
            dfn.health -= taken
            self._knockback(att, dfn, full_strength)
            att.sprinting = False       # sprint resets on a landed hit
        att.charge = 0.0

    def _knockback(self, att, dfn, full_strength):
        dx, dz = dfn.x - att.x, dfn.z - att.z
        d = math.hypot(dx, dz) or 1e-6
        nx, nz = dx / d, dz / d
        # Sprint-knockback bonus only on a full-strength hit (Java: sprinting && fullStrengthAttack).
        strength = BASE_KNOCKBACK + (SPRINT_KNOCKBACK if (att.sprinting and full_strength) else 0.0)
        dfn.vx = dfn.vx / 2.0 + nx * strength
        dfn.vz = dfn.vz / 2.0 + nz * strength
        if dfn.on_ground:
            dfn.vy = min(0.4, dfn.vy / 2.0 + strength)
            dfn.on_ground = False
        dfn.was_hit = True

    def _recharge(self, b):
        b.charge = min(1.0, b.charge + CHARGE_PER_TICK)

    # ----------------------------------------------- scripted opponent (eval)
    def _opponent_ai(self):
        b1, b2 = self.bot1, self.bot2
        dx, dy, dz = b1.x - b2.x, b1.y - b2.y, b1.z - b2.z
        b2.yaw = math.degrees(math.atan2(dz, dx))
        b2.pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
        yaw = math.radians(b2.yaw)
        walk = WALK_IMPULSE
        if self._stap_ticks > 0:
            self._stap_ticks -= 1
            b2.sprinting = False
            b2.vx -= math.cos(yaw) * walk
            b2.vz -= math.sin(yaw) * walk
        else:
            b2.sprinting = True
            b2.vx += math.cos(yaw) * walk * SPRINT_MULTIPLIER
            b2.vz += math.sin(yaw) * walk * SPRINT_MULTIPLIER
        before = b2.charge
        self._perform_attack(b2, b1)
        if b2.charge < before:
            sampled = S_TAP_MEAN + self.rng.normal() * S_TAP_STDDEV
            self._stap_ticks = int(round(max(1.0, min(float(S_TAP_MAX), sampled))))

    # ------------------------------------------------------- physics
    def _physics(self, b):
        b.x += b.vx
        b.y += b.vy
        b.z += b.vz
        b.vx *= H_FRICTION
        b.vz *= H_FRICTION
        if b.y <= 0.0:
            b.y = 0.0
            b.vy = 0.0
            b.on_ground = True
        else:
            b.vy = (b.vy - GRAVITY) * V_DRAG
            b.on_ground = False

    # ------------------------------------------------------- observation
    def _dist(self):
        return math.hypot(self.bot2.x - self.bot1.x, self.bot2.z - self.bot1.z)

    def _aim_error(self, me, opp):
        tx, ty, tz = opp.x - me.x, opp.y - me.y, opp.z - me.z   # eye heights cancel
        tl = math.sqrt(tx * tx + ty * ty + tz * tz) or 1e-6
        lx, ly, lz = _look_vec(me.yaw, me.pitch)
        dot = (tx * lx + ty * ly + tz * lz) / tl
        return math.acos(max(-1.0, min(1.0, dot)))

    def _observe(self, me, opp):
        dx, dz, dy = opp.x - me.x, opp.z - me.z, opp.y - me.y
        dist = math.hypot(dx, dz)
        bearing = math.degrees(math.atan2(dz, dx))
        rel_yaw = math.radians(_wrap_deg(bearing - me.yaw))
        desired_pitch = -math.degrees(math.atan2(dy, max(dist, 1e-6)))
        rel_pitch = math.radians(_wrap_deg(desired_pitch - me.pitch))
        syaw = math.radians(me.yaw)
        fwd_v = me.vx * math.cos(syaw) + me.vz * math.sin(syaw)
        strafe_v = -me.vx * math.sin(syaw) + me.vz * math.cos(syaw)
        return np.array([
            min(dist / 10.0, 2.0),
            dy,
            math.sin(rel_yaw), math.cos(rel_yaw),
            math.sin(rel_pitch), math.cos(rel_pitch),
            fwd_v, strafe_v,
            me.charge,
            me.health / MAX_HEALTH,
            opp.health / MAX_HEALTH,
            1.0 if me.on_ground else 0.0,
            1.0 if _ray_hits(me, opp) else 0.0,
        ], dtype=np.float32)

    # ------------------------------------------------------- reward
    def _reward(self, me, opp, action, dmg_dealt, dmg_taken, could_hit, was_ready, prev_dist, prev_aim, dist, aim):
        r = dmg_dealt - DMG_TAKEN_W * dmg_taken
        r += DIST_SHAPE_W * (prev_dist - dist)
        r += AIM_SHAPE_W * (prev_aim - aim)
        if action == ATTACK:
            if could_hit:
                r += GOOD_SWING_W          # charged + on target -> land the hit
            elif was_ready:
                r -= self.miss_penalty     # charged but mis-aimed -> wasted swing
            # swinging on cooldown is harmless in MC: no penalty
        r -= self.time_penalty
        if opp.health <= 0:
            r += TERMINAL
        if me.health <= 0:
            r -= TERMINAL
        return r

    def _reward_discrete(self, me, opp, action, dmg_dealt, dmg_taken, could_hit,
                         prev_db, prev_dir, db, dirb):
        """Java-style bucket reward (RL/Simulator Main.computeReward). Keys off the same coarse
        distance/direction buckets the 24-state table indexes on, so the tabular learner can
        actually act on it (unlike the continuous reward, which needs sub-bucket precision)."""
        r = 0.0
        is_turn = action in (YAW_L, YAW_R, YAW_L_FINE, YAW_R_FINE)
        if is_turn:
            if dirb == 0:
                r += D_TURN_FRONT
            elif min(dirb, 8 - dirb) < min(prev_dir, 8 - prev_dir):
                r += D_TURN_CLOSER
            r -= D_TURN_PENALTY
        if db < prev_db:
            r += D_CLOSER                   # moved into a nearer distance bucket
        if action == ATTACK and not could_hit:
            r -= D_MISS
        r += D_DMG_DEALT_W * dmg_dealt
        r -= D_DMG_TAKEN_W * dmg_taken
        if opp.health <= 0:
            r += D_KILL
        if me.health <= 0:
            r -= D_DEATH
        return r