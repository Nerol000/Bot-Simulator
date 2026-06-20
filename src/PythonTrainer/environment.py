"""
Duel environment for the neural-net trainer — a Python port of the Java
`RL/Simulator/bot_simulator/Environment.java` (+ PhysicsEngine / Knockback / RayTrace
and the bot2 s-tap AI), extended to a full 360-degree yaw + pitch action space.

bot1 is the learning agent; bot2 is the scripted s-tap opponent. Conventions match the
Java sim: xz forward = (cos yaw, sin yaw); attacks are gated by an eye->look raytrace
against the target's AABB, so the agent must actually aim (now in 3D) to land a hit.

The env is Gym-like: reset() -> obs, step(action) -> (obs, reward, done, info).
"""

import math
import numpy as np

# --- MC-like constants (mirror Environment.java) ---
MOVEMENT_SPEED = 0.1
WALK_IMPULSE_PER_SPEED = 0.1943
SPRINT_MULTIPLIER = 1.3
WALK_IMPULSE = MOVEMENT_SPEED * WALK_IMPULSE_PER_SPEED   # per-tick forward impulse

H_FRICTION = 0.91          # horizontal ground friction
GRAVITY = 0.08
V_DRAG = 0.98
JUMP_VELOCITY = 0.42

REACH = 3.0                # ENTITY_INTERACTION_RANGE
PLAYER_WIDTH = 0.6
PLAYER_HEIGHT = 1.8
EYE_HEIGHT = 1.62

ATTACK_DAMAGE = 7.0                 # diamond sword physical damage
CHARGE_PER_TICK = 1.0 / 12.5        # diamond sword attack speed (~1.6/s -> full in 12.5 ticks)
FULL_STRENGTH = 0.9
# Net damage multiplier of full diamond armor + Protection IV (approximate; the Java sim
# computes this from the Equipment, here it's a constant so a clean hit deals ~2 HP).
PROTECTION_MULT = 0.30
BASE_KNOCKBACK = 0.4
SPRINT_KNOCKBACK = 0.5

# bot2 s-tap timing (mirror Environment.java)
S_TAP_MEAN = 3.0
S_TAP_STD = 0.75
S_TAP_MAX = 5

MAX_HEALTH = 20.0

# --- Action space (discrete; small turn deltas compose to any yaw/pitch) ---
(IDLE, FORWARD, SPRINT_FORWARD, BACK, STRAFE_LEFT, STRAFE_RIGHT,
 ATTACK, JUMP, YAW_L, YAW_R, YAW_L_FINE, YAW_R_FINE, PITCH_UP, PITCH_DOWN) = range(14)
NUM_ACTIONS = 14
YAW_STEP = 15.0
YAW_STEP_FINE = 4.0
PITCH_STEP = 8.0

OBS_DIM = 13


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _look_vec(yaw_deg, pitch_deg):
    """Unit look vector; xz forward = (cos yaw, sin yaw), matching the Java RayTrace."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    return cp * math.cos(yaw), -math.sin(pitch), cp * math.sin(yaw)


def _ray_hits(att, dfn, reach=REACH):
    """Slab-method port of RayTrace.canHit: does att's eye->look ray clip dfn's AABB?"""
    ex, ey, ez = att.x, att.y + EYE_HEIGHT, att.z
    lx, ly, lz = _look_vec(att.yaw, att.pitch)
    dx, dy, dz = lx * reach, ly * reach, lz * reach
    half = PLAYER_WIDTH / 2.0
    box = ((dfn.x - half, dfn.x + half),
           (dfn.y, dfn.y + PLAYER_HEIGHT),
           (dfn.z - half, dfn.z + half))
    origin = (ex, ey, ez)
    delta = (dx, dy, dz)
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
                 "health", "charge", "on_ground", "sprinting", "was_hit")

    def reset(self, x, z, yaw):
        self.x, self.y, self.z = x, 0.0, z
        self.vx = self.vy = self.vz = 0.0
        self.yaw, self.pitch = yaw, 0.0
        self.health = MAX_HEALTH
        self.charge = 1.0
        self.on_ground = True
        self.sprinting = False
        self.was_hit = False


class DuelEnv:
    def __init__(self, max_steps=1200, seed=0):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.bot1 = Bot()
        self.bot2 = Bot()
        self.reset()

    # ------------------------------------------------------------------ core
    def reset(self):
        self.bot1.reset(0.0, 0.0, 0.0)
        self.bot2.reset(10.0, 0.0, 180.0)
        self.steps = 0
        self._stap_ticks = 0
        self._prev_dist, self._prev_aim = self._dist(), self._aim_error()
        return self._observe()

    def step(self, action):
        self.bot1.was_hit = False
        self.bot2.was_hit = False
        hp2_before = self.bot2.health
        hp1_before = self.bot1.health

        could_hit = self.bot1.charge > FULL_STRENGTH and _ray_hits(self.bot1, self.bot2)
        self._apply_action(self.bot1, action)
        self._opponent_ai()

        self._physics(self.bot1)
        self._physics(self.bot2)
        self._recharge(self.bot1)
        self._recharge(self.bot2)

        self.steps += 1
        dmg_dealt = max(0.0, hp2_before - self.bot2.health)
        dmg_taken = max(0.0, hp1_before - self.bot1.health)
        reward = self._reward(action, dmg_dealt, dmg_taken, could_hit)
        done = self.bot1.health <= 0 or self.bot2.health <= 0 or self.steps >= self.max_steps
        return self._observe(), reward, done, {"dmg_dealt": dmg_dealt, "dmg_taken": dmg_taken}

    # ------------------------------------------------------------- actions
    def _apply_action(self, b, a):
        yaw = math.radians(b.yaw)
        if a == FORWARD:
            b.vx += math.cos(yaw) * WALK_IMPULSE
            b.vz += math.sin(yaw) * WALK_IMPULSE
        elif a == SPRINT_FORWARD:
            b.sprinting = True
            b.vx += math.cos(yaw) * WALK_IMPULSE * SPRINT_MULTIPLIER
            b.vz += math.sin(yaw) * WALK_IMPULSE * SPRINT_MULTIPLIER
        elif a == BACK:
            b.sprinting = False
            b.vx -= math.cos(yaw) * WALK_IMPULSE
            b.vz -= math.sin(yaw) * WALK_IMPULSE
        elif a == STRAFE_LEFT:
            b.sprinting = False
            b.vx += math.cos(yaw - math.pi / 2) * WALK_IMPULSE
            b.vz += math.sin(yaw - math.pi / 2) * WALK_IMPULSE
        elif a == STRAFE_RIGHT:
            b.sprinting = False
            b.vx += math.cos(yaw + math.pi / 2) * WALK_IMPULSE
            b.vz += math.sin(yaw + math.pi / 2) * WALK_IMPULSE
        elif a == ATTACK:
            self._perform_attack(self.bot1, self.bot2)
        elif a == JUMP:
            if b.on_ground:
                b.vy = JUMP_VELOCITY
                b.on_ground = False
        elif a == YAW_L:
            b.yaw = _wrap_deg(b.yaw - YAW_STEP)
        elif a == YAW_R:
            b.yaw = _wrap_deg(b.yaw + YAW_STEP)
        elif a == YAW_L_FINE:
            b.yaw = _wrap_deg(b.yaw - YAW_STEP_FINE)
        elif a == YAW_R_FINE:
            b.yaw = _wrap_deg(b.yaw + YAW_STEP_FINE)
        elif a == PITCH_UP:
            b.pitch = max(-90.0, b.pitch - PITCH_STEP)
        elif a == PITCH_DOWN:
            b.pitch = min(90.0, b.pitch + PITCH_STEP)
        # IDLE: nothing — friction/gravity still apply in _physics

    # ----------------------------------------------------------- combat
    def _perform_attack(self, att, dfn):
        if att.charge <= FULL_STRENGTH:
            return
        if not _ray_hits(att, dfn):
            return
        scale = max(0.0, min(1.0, att.charge))
        dmg = ATTACK_DAMAGE * (0.2 + 0.8 * scale * scale)
        crit = (not att.on_ground) and att.vy < 0.0 and not att.sprinting
        if crit:
            dmg *= 1.5
        taken = dmg * PROTECTION_MULT
        if taken > 0.0:
            dfn.health -= taken
            self._knockback(att, dfn)
        att.charge = 0.0
        att.sprinting = False   # sprint-reset / w-tap on a landed hit

    def _knockback(self, att, dfn):
        dx, dz = dfn.x - att.x, dfn.z - att.z
        d = math.hypot(dx, dz) or 1e-6
        nx, nz = dx / d, dz / d
        strength = BASE_KNOCKBACK + (SPRINT_KNOCKBACK if att.sprinting else 0.0)
        dfn.vx = dfn.vx / 2.0 + nx * strength
        dfn.vz = dfn.vz / 2.0 + nz * strength
        if dfn.on_ground:
            dfn.vy = min(0.4, dfn.vy / 2.0 + strength)
            dfn.on_ground = False
        dfn.was_hit = True

    def _recharge(self, b):
        b.charge = min(1.0, b.charge + CHARGE_PER_TICK)

    # ---------------------------------------------------- scripted bot2
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
            sampled = S_TAP_MEAN + self.rng.normal() * S_TAP_STD
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

    def _aim_error(self):
        """Angle (radians) between bot1's look vector and the direction to bot2's eye."""
        b1, b2 = self.bot1, self.bot2
        tx, ty, tz = b2.x - b1.x, (b2.y + EYE_HEIGHT) - (b1.y + EYE_HEIGHT), b2.z - b1.z
        tl = math.sqrt(tx * tx + ty * ty + tz * tz) or 1e-6
        lx, ly, lz = _look_vec(b1.yaw, b1.pitch)
        dot = (tx * lx + ty * ly + tz * lz) / tl
        return math.acos(max(-1.0, min(1.0, dot)))

    def _observe(self):
        b1, b2 = self.bot1, self.bot2
        dx, dz = b2.x - b1.x, b2.z - b1.z
        dy = b2.y - b1.y
        dist = math.hypot(dx, dz)
        # relative yaw to target
        bearing = math.degrees(math.atan2(dz, dx))
        rel_yaw = math.radians(_wrap_deg(bearing - b1.yaw))
        # relative pitch error (desired pitch onto target eye vs current pitch)
        desired_pitch = -math.degrees(math.atan2((b2.y + EYE_HEIGHT) - (b1.y + EYE_HEIGHT), max(dist, 1e-6)))
        rel_pitch = math.radians(_wrap_deg(desired_pitch - b1.pitch))
        # Facing-relative velocity (forward / strafe components) so it transfers cleanly to the
        # mod regardless of the sim-vs-MC yaw convention.
        syaw = math.radians(b1.yaw)
        fwd_v = b1.vx * math.cos(syaw) + b1.vz * math.sin(syaw)
        strafe_v = -b1.vx * math.sin(syaw) + b1.vz * math.cos(syaw)
        obs = np.array([
            min(dist / 10.0, 2.0),
            dy,
            math.sin(rel_yaw), math.cos(rel_yaw),
            math.sin(rel_pitch), math.cos(rel_pitch),
            fwd_v, strafe_v,
            b1.charge,
            b1.health / MAX_HEALTH,
            b2.health / MAX_HEALTH,
            1.0 if b1.on_ground else 0.0,
            1.0 if _ray_hits(b1, b2) else 0.0,
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------- reward
    def _reward(self, action, dmg_dealt, dmg_taken, could_hit):
        r = 0.0
        r += dmg_dealt
        r -= 0.75 * dmg_taken

        dist = self._dist()
        r += 0.02 * (self._prev_dist - dist)          # reward closing distance
        self._prev_dist = dist

        aim = self._aim_error()
        r += 0.1 * (self._prev_aim - aim)             # reward reducing aim error (drives 360+pitch)
        self._prev_aim = aim

        if action == ATTACK and not could_hit:
            r -= 0.1                                   # punish swinging when it can't connect

        if self.bot2.health <= 0:
            r += 20.0
        if self.bot1.health <= 0:
            r -= 20.0
        return r
