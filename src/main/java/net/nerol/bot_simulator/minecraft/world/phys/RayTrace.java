package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.util.Vec3;
import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;

/**
 * Faithful port of the entity-pick raycast the deployed bot runs in
 * {@code ActionPack.leftClick}: a ray is cast from the attacker's eye along its look
 * direction out to the entity-interaction reach (3 blocks), and the swing only lands if
 * that ray clips the target's bounding box.
 *
 * <p>This replaces the simulator's old "within 3 blocks (distance-squared)" reach check,
 * which ignored facing entirely — a bot could "hit" a target standing directly behind it.
 * With the raytrace the attacker must actually be aiming at the target's hitbox, exactly as
 * the real swing requires, so the learned policy can't exploit hits it could never land in
 * the mod.
 *
 * <p>Convention note: the simulator's forward vector is {@code (cos yaw, sin yaw)} in xz
 * (see {@code Environment}'s movement impulses and {@code Knockback}), so the look vector
 * here follows the same convention rather than MC's {@code (-sin yaw, cos yaw)}.
 */
public class RayTrace {
    // MC player hitbox dimensions and standing eye height.
    public static final double PLAYER_WIDTH  = 0.6;
    public static final double PLAYER_HEIGHT = 1.8;
    public static final double EYE_HEIGHT    = 1.62;

    // ENTITY_INTERACTION_RANGE default — the same reach ActionPack.leftClick reads.
    public static final double DEFAULT_REACH = 3.0;

    /** True if {@code attacker}, aiming along its current look direction, can hit {@code defender}. */
    public static boolean canHit(PvPBot attacker, PvPBot defender) {
        return canHit(attacker, defender, DEFAULT_REACH);
    }

    public static boolean canHit(PvPBot attacker, PvPBot defender, double reach) {
        Vec3 eye  = eyePosition(attacker);
        Vec3 look = lookAngle(attacker);
        Vec3 end  = eye.add(look.x * reach, look.y * reach, look.z * reach);

        Vec3 hit = boundingBox(defender).clip(eye, end);
        if (hit == null) return false;

        // Mirror MC's closestDistSq < reach*reach gate. The clip point already lies on the
        // [eye, end] segment so this can only fail to floating-point slop; keep it as a guard.
        double dx = hit.x - eye.x, dy = hit.y - eye.y, dz = hit.z - eye.z;
        return dx * dx + dy * dy + dz * dz <= reach * reach + 1e-9;
    }

    static Vec3 eyePosition(PvPBot bot) {
        return new Vec3(bot.Pos.x, bot.Pos.y + EYE_HEIGHT, bot.Pos.z);
    }

    /** Unit look vector; xz forward is {@code (cos yaw, sin yaw)} to match the simulator. */
    static Vec3 lookAngle(PvPBot bot) {
        double yaw   = Math.toRadians(bot.getYaw());
        double pitch = Math.toRadians(bot.getPitch());
        double cosPitch = Math.cos(pitch);
        return new Vec3(cosPitch * Math.cos(yaw), -Math.sin(pitch), cosPitch * Math.sin(yaw));
    }

    /** Defender's hitbox: {@value #PLAYER_WIDTH} wide, {@value #PLAYER_HEIGHT} tall, centered
     *  horizontally on {@code Pos} with feet at {@code Pos.y}. PickRadius is 0 for players. */
    static AABB boundingBox(PvPBot bot) {
        double half = PLAYER_WIDTH / 2.0;
        return new AABB(
                bot.Pos.x - half, bot.Pos.y,                 bot.Pos.z - half,
                bot.Pos.x + half, bot.Pos.y + PLAYER_HEIGHT, bot.Pos.z + half);
    }

    /** Minimal axis-aligned bounding box with a segment-clip mirroring MC's {@code AABB.clip}. */
    public static final class AABB {
        public final double minX, minY, minZ, maxX, maxY, maxZ;

        public AABB(double minX, double minY, double minZ, double maxX, double maxY, double maxZ) {
            this.minX = Math.min(minX, maxX); this.maxX = Math.max(minX, maxX);
            this.minY = Math.min(minY, maxY); this.maxY = Math.max(minY, maxY);
            this.minZ = Math.min(minZ, maxZ); this.maxZ = Math.max(minZ, maxZ);
        }

        /**
         * Returns the point where the segment {@code from -> to} first enters this box, or
         * {@code null} if it never does. Slab method: shrink the [0,1] parameter window
         * against each axis pair; if {@code tmin <= tmax} survives all three the segment
         * intersects, entering at {@code tmin}.
         */
        public Vec3 clip(Vec3 from, Vec3 to) {
            double dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;

            // window[0] = tmin (entry), window[1] = tmax (exit), parameterized along the segment.
            double[] window = new double[]{0.0, 1.0};
            if (!clipAxis(from.x, dx, minX, maxX, window)) return null;
            if (!clipAxis(from.y, dy, minY, maxY, window)) return null;
            if (!clipAxis(from.z, dz, minZ, maxZ, window)) return null;

            double tEnter = window[0];
            return new Vec3(from.x + tEnter * dx, from.y + tEnter * dy, from.z + tEnter * dz);
        }

        private static boolean clipAxis(double start, double delta, double lo, double hi, double[] window) {
            if (Math.abs(delta) < 1e-8) {
                // Segment is parallel to this slab: it can only intersect if it already
                // starts between the slab's faces.
                return start >= lo && start <= hi;
            }
            double t1 = (lo - start) / delta;
            double t2 = (hi - start) / delta;
            if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }
            if (t1 > window[0]) window[0] = t1;
            if (t2 < window[1]) window[1] = t2;
            return window[0] <= window[1];
        }
    }
}