package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;

public class Gravity {
    // tuned similar to Minecraft (but simplified)
    private static final double GRAVITY = -0.08;
    private static final double TERMINAL_VELOCITY = -3.92;

    public void apply(PvPBot e) {
        // Update position
        e.Pos.x += e.Motion.x;
        e.Pos.y += e.Motion.y;
        e.Pos.z += e.Motion.z;

        // Track fall distance while descending; it gates critical hits (canCriticalAttack
        // requires fall_distance > 0). Accumulates only on the way down, reset on landing.
        if (!e.onGround && e.Motion.y < 0) {
            e.fall_distance += (float) (-e.Motion.y);
        }

        // Gravity
        if (!e.onGround) {
            e.Motion.y += GRAVITY;
        }

        // Drag
        e.Motion.x *= 0.91F;
        e.Motion.y *= 0.98F;
        e.Motion.z *= 0.91F;

        // Very simple ground collision
        if (e.Pos.y <= 0) {
            e.Pos.y = 0;
            e.Motion.y = 0;
            e.onGround = true;
            e.fall_distance = 0;

            // Slight extra ground friction feel
            e.Motion.x *= 0.6;
            e.Motion.z *= 0.6;
        } else {
            e.onGround = false;
        }
    }
}