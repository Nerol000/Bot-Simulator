package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;

public class Knockback {

    public void applyPendingKnockback(PvPBot defender) {
        defender.Motion.x += defender.pendingKnockback.x;
        defender.Motion.y += defender.pendingKnockback.y;
        defender.Motion.z += defender.pendingKnockback.z;

        defender.pendingKnockback.x = 0;
        defender.pendingKnockback.y = 0;
        defender.pendingKnockback.z = 0;
    }

    public void applyAttackKnockback(PvPBot attacker, PvPBot defender) {
        double dx = defender.Pos.x - attacker.Pos.x;
        double dz = defender.Pos.z - attacker.Pos.z;

        double dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 1e-6) return;

        // normalized horizontal attack direction
        double nx = dx / dist;
        double nz = dz / dist;

        // relative horizontal motion matters
        double relVx = attacker.Motion.x - defender.Motion.x;
        double relVz = attacker.Motion.z - defender.Motion.z;
        double relativeMotionTowardTarget = relVx * nx + relVz * nz;

        // base knockback (approximation)
        double kb = 0.35;

        // sprint bonus
        if (attacker.sprinting) {
            kb += 0.25;
        }

        // fully charged sword bonus
        if (attacker.equipment.hasDiamondSword()) {
            kb += 0.15 * attacker.attackCharge;
        }

        // sweeping is wider but slightly weaker direct KB
        if (attacker.sweepingAttack) {
            kb *= 0.85;
        }

        // attacker momentum into target
        kb += Math.max(0, relativeMotionTowardTarget) * 0.20;

        // defender existing motion resists / redirects feel
        double defenderHorizontalSpeed = defender.Motion.horizontalLength();
        kb -= Math.min(0.08, defenderHorizontalSpeed * 0.05);

        // armor reduces knockback somewhat
        int armor = defender.equipment.getArmorPoints();
        double armorFactor = 1.0 - Math.min(0.40, armor * 0.02); // full diamond -> 0.60
        kb *= armorFactor;

        if (kb < 0.05) kb = 0.05;

        // apply as pending impulse
        defender.pendingKnockback.x += nx * kb;
        defender.pendingKnockback.z += nz * kb;

        // vertical pop
        double upward = attacker.sweepingAttack ? 0.08 : 0.12;
        if (attacker.sprinting) upward += 0.03;
        defender.pendingKnockback.y += upward;

        defender.hurtTime = 5;
        defender.wasHit = true;
    }
}
