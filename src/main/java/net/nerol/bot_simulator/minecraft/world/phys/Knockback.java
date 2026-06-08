package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;
import net.nerol.bot_simulator.minecraft.world.entity.ai.attributes.AttributeType;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.EnchantmentType;

public class Knockback {

    public void applyPendingKnockback(PvPBot defender) {
        defender.Motion.x += defender.pendingKnockback.x;
        defender.Motion.y += defender.pendingKnockback.y;
        defender.Motion.z += defender.pendingKnockback.z;

        defender.pendingKnockback.x = 0;
        defender.pendingKnockback.y = 0;
        defender.pendingKnockback.z = 0;
    }

    /**
     * Compute and queue knockback for a melee hit, using the values Minecraft uses:
     *  base                       0.4
     *  + sprint  (when attacker sprinting)   +0.5      (the "w-tap" bonus)
     *  + Knockback enchant level             +0.5 / lvl
     *  * (1 - defender's KNOCKBACK_RESISTANCE)
     * Vertical pop of 0.4 is added when the defender is on the ground (matches MC's cap).
     */
    public void applyAttackKnockback(PvPBot attacker, PvPBot defender) {
        double dx = defender.Pos.x - attacker.Pos.x;
        double dz = defender.Pos.z - attacker.Pos.z;
        double dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < 1e-6) return;

        double nx = dx / dist;
        double nz = dz / dist;

        double strength = 0.4;
        if (attacker.sprinting) {
            strength += 0.5;
        }
        int knockbackLevel = attacker.equipment.SelectedItem.getEnchantmentLevel(EnchantmentType.KNOCKBACK);
        strength += knockbackLevel * 0.5;

        double resistance = defender.attributes.get(AttributeType.KNOCKBACK_RESISTANCE);
        strength *= 1.0 - Math.max(0.0, Math.min(1.0, resistance));

        if (strength <= 0.0) return;

        defender.pendingKnockback.x += nx * strength;
        defender.pendingKnockback.z += nz * strength;

        if (defender.onGround) {
            defender.pendingKnockback.y += 0.4;
        }

        defender.hurtTime = 5;
        defender.wasHit = true;
    }
}