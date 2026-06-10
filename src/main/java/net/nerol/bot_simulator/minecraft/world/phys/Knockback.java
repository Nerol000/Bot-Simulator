package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;
import net.nerol.bot_simulator.minecraft.world.entity.ai.attributes.AttributeType;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.EnchantmentType;

public class Knockback {

    /**
     * Applies the knockback for a melee hit, mirroring the mod's combat flow:
     *
     *  1. The base 0.4 hit reaction that {@code LivingEntity.hurt} applies to any damaged
     *     entity, directed along the attacker -> defender line.
     *  2. The extra knockback that {@code PvPBot.causeExtraKnockback} applies on top —
     *     Knockback-enchant + the +0.5 sprint bonus — directed along the attacker's facing.
     *     When that extra amount is positive, causeExtraKnockback also slows the attacker
     *     (horizontal motion * 0.6) and clears its sprint flag (the sprint-reset / "w-tap").
     *
     * @param extraStrength the {@code causeExtraKnockback} amount (enchant + sprint bonus),
     *                      i.e. everything beyond the base 0.4 hit reaction.
     */
    public void applyHitKnockback(PvPBot attacker, PvPBot defender, double extraStrength) {
        // 1) Base hit reaction: push the defender away from the attacker's position.
        knockback(defender, 0.4,
                defender.Pos.x - attacker.Pos.x,
                defender.Pos.z - attacker.Pos.z);

        // 2) Extra knockback (enchant + sprint) along the attacker's facing. In the
        //    simulator's yaw convention the attacker's forward is (cos yaw, sin yaw).
        if (extraStrength > 0.0) {
            double yaw = Math.toRadians(attacker.getYaw());
            knockback(defender, extraStrength, Math.cos(yaw), Math.sin(yaw));

            // causeExtraKnockback side effects on the ATTACKER.
            attacker.Motion.x *= 0.6;
            attacker.Motion.z *= 0.6;
            attacker.sprinting = false;
        }

        defender.hurtTime = 5;
        defender.wasHit = true;
    }

    /**
     * Port of {@code LivingEntity.knockback(strength, x, z)}: the existing horizontal
     * motion is halved and the push (normalised (pushX, pushZ) scaled by strength) is added;
     * the vertical pop is capped at 0.4 and only applied while grounded. Strength is reduced
     * by the defender's KNOCKBACK_RESISTANCE first.


     */
    private void knockback(PvPBot e, double strength, double pushX, double pushZ) {
        double resistance = e.attributes.get(AttributeType.KNOCKBACK_RESISTANCE);

        strength *= 1.0 - Math.max(0.0, Math.min(1.0, resistance));

        if (strength <= 0.0) return;

        double len = Math.sqrt(pushX * pushX + pushZ * pushZ);
        if (len < 1e-8) return;
        double nx = pushX / len;
        double nz = pushZ / len;

        e.Motion.x = e.Motion.x / 2.0 + nx * strength;
        e.Motion.z = e.Motion.z / 2.0 + nz * strength;
        if (e.onGround) {
            e.Motion.y = Math.min(0.4, e.Motion.y / 2.0 + strength);
        }
    }
}