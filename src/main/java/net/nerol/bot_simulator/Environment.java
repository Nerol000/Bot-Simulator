package net.nerol.bot_simulator;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;
import net.nerol.bot_simulator.minecraft.world.entity.ai.attributes.AttributeType;
import net.nerol.bot_simulator.minecraft.world.item.ItemStack;
import net.nerol.bot_simulator.minecraft.world.item.ItemType;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.Enchantment;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.EnchantmentType;
import net.nerol.bot_simulator.minecraft.world.phys.PhysicsEngine;

public class Environment {
    // MC's MOVEMENT_SPEED attribute (default 0.1) corresponds to ~0.21585 blocks/tick
    // walking equilibrium under 0.91 horizontal friction. Per-tick impulse therefore
    // equals MOVEMENT_SPEED * (0.21585 / 0.1) * (1 - 0.91) = MOVEMENT_SPEED * 0.1943.
    // Sprint multiplies the walking velocity by 1.3.
    private static final double WALK_IMPULSE_PER_SPEED = 0.1943;
    private static final double SPRINT_MULTIPLIER = 1.3;

    // MC's JUMP_STRENGTH default (~0.42 blocks/tick initial vertical velocity).
    private static final double JUMP_VELOCITY = 0.42;

    public PvPBot bot1;
    public PvPBot bot2;
    private PhysicsEngine physics = new PhysicsEngine();
    public Environment() {
        bot1 = new PvPBot();
        bot2 = new PvPBot();

        reset();
    }

    public void executeAction(Action action) {
        /*System.out.printf(
                "Before: player=(%.2f, %.2f) enemy=(%.2f, %.2f)%n",
                bot1.Pos.x, bot1.Pos.z, bot2.Pos.x, bot2.Pos.z
        );*/

        // Clear both per-tick hit flags; they get set true again only if a hit lands this tick.
        bot1.wasHit = false;
        bot2.wasHit = false;

        switch (action) {
            case SPRINT:
                if (!bot1.walking_back) {
                    bot1.sprinting = !bot1.sprinting;
                }
                break;
            case MOVE_FORWARD:
                if (!bot1.sprinting) {
                    bot1.walking_forward = !bot1.walking_forward;
                }
                break;
            case MOVE_BACK:
                if (!bot1.walking_forward && !bot1.sprinting) {
                    bot1.walking_back = !bot1.walking_back;
                }
                break;

            case STRAFE_LEFT:
                if (!bot1.strafing_right) {
                    bot1.strafing_left = !bot1.strafing_left;
                }
                break;

            case STRAFE_RIGHT:
                if (!bot1.strafing_left) {
                    bot1.strafing_right = !bot1.strafing_right;
                }
                break;

            case ATTACK:
                attack();
                break;

            case TURN_LEFT_45:
                turn(-45);
                break;

            case TURN_RIGHT_45:
                turn(45);
                break;

            case TURN_LEFT_90:
                turn(-90);
                break;

            case TURN_RIGHT_90:
                turn(90);
                break;

            case JUMP:
                jump();
                break;
        }

        if (bot1.sprinting) setSprinting();
        if (bot1.walking_forward) setWalking();
        if (bot1.walking_back) setBackward();
        if (bot1.strafing_left) setStrafeLeft();
        if (bot1.strafing_right) setStrafeRight();

        updatebot2();
        physics.update(bot1);
        physics.update(bot2);

        /*System.out.printf(
                "After: player=(%.2f, %.2f) enemy=(%.2f, %.2f)%n",
                bot1.Pos.x, bot1.Pos.z, bot2.Pos.x, bot2.Pos.z
        );*/
    }

    // --- Movement impulses driven by the MOVEMENT_SPEED attribute ---
    void setSprinting() {
        double impulse = bot1.attributes.get(AttributeType.MOVEMENT_SPEED) * WALK_IMPULSE_PER_SPEED * SPRINT_MULTIPLIER;
        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw())) * impulse;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw())) * impulse;
    }

    void setWalking() {
        double impulse = bot1.attributes.get(AttributeType.MOVEMENT_SPEED) * WALK_IMPULSE_PER_SPEED;
        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw())) * impulse;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw())) * impulse;
    }
    void setBackward() {
        double impulse = -bot1.attributes.get(AttributeType.MOVEMENT_SPEED) * WALK_IMPULSE_PER_SPEED;
        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw())) * impulse;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw())) * impulse;
    }
    void setStrafeLeft() {
        double impulse = bot1.attributes.get(AttributeType.MOVEMENT_SPEED) * WALK_IMPULSE_PER_SPEED;
        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw() - 90)) * impulse;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw() - 90)) * impulse;
    }
    void setStrafeRight() {
        double impulse = bot1.attributes.get(AttributeType.MOVEMENT_SPEED) * WALK_IMPULSE_PER_SPEED;
        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw() + 90)) * impulse;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw() + 90)) * impulse;
    }
    void attack() {
        // Enforce per-weapon cooldown: an early swing is dropped (matches MC's
        // attack-cooldown UX). When the swing does fire it consumes the charge
        // regardless of whether it connects.
        if (bot1.attackCharge < 1.0) {
            return;
        }

        double dist = distanceSquaredTobot2();

        if (dist <= 9.0) { // attack range (3^2)
            // Damage = weapon's attack damage (base + Sharpness bonus)
            // scaled by defender's Protection multiplier.
            double rawDamage = bot1.equipment.SelectedItem.getAttackDamage();
            // Critical hit: 1.5x damage when airborne AND not sprinting (MC rule).
            if (!bot1.onGround && !bot1.sprinting) {
                rawDamage *= 1.5;
            }
            double damageTaken = rawDamage * bot2.equipment.getProtectionDamageMultiplier();
            bot2.Health -= (float)damageTaken;

            // MC-faithful knockback: base 0.4, +0.5 if sprinting, +0.5/level Knockback,
            // multiplied by (1 - defender's KNOCKBACK_RESISTANCE). Sets hurtTime/wasHit.
            physics.getKnockbackSystem().applyAttackKnockback(bot1, bot2);
        }

        // Cooldown starts on swing, hit or miss.
        bot1.attackCharge = 0.0;
    }
    void jump() {
        if (bot1.onGround) {
            bot1.Motion.y = JUMP_VELOCITY;
            bot1.onGround = false;
        }
    }
    void turn(int yawDelta) {
        bot1.Rotation.x += yawDelta;
        bot1.Rotation.x = (bot1.Rotation.x % 360 + 360) % 360;
    }


    void lookAt(PvPBot self, PvPBot target) {
        double dx = target.Pos.x - self.Pos.x;
        double dz = target.Pos.z - self.Pos.z;

        double angle = Math.toDegrees(Math.atan2(dz, dx));

        self.Rotation.x = (float)angle;
    }

    void stopMovement() {
        bot1.Motion.x = 0;
        bot1.Motion.y = 0;
    }

    void updatebot2() {
        // Reactive trainer: bot2 always tracks bot1 and swings when in range + charged.
        // No movement, no strafing — bot1 has to learn against a present-but-predictable
        // attacker. Once policy stabilizes, swap this for a smarter opponent.

        // Face bot1 every tick.
        double dx = bot1.Pos.x - bot2.Pos.x;
        double dz = bot1.Pos.z - bot2.Pos.z;
        bot2.Rotation.x = (float) Math.toDegrees(Math.atan2(dz, dx));

        // Enforce bot2's own attack cooldown so it can't spam.
        if (bot2.attackCharge < 1.0) return;

        double dist2 = dx * dx + dz * dz;
        if (dist2 <= 9.0) { // attack range (3^2)
            // Damage = bot2's sword (+ Sharpness) * bot1's Protection multiplier.
            double rawDamage = bot2.equipment.SelectedItem.getAttackDamage();
            double damageTaken = rawDamage * bot1.equipment.getProtectionDamageMultiplier();
            bot1.Health -= (float) damageTaken;

            physics.getKnockbackSystem().applyAttackKnockback(bot2, bot1);
        }

        bot2.attackCharge = 0.0;
    }

    /** Episode is over when either bot has been reduced to 0 HP. */
    public boolean isEpisodeOver() {
        return bot1.Health <= 0 || bot2.Health <= 0;
    }

    // --- Observations (you implement these) ---
    public State getCurrentState() {
        int distance = computeDistanceBucket();
        int direction = computeDirectionBucket();
        return new State(distance, direction);
    }

    int computeDistanceBucket() {
        // Distance from bot1 to bot2, not from world origin. Same convention the MC
        // LiveController will use, so the Q-table transfers cleanly.
        double dx = bot2.Pos.x - bot1.Pos.x;
        double dz = bot2.Pos.z - bot1.Pos.z;
        double dist = Math.sqrt(dx * dx + dz * dz);

        if (dist < 5) return 0;
        if (dist < 10) return 1;
        return 2;
    }
    int computeDirectionBucket() {
        // Target's bearing relative to bot1's facing, bucketed into 8 sectors of 45 degrees.
        // 0 = target dead ahead, 2 = right, 4 = behind, 6 = left. The +22.5 offset centers
        // bucket 0 on "directly in front" so a small mis-aim still reads as FRONT.
        double dx = bot2.Pos.x - bot1.Pos.x;
        double dz = bot2.Pos.z - bot1.Pos.z;
        double bearingToTarget = Math.toDegrees(Math.atan2(dz, dx));
        double relative = ((bearingToTarget - bot1.getYaw()) % 360 + 360 + 22.5) % 360;
        return (int)(relative / 45.0); // 0..7
    }
    double distanceSquaredTobot2() {
        double dx = bot2.Pos.x - bot1.Pos.x;
        double dz = bot2.Pos.z - bot1.Pos.z;

        return dx * dx + dz * dz;
    }
    public void reset() {
        // player
        bot1.Pos.x = 0;
        bot1.Pos.y = 0;
        bot1.Pos.z = 0;
        bot1.Motion.x = 0;
        bot1.Motion.y = 0;
        bot1.Motion.z = 0;
        bot1.Rotation.x = 0.0f;
        bot1.onGround = true;
        bot1.attackCharge = 1.0;
        bot1.sweepingAttack = false;
        bot1.wasHit = false;
        bot1.sprinting = false;
        bot1.walking_forward = false;
        bot1.strafing_left = false;
        bot1.strafing_right = false;
        bot1.walking_back = false;

        bot1.Health = 20.0f;
        bot1.equipment.SelectedItem = new ItemStack(ItemType.DIAMOND_SWORD, new Enchantment(EnchantmentType.SHARPNESS, 2));
        bot1.equipment.head  = new ItemStack(ItemType.DIAMOND_HELMET,     new Enchantment(EnchantmentType.PROTECTION, 3));
        bot1.equipment.chest = new ItemStack(ItemType.DIAMOND_CHESTPLATE, new Enchantment(EnchantmentType.PROTECTION, 3));
        bot1.equipment.leg   = new ItemStack(ItemType.DIAMOND_LEGGINGS,   new Enchantment(EnchantmentType.PROTECTION, 3));
        bot1.equipment.feet  = new ItemStack(ItemType.DIAMOND_BOOTS,      new Enchantment(EnchantmentType.PROTECTION, 3));
        resetAttributesFromEquipment(bot1);

        // bot2
        bot2.Pos.x = 10;
        bot2.Pos.y = 0;
        bot2.Pos.z = 0;
        bot2.Motion.x = 0;
        bot2.Motion.y = 0;
        bot2.Motion.z = 0;
        bot2.Rotation.x = 180;
        bot2.onGround = true;
        bot2.attackCharge = 1.0;
        bot2.sweepingAttack = false;
        bot2.wasHit = false;
        bot2.sprinting = false;
        bot2.walking_forward = false;
        bot2.strafing_left = false;
        bot2.strafing_right = false;
        bot2.walking_back = false;

        bot2.Health = 20.0f;
        bot2.equipment.SelectedItem = new ItemStack(ItemType.DIAMOND_SWORD, new Enchantment(EnchantmentType.SHARPNESS, 2));
        bot2.equipment.head  = new ItemStack(ItemType.DIAMOND_HELMET,     new Enchantment(EnchantmentType.PROTECTION, 3));
        bot2.equipment.chest = new ItemStack(ItemType.DIAMOND_CHESTPLATE, new Enchantment(EnchantmentType.PROTECTION, 3));
        bot2.equipment.leg   = new ItemStack(ItemType.DIAMOND_LEGGINGS,   new Enchantment(EnchantmentType.PROTECTION, 3));
        bot2.equipment.feet  = new ItemStack(ItemType.DIAMOND_BOOTS,      new Enchantment(EnchantmentType.PROTECTION, 3));
        resetAttributesFromEquipment(bot2);
    }

    /** Repopulate attributes from the bot's current gear. Movement speed stays at the
     *  player base value; damage/speed/armor/toughness come from the equipped items. */
    private void resetAttributesFromEquipment(PvPBot bot) {
        bot.attributes.set(AttributeType.MOVEMENT_SPEED, 0.1);
        bot.attributes.set(AttributeType.ATTACK_DAMAGE, bot.equipment.SelectedItem.getAttackDamage());
        bot.attributes.set(AttributeType.ATTACK_SPEED, bot.equipment.SelectedItem.getAttackSpeed());
        bot.attributes.set(AttributeType.ARMOR, bot.equipment.getArmorPoints());
        bot.attributes.set(AttributeType.ARMOR_TOUGHNESS, bot.equipment.getArmorToughness());
    }
}