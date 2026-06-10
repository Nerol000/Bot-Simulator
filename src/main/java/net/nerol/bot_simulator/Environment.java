package net.nerol.bot_simulator;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;
import net.nerol.bot_simulator.minecraft.world.entity.ai.attributes.AttributeType;
import net.nerol.bot_simulator.minecraft.world.item.ItemStack;
import net.nerol.bot_simulator.minecraft.world.item.ItemType;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.Enchantment;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.EnchantmentType;
import net.nerol.bot_simulator.minecraft.world.phys.PhysicsEngine;

import java.util.Random;

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
        performAttack(bot1, bot2);
    }

    /** Faithful port of the mod's {@code PvPBot.attack(Entity)}: attack-strength scaling,
     *  separate physical vs. enchantment (Sharpness) damage, full-strength gating, critical
     *  hits, and the sprint knockback bonus. The deployed bot (ActionPack.leftClick) only
     *  swings at strength > 0.9, and an out-of-reach swing hits air without consuming the
     *  cooldown — both mirrored here. */
    void performAttack(PvPBot attacker, PvPBot defender) {
        // attackStrengthScale = getAttackStrengthScale(0.5F): cooldown charge in [0,1].
        double scale = Math.max(0.0, Math.min(1.0, attacker.attackCharge));

        boolean fullStrengthAttack = scale > 0.9;
        if (!fullStrengthAttack) {
            return; // leftClick drops a non-full swing before it even swings.
        }

        // Out of melee reach (3 blocks): a swing at air — no damage, cooldown not consumed.
        double ddx = defender.Pos.x - attacker.Pos.x;
        double ddz = defender.Pos.z - attacker.Pos.z;
        if (ddx * ddx + ddz * ddz > 9.0) {
            return;
        }

        // baseDamage = weapon's physical ATTACK_DAMAGE; magicBoost = the Sharpness portion,
        // scaled by attack strength (getEnchantedDamage - baseDamage).
        double baseDamage = attacker.equipment.SelectedItem.type.attackDamage;
        double enchantedDamage = attacker.equipment.SelectedItem.getAttackDamage();
        double magicBoost = scale * (enchantedDamage - baseDamage);

        // Cooldown ramp on the physical portion: baseDamageScaleFactor() = 0.2 + 0.8*scale^2.
        baseDamage *= 0.2 + 0.8 * scale * scale;

        boolean knockbackAttack = attacker.sprinting && fullStrengthAttack;

        // Critical hit multiplies ONLY the physical portion; magicBoost is added afterwards.
        boolean criticalAttack = fullStrengthAttack && canCriticalAttack(attacker);
        if (criticalAttack) {
            baseDamage *= 1.5;
        }

        double totalDamage = baseDamage + magicBoost;

        // hurtOrSimulate analogue: reduce by the defender's Protection multiplier.
        double damageTaken = totalDamage * defender.equipment.getProtectionDamageMultiplier();
        if (damageTaken > 0.0) {
            defender.Health -= (float) damageTaken;

            // causeExtraKnockback amount = Knockback-enchant + (sprint ? 0.5 : 0). The base
            // 0.4 hit reaction is applied inside the knockback system.
            double enchantKnockback =
                    attacker.equipment.SelectedItem.getEnchantmentLevel(EnchantmentType.KNOCKBACK) * 0.5;
            double extraKnockback = enchantKnockback + (knockbackAttack ? 0.5 : 0.0);
            physics.getKnockbackSystem().applyHitKnockback(attacker, defender, extraKnockback);
        }

        // resetAttackStrengthTicker: the charge is consumed only on a real (in-reach) attack.
        attacker.attackCharge = 0.0;
    }

    /** Subset of the mod's {@code canCriticalAttack} the simulator can observe: the attacker
     *  must be descending (fall_distance > 0), airborne, and not sprinting. The remaining
     *  guards (not on a ladder / in water / a passenger; target is a LivingEntity) always
     *  hold in this 1v1 setup. */
    boolean canCriticalAttack(PvPBot attacker) {
        return attacker.fall_distance > 0.0f
                && !attacker.onGround
                && !attacker.sprinting;
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

    // --- bot2 s-tap combo state ---
    // Ticks remaining in the post-hit backward tap. The s-tap is a brief retreat between
    // hits that keeps bot2 from overshooting bot1 while its attack recharges, then lets it
    // sprint back in so every re-approach lands a fresh sprint-knockback hit.
    private int bot2StapTicks = 0;

    private static final double S_TAP_MEAN = 3;
    private static final double S_TAP_STDDEV = 0.75;
    private static final int S_TAP_MAX = 5;
    private static final long S_TAP_SEED = 42L;
    private final Random random = new Random(S_TAP_SEED);

    void updatebot2() {
        // S-tapping combo trainer: bot2 sprint-chases bot1, lands a sprint-knockback hit,
        // then taps backward for a couple ticks (the s-tap) before sprinting back in —
        // keeping bot1 pinned in a knockback loop. Damage, criticals and the +0.5 sprint-
        // knockback bonus all run through performAttack, exactly as bot1's attacks do.
        // bot2's movement is applied inline here so this stays self-contained to bot2.

        // Face bot1 every tick.
        double dx = bot1.Pos.x - bot2.Pos.x;
        double dz = bot1.Pos.z - bot2.Pos.z;
        bot2.Rotation.x = (float) Math.toDegrees(Math.atan2(dz, dx));
        double yaw = Math.toRadians(bot2.getYaw());

        double walkImpulse = bot2.attributes.get(AttributeType.MOVEMENT_SPEED) * WALK_IMPULSE_PER_SPEED;
        if (bot2StapTicks > 0) {
            // Post-hit s-tap: walk backward briefly. The landed hit's knockback already
            // cleared bot2's sprint, so this just spaces the approach and re-times the next
            // sprint-in to arrive in reach right as the attack finishes recharging.
            bot2StapTicks--;
            bot2.sprinting = false;
            bot2.walking_back = true;
            bot2.Motion.x -= Math.cos(yaw) * walkImpulse;
            bot2.Motion.z -= Math.sin(yaw) * walkImpulse;
        } else {
            // Chase: sprint straight at bot1 to close and hold melee reach. sprinting must
            // be true at swing time for performAttack to add the +0.5 sprint-knockback bonus.
            bot2.sprinting = true;
            bot2.walking_back = false;
            double sprintImpulse = walkImpulse * SPRINT_MULTIPLIER;
            bot2.Motion.x += Math.cos(yaw) * sprintImpulse;
            bot2.Motion.z += Math.sin(yaw) * sprintImpulse;
        }

        // Swing with the faithful combat rules (reach + full-strength gating, knockback).
        // performAttack consumes the charge only on a real in-reach hit, so a drop in the
        // charge tells us a hit just landed — that kicks off the next s-tap.
        double chargeBeforeSwing = bot2.attackCharge;
        performAttack(bot2, bot1);
        if (bot2.attackCharge < chargeBeforeSwing) {
            // Jittered s-tap length: round a Gaussian draw and clamp to [0, S_TAP_MAX].
            double sampled = S_TAP_MEAN + random.nextGaussian() * S_TAP_STDDEV;
            bot2StapTicks = (int) Math.round(Math.max(1.0, Math.min(S_TAP_MAX, sampled)));
        }
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

        if (dist < 1.66666) return 0;
        if (dist <= 3) return 1;
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
        bot1.equipment.SelectedItem = new ItemStack(ItemType.DIAMOND_SWORD);
        bot1.equipment.head  = new ItemStack(ItemType.DIAMOND_HELMET,     new Enchantment(EnchantmentType.PROTECTION, 4));
        bot1.equipment.chest = new ItemStack(ItemType.DIAMOND_CHESTPLATE, new Enchantment(EnchantmentType.PROTECTION, 4));
        bot1.equipment.leg   = new ItemStack(ItemType.DIAMOND_LEGGINGS,   new Enchantment(EnchantmentType.PROTECTION, 4));
        bot1.equipment.feet  = new ItemStack(ItemType.DIAMOND_BOOTS,      new Enchantment(EnchantmentType.PROTECTION, 4));
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
        bot2.equipment.SelectedItem = new ItemStack(ItemType.DIAMOND_SWORD);
        bot2.equipment.head  = new ItemStack(ItemType.DIAMOND_HELMET,     new Enchantment(EnchantmentType.PROTECTION, 4));
        bot2.equipment.chest = new ItemStack(ItemType.DIAMOND_CHESTPLATE, new Enchantment(EnchantmentType.PROTECTION, 4));
        bot2.equipment.leg   = new ItemStack(ItemType.DIAMOND_LEGGINGS,   new Enchantment(EnchantmentType.PROTECTION, 4));
        bot2.equipment.feet  = new ItemStack(ItemType.DIAMOND_BOOTS,      new Enchantment(EnchantmentType.PROTECTION, 4));
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