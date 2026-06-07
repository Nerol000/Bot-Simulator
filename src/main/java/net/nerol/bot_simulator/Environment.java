package net.nerol.bot_simulator;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;
import net.nerol.bot_simulator.minecraft.world.item.ItemType;
import net.nerol.bot_simulator.minecraft.world.phys.PhysicsEngine;

public class Environment {
    public PvPBot bot1;
    public PvPBot bot2;
    private PhysicsEngine physics = new PhysicsEngine();
    public Environment() {
        bot1 = new PvPBot();
        bot2 = new PvPBot();

        reset();
    }

    public void executeAction(Action action) {
        System.out.printf(
                "Before: player=(%.2f, %.2f) enemy=(%.2f, %.2f)%n",
                bot1.Pos.x, bot1.Pos.z, bot2.Pos.x, bot2.Pos.z
        );

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

        System.out.printf(
                "After: player=(%.2f, %.2f) enemy=(%.2f, %.2f)%n",
                bot1.Pos.x, bot1.Pos.z, bot2.Pos.x, bot2.Pos.z
        );
    }

    // --- Your existing functions ---
    void setSprinting() {
        double speed = 0.2806;
        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw())) * speed;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw())) * speed;
    }

    void setWalking() {
        double speed = 0.21585;

        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw())) * speed;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw())) * speed;
    }
    void setBackward() {
        double speed = -0.21585;

        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw())) * speed;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw())) * speed;
    }
    void setStrafeLeft() {
        double speed = 0.21585;

        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw() - 90)) * speed;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw() - 90)) * speed;
    }
    void setStrafeRight() {
        double speed = 0.21585;

        bot1.Motion.x += Math.cos(Math.toRadians(bot1.getYaw() + 90)) * speed;
        bot1.Motion.z += Math.sin(Math.toRadians(bot1.getYaw() + 90)) * speed;
    }
    void attack() {
        double dist = distanceSquaredTobot2();

        if (dist <= 9.0) { // attack range (3^2)
            // apply knockback to bot2
            bot2.hurtTime = 5;

            double strength = 2.0;
            bot2.pendingKnockback.x = Math.cos(Math.toRadians(bot1.getYaw())) * strength;
            bot2.pendingKnockback.z = Math.sin(Math.toRadians(bot1.getYaw())) * strength;

            // optional: mark as successful hit
            bot2.wasHit = true;
        }
    }
    void jump() {
        if (bot1.onGround) {
            bot1.Motion.y = 5.0;
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
        // simplest: do nothing (static bot2)
    }

    // --- Observations (you implement these) ---
    public State getCurrentState() {
        int distance = computeDistanceBucket();
        int direction = computeDirectionBucket();
        return new State(distance, direction);
    }

    int computeDistanceBucket() {
        double dist = Math.sqrt(bot1.Pos.x * bot1.Pos.x + bot1.Pos.z * bot1.Pos.z);

        if (dist < 5) return 0;
        if (dist < 10) return 1;
        return 2;
    }
    int computeDirectionBucket() {
        int yaw = ((int) bot1.getYaw() % 360 + 360) % 360;

        if (yaw < 90) return 0;
        if (yaw < 180) return 1;
        if (yaw < 270) return 2;
        return 3;
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

        bot1.equipment.SelectedItem = ItemType.DIAMOND_SWORD;
        bot1.equipment.head = ItemType.DIAMOND_HELMET;
        bot1.equipment.chest = ItemType.DIAMOND_CHESTPLATE;
        bot1.equipment.leg = ItemType.DIAMOND_LEGGINGS;
        bot1.equipment.feet = ItemType.DIAMOND_BOOTS;

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

        bot2.equipment.SelectedItem = ItemType.DIAMOND_SWORD;
        bot2.equipment.head = ItemType.DIAMOND_HELMET;
        bot2.equipment.chest = ItemType.DIAMOND_CHESTPLATE;
        bot2.equipment.leg = ItemType.DIAMOND_LEGGINGS;
        bot2.equipment.feet = ItemType.DIAMOND_BOOTS;
    }
}
