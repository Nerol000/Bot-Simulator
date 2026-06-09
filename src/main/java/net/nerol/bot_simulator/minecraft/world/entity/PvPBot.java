package net.nerol.bot_simulator.minecraft.world.entity;

import net.nerol.bot_simulator.minecraft.util.Vec2;
import net.nerol.bot_simulator.minecraft.util.Vec3;
import net.nerol.bot_simulator.minecraft.world.entity.ai.attributes.Attributes;
import net.nerol.bot_simulator.minecraft.world.item.ItemType;
import net.nerol.bot_simulator.minecraft.world.item.equipment.Equipment;

public class PvPBot {
    public Vec3 Pos = new Vec3();
    public Vec3 Motion = new Vec3();

    public Vec2 Rotation = new Vec2();

    public int attackCooldown;

    public float Health;
    public float foodSaturationLevel;
    public float foodExhaustionLevel;
    public int foodTickTimer;
    public float foodLevel;
    public float fall_distance;

    public ItemType SelectedItem;
    public byte SelectedItemSlot;

    public boolean onGround = true;
    public boolean wasHit = false;
    public int hurtTime = 0;

    // attack timing / state
    public double attackCharge = 1.0;   // 0.0 to 1.0
    public boolean sweepingAttack = false;

    public Equipment equipment = new Equipment();
    public Attributes attributes = new Attributes();

    public boolean sprinting = false;
    public boolean walking_forward = false;
    public boolean strafing_left = false;
    public boolean strafing_right = false;
    public boolean walking_back = false;


    public float getYaw() {
        return (float)Rotation.x;
    }

    public float getPitch() {
        return (float)Rotation.y;
    }
}