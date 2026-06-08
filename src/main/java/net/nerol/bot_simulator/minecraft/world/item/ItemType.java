package net.nerol.bot_simulator.minecraft.world.item;

public enum ItemType {
    // attackDamage    — Minecraft "Attack Damage" attribute (HP per fully charged hit).
    // attackSpeed     — Minecraft "Attack Speed" attribute, in attacks-per-second.
    //                   Convert to per-tick cooldown via getAttackCooldownTicks().
    // Armor pieces in main-hand inherit the bare-hands defaults (1 damage, 4 atk/s).
    AIR               (1.0, 4.0),
    DIAMOND_SWORD     (7.0, 1.6),
    DIAMOND_HELMET    (1.0, 4.0),
    DIAMOND_CHESTPLATE(1.0, 4.0),
    DIAMOND_LEGGINGS  (1.0, 4.0),
    DIAMOND_BOOTS     (1.0, 4.0);

    public static final int TICKS_PER_SECOND = 20;

    public final double attackDamage;
    public final double attackSpeed;

    ItemType(double attackDamage, double attackSpeed) {
        this.attackDamage = attackDamage;
        this.attackSpeed = attackSpeed;
    }

    /** Conversion of the per-second attackSpeed attribute into the engine's native
     *  unit (ticks). Result is the number of ticks needed for a full charge.
     *  Diamond sword (1.6 atk/s) -> 20 / 1.6 = 12.5 ticks. */
    public int getAttackCooldownTicks() {
        return (int)Math.round(TICKS_PER_SECOND / attackSpeed);
    }
}