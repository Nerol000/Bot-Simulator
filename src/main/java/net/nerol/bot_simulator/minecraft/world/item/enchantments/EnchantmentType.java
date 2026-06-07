package net.nerol.bot_simulator.minecraft.world.item.enchantments;

public enum EnchantmentType {
    PROTECTION(4),
    UNBREAKING(3),
    SHARPNESS(5),
    KNOCKBACK(2);

    public final int maxLevel;

    EnchantmentType(int maxLevel) {
        this.maxLevel = maxLevel;
    }
}
