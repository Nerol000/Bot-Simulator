package net.nerol.bot_simulator.minecraft.world.item.enchantments;

public final class Enchantment {
    public final EnchantmentType type;
    public final int level;

    public Enchantment(EnchantmentType type, int level) {
        if (level < 1) throw new IllegalArgumentException("enchantment level must be >= 1");
        this.type = type;
        this.level = Math.min(level, type.maxLevel);
    }
}
