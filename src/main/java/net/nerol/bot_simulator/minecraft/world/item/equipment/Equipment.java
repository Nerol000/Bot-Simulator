package net.nerol.bot_simulator.minecraft.world.item.equipment;

import net.nerol.bot_simulator.minecraft.world.item.ItemStack;
import net.nerol.bot_simulator.minecraft.world.item.ItemType;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.EnchantmentType;

public class Equipment {
    public ItemStack SelectedItem = ItemStack.EMPTY;

    public ItemStack head    = ItemStack.EMPTY;
    public ItemStack chest   = ItemStack.EMPTY;
    public ItemStack leg     = ItemStack.EMPTY;
    public ItemStack feet    = ItemStack.EMPTY;
    public ItemStack offhand = ItemStack.EMPTY;

    public int getArmorPoints() {
        int armor = 0;
        if (head.type  == ItemType.DIAMOND_HELMET)     armor += 3;
        if (chest.type == ItemType.DIAMOND_CHESTPLATE) armor += 8;
        if (leg.type   == ItemType.DIAMOND_LEGGINGS)   armor += 6;
        if (feet.type  == ItemType.DIAMOND_BOOTS)      armor += 3;
        return armor; // full diamond = 20
    }

    public int getArmorToughness() {
        int toughness = 0;
        if (head.type  == ItemType.DIAMOND_HELMET)     toughness += 2;
        if (chest.type == ItemType.DIAMOND_CHESTPLATE) toughness += 2;
        if (leg.type   == ItemType.DIAMOND_LEGGINGS)   toughness += 2;
        if (feet.type  == ItemType.DIAMOND_BOOTS)      toughness += 2;
        return toughness; // full diamond = 8
    }

    public int getTotalProtectionLevel() {
        return head.getEnchantmentLevel(EnchantmentType.PROTECTION)
                + chest.getEnchantmentLevel(EnchantmentType.PROTECTION)
                + leg.getEnchantmentLevel(EnchantmentType.PROTECTION)
                + feet.getEnchantmentLevel(EnchantmentType.PROTECTION);
    }

    // Approximates Minecraft's Protection: each level on a piece = 1 EPF, total EPF
    // is capped at 20, and incoming damage is multiplied by (1 - 0.04 * totalEpf).
    // Full diamond + Protection IV on every piece -> EPF 16 -> 0.36x damage taken.
    public double getProtectionDamageMultiplier() {
        int totalEpf = Math.min(20, getTotalProtectionLevel());
        return 1.0 - 0.04 * totalEpf;
    }

    public boolean hasDiamondSword() {
        return SelectedItem.type == ItemType.DIAMOND_SWORD;
    }
}