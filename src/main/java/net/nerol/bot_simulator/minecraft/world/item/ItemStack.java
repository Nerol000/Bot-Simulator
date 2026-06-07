package net.nerol.bot_simulator.minecraft.world.item;

import net.nerol.bot_simulator.minecraft.world.item.enchantments.Enchantment;
import net.nerol.bot_simulator.minecraft.world.item.enchantments.EnchantmentType;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class ItemStack {
    public static final ItemStack EMPTY = new ItemStack(ItemType.AIR);

    public final ItemType type;
    private final List<Enchantment> enchantments;

    public ItemStack(ItemType type, Enchantment... enchantments) {
        this.type = type;
        this.enchantments = (enchantments.length == 0)
                ? Collections.emptyList()
                : Collections.unmodifiableList(Arrays.asList(enchantments));
    }

    public int getEnchantmentLevel(EnchantmentType type) {
        for (Enchantment e : enchantments) {
            if (e.type == type) return e.level;
        }
        return 0;
    }

    public boolean isEmpty() {
        return type == ItemType.AIR;
    }

    // --- Combat stats ---

    /** Base attack damage of this item plus Sharpness bonus, if any.
     *  Sharpness in modern Minecraft: damage bonus = 0.5 * level + 0.5
     *  (Sharpness I = +1, V = +3). */
    public double getAttackDamage() {
        double damage = type.attackDamage;
        int sharpness = getEnchantmentLevel(EnchantmentType.SHARPNESS);
        if (sharpness > 0) {
            damage += 0.5 * sharpness + 0.5;
        }
        return damage;
    }

    /** Attack speed in attacks-per-second (Minecraft's attribute value).
     *  Use {@link #getAttackCooldownTicks()} for the engine-native (tick) form. */
    public double getAttackSpeed() {
        return type.attackSpeed;
    }

    /** Cooldown duration in ticks for the attack to fully recharge. Delegates to
     *  {@link ItemType#getAttackCooldownTicks()} so there's one source of truth. */
    public int getAttackCooldownTicks() {
        return type.getAttackCooldownTicks();
    }

    // --- Unbreaking ---

    public int getUnbreakingLevel() {
        return getEnchantmentLevel(EnchantmentType.UNBREAKING);
    }

    /** Probability that a durability-consuming use will NOT actually decrement durability.
     *  Minecraft formula: level / (level + 1). Unbreaking I = 50%, II = 67%, III = 75%.
     *  Returns 0.0 when no Unbreaking is present. Ready for whenever durability is wired up. */
    public double getUnbreakingSkipChance() {
        int level = getUnbreakingLevel();
        if (level <= 0) return 0.0;
        return level / (double)(level + 1);
    }
}
