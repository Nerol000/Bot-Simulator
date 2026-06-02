package net.nerol.bot_simulator.minecraft.world.item.equipment;

import net.nerol.bot_simulator.minecraft.world.item.ItemType;

public class Equipment {
    public ItemType SelectedItem = ItemType.AIR;

    public ItemType head = ItemType.AIR;
    public ItemType chest = ItemType.AIR;
    public ItemType leg = ItemType.AIR;
    public ItemType feet = ItemType.AIR;
    public ItemType offhand = ItemType.AIR;

    public int getArmorPoints() {
        int armor = 0;

        if (head == ItemType.DIAMOND_HELMET) armor += 3;
        if (chest == ItemType.DIAMOND_CHESTPLATE) armor += 8;
        if (leg == ItemType.DIAMOND_LEGGINGS) armor += 6;
        if (feet == ItemType.DIAMOND_BOOTS) armor += 3;

        return armor; // full diamond = 20
    }

    public boolean hasDiamondSword() {
        return SelectedItem == ItemType.DIAMOND_SWORD;
    }

}
