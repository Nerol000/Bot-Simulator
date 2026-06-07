package net.nerol.bot_simulator.minecraft.world.entity.ai.attributes;

public enum AttributeType {
    /** Movement Speed. MC default for player = 0.1. With friction 0.91 this yields
     *  ~0.21585 blocks/tick walking velocity; sprint multiplies by 1.3 → ~0.281. */
    MOVEMENT_SPEED  (0.1),

    /** Base attack damage in HP per fully-charged hit. Bare hands = 1.0. */
    ATTACK_DAMAGE   (1.0),

    /** Armor points. Each point ≈ 4% physical damage reduction (capped by toughness). */
    ARMOR           (0.0),

    /** Armor toughness. Reduces the low-armor scaling of incoming damage. */
    ARMOR_TOUGHNESS (0.0),

    /** Attack speed in attacks-per-second. Bare hands = 4.0, diamond sword = 1.6. */
    ATTACK_SPEED    (4.0),

    /** Knockback Resistance multiplier (0.0 = no resistance, 1.0 = full immunity).
     *  Vanilla diamond armor adds 0 resistance; Netherite adds 0.1 per piece. */
    KNOCKBACK_RESISTANCE (0.0);

    public final double defaultValue;

    AttributeType(double defaultValue) {
        this.defaultValue = defaultValue;
    }
}
