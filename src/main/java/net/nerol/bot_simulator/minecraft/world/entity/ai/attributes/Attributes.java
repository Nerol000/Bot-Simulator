package net.nerol.bot_simulator.minecraft.world.entity.ai.attributes;

import java.util.EnumMap;

/**
 * Mirrors Minecraft's per-entity attribute system in a simplified form.
 * Each {@link AttributeType} maps to a single {@code double} value initialized
 * to the attribute's default. Equipment / environment code overrides the values
 * as needed (typically in {@code Environment.reset()} after gear is applied).
 */
public class Attributes {
    private final EnumMap<AttributeType, Double> values = new EnumMap<>(AttributeType.class);

    public Attributes() {
        for (AttributeType t : AttributeType.values()) {
            values.put(t, t.defaultValue);
        }
    }

    public double get(AttributeType type) {
        return values.get(type);
    }

    public void set(AttributeType type, double value) {
        values.put(type, value);
    }
}
