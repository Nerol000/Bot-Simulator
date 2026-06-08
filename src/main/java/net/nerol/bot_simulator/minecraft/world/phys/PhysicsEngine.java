package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;
import net.nerol.bot_simulator.minecraft.world.item.ItemType;

public class PhysicsEngine {
    private final Gravity gravitySystem = new Gravity();
    private final Knockback knockbackSystem = new Knockback();

    public Knockback getKnockbackSystem() {
        return knockbackSystem;
    }

    public void update(PvPBot e) {
        // first apply pending knockback impulses into velocity
        knockbackSystem.applyPendingKnockback(e);

        // then run position/gravity/drag
        gravitySystem.apply(e);

        if (e.hurtTime > 0) {
            e.hurtTime--;
        }

        // Recharge attack: rises by (attackSpeed / 20) per tick, capped at 1.0.
        // Diamond sword (1.6 atk/s) finishes in 12.5 ticks; full damage requires 1.0.
        double chargePerTick = e.equipment.SelectedItem.getAttackSpeed() / ItemType.TICKS_PER_SECOND;
        e.attackCharge = Math.min(1.0, e.attackCharge + chargePerTick);
    }
}