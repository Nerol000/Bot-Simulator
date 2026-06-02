package net.nerol.bot_simulator.minecraft.world.phys;

import net.nerol.bot_simulator.minecraft.world.entity.PvPBot;

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
    }
}

