package net.nerol;

import net.nerol.bot_simulator.*;
import net.nerol.bot_simulator.minecraft.world.phys.RayTrace;
import java.io.FileWriter;
import java.io.PrintWriter;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    private static Action prevAction;

    // Where the Q-table is persisted to / resumed from.
    private static final String CHECKPOINT_PATH = "qtable.csv";

    public static void main(String[] args) {

        QTable qTable = new QTable();
        // Resume from the last checkpoint if one exists, so successive runs keep improving the
        // same table instead of restarting from all-zeros. Skipped cleanly on the first run.
        if (new java.io.File(CHECKPOINT_PATH).exists()) {
            qTable.load(CHECKPOINT_PATH);
            System.out.println("Resumed Q-table from " + CHECKPOINT_PATH);
        }
        Agent agent = new Agent(qTable);
        Environment env = new Environment();

        try {
            /*PrintWriter writer =
                    new PrintWriter(
                            new FileWriter("replay.csv")
                    );
            writer.println(
                    "episode,step,"
                            + "playerX,playerY,playerZ,"
                            + "enemyX,enemyY,enemyZ,"
                            + "distance,direction,stateIndex,"
                            + "action,reward"
            );*/
            for (int episode = 0; episode < 1100100; episode++) {
                env.reset();
                double totalReward = 0;
                State state = env.getCurrentState();

                for (int step = 0; step < 1200; step++) { // 60 sec duel

                    Action action = agent.chooseAction(state);

                    // Snapshot HP before the tick so we can derive per-tick damage deltas.
                    float bot2HealthBefore = env.bot2.Health;
                    float bot1HealthBefore = env.bot1.Health;

                    // Whether an ATTACK would geometrically connect, captured BEFORE the tick
                    // mutates positions — the same eye/look raycast + reach gate performAttack
                    // uses. A true value here means the swing missed: out of melee range OR not
                    // aimed at bot2. (Checking after the tick would misread a landed hit whose
                    // own knockback shoved bot2 out of range as a miss.)
                    boolean attackMissed = action == Action.ATTACK && !RayTrace.canHit(env.bot1, env.bot2);

                    env.executeAction(action);

                    State nextState = env.getCurrentState();

                    double damageDealt = Math.max(0.0, bot2HealthBefore - env.bot2.Health);
                    double damageTaken = Math.max(0.0, bot1HealthBefore - env.bot1.Health);

                    double reward = computeReward(state, action, nextState, prevAction, env, damageDealt, damageTaken, attackMissed);

                    agent.learn(state, action, reward, nextState);

                    totalReward += reward;
                    /*System.out.printf(
                            "Action=%s reward=%.3f dist=%d nextDist=%d hit=%b%n",
                            action,
                            reward,
                            state.distance,
                            nextState.distance,
                            env.bot2.wasHit
                    );*/
                    /*writer.printf(
                            "%d,%d, %.3f,%.3f,%.3f, %.3f,%.3f,%.3f, %d,%d,%d, %s,%.3f%n",

                            episode, step,

                            env.bot1.Pos.x, env.bot1.Pos.y, env.bot1.Pos.z,

                            env.bot2.Pos.x, env.bot2.Pos.y, env.bot2.Pos.z,

                            nextState.distance,
                            nextState.direction,
                            nextState.toIndex(),

                            action,
                            reward
                    );*/
                    state = nextState;
                    prevAction = action;

                    if (env.isEpisodeOver()) break;
                }
                if (episode % 100 == 0) System.out.println("Episode " + episode + " total reward = " + totalReward);

                // Checkpoint on a cadence that widens as training progresses (see
                // checkpointInterval): frequent early when the table changes fast, sparse late
                // once it has largely converged and back-to-back saves are redundant.
                if (episode % checkpointInterval(episode) == 0) qTable.save(CHECKPOINT_PATH);
            }
            //writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Persist the trained Q-table for the mod to load. The replay CSV above is
        // kept too — it's useful for debugging individual transitions even after the
        // policy ships.
        qTable.save(CHECKPOINT_PATH);
    }

    static double computeReward(State s, Action a, State s2, Action prevAction, Environment env,
                                double damageDealt, double damageTaken, boolean attackMissed) {

        double reward = 0.0;

        // 1. Reward turning to look at bot2 (distance-independent). The bot can only hold the
        //    8 cardinal facings, and FRONT — bot2 dead ahead — is the best aim it can pick, so
        //    a turn that lands bot2 in FRONT earns the full +0.05; a turn that improves
        //    alignment without reaching FRONT earns a smaller nudge, and turning away earns
        //    nothing. This drives the bot to pick the turn that best aims at its target, which
        //    is now a prerequisite for the raycast attack to connect.
        if (a.name().startsWith("TURN")) {
            if (s2.direction == 0) {
                reward += 0.05;
            } else if (isCloserToFront(s.direction, s2.direction)) {
                reward += 0.02;
            }
        }

        // 2. Reward getting closer in distance
        if (s2.distance < s.distance) {
            reward += 0.05;
        }

        // 3. Penalty for a missed attack — swung but the raycast/reach gate didn't connect
        //    (out of melee range OR not aimed at bot2). Captured pre-tick from RayTrace.canHit,
        //    the same gate performAttack uses, so it isn't fooled by the knockback a landed hit
        //    applies after the fact.
        if (attackMissed) {
            reward -= 0.1;
        }

        // 4. Discourage spam jump
        //if (a == Action.JUMP) {
        //    reward -= 0.02;
        //}

        // 5. Small penalty for turning (encourage efficiency)
        if (a.name().startsWith("TURN")) {
            reward -= 0.01;
        }

        // 6. Reward damage dealt to opponent (1 reward per HP). Naturally scales with
        //    crits (1.5x), Sharpness, and Protection on the target — those all move
        //    the underlying damage number, which propagates here.
        reward += damageDealt;

        // 7. Punish damage taken (1 penalty per HP).
        reward -= damageTaken * 0.75;

        // 8. Terminal: killed opponent
        if (env.bot2.Health <= 0) {
            reward += 20.0;
        }

        // 9. Terminal: died
        if (env.bot1.Health <= 0) {
            reward -= 20.0;
        }

        return reward;
    }

    static int distanceToFront(int direction) {
        return Math.min(direction, 8 - direction);
    }

    static boolean isCloserToFront(int d1, int d2) {
        return distanceToFront(d2) < distanceToFront(d1);
    }

    /** Checkpoint cadence as a function of how far training has progressed. Early episodes
     *  drive the largest Q-table changes, so save often; once the policy is converging the
     *  table barely moves between episodes, so widen the interval to cut redundant writes. */
    static int checkpointInterval(int episode) {
        if (episode < 10_000)  return 100;
        if (episode < 100_000) return 1_000;
        return 10_000;
    }
}