package net.nerol;

import net.nerol.bot_simulator.*;
import net.nerol.bot_simulator.minecraft.world.phys.RayTrace;

import java.io.FileWriter;
import java.io.PrintWriter;

public class Main {
    private static Action prevAction;

    private static final String CHECKPOINT_PATH = "qtable.csv";

    public static void main(String[] args) {
        QTable qTable = new QTable();

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
            for (int episode = 0; episode < 5000; episode++) {
                env.reset();
                double totalReward = 0;
                State state = env.getCurrentState();

                for (int step = 0; step < 1800; step++) { // 120 sec duel

                    Action action = agent.chooseAction(state);

                    // Snapshot HP before the tick so we can derive per-tick damage deltas.
                    float bot2HealthBefore = env.bot2.Health;
                    float bot1HealthBefore = env.bot1.Health;

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
                if (episode % 10000 == 0) System.out.println("Episode " + episode + " total reward = " + totalReward);

                if (episode % checkpointInterval(episode) == 0) qTable.save(CHECKPOINT_PATH);
            }
            //writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        qTable.save(CHECKPOINT_PATH);
    }

    static double computeReward(State s, Action a, State s2, Action prevAction, Environment env, double damageDealt, double damageTaken, boolean attackMissed) {
        double reward = 0.0;

        if (a.name().startsWith("TURN")) {
            if (s2.direction == 0) {
                reward += 0.0025;
            } else if (isCloserToFront(s.direction, s2.direction)) {
                reward += 0.0015;
            }
        }

        // Reward getting closer in distance
        if (s2.distance < s.distance) {
            reward += 0.01;
        }

        if (attackMissed) {
            reward -= 0.003;
        }

        // Discourage spam jump
        //if (a == Action.JUMP) {
        //    reward -= 0.02;
        //}

        // Small penalty for turning (encourage efficiency)
        if (a.name().startsWith("TURN")) {
            reward -= 0.001;
        }

        reward += damageDealt * 2;
        reward -= damageTaken;

        // killed opponent
        if (env.bot2.Health <= 0) {
            reward += 30.0;
        }

        // died
        if (env.bot1.Health <= 0) {
            reward -= 25.0;
        }

        return reward;
    }

    static boolean isFacingFront(int direction) {
        // Allow small tolerance, not just exact FRONT
        return direction == 0 || direction == 1 || direction == 7;
    }

    static int distanceToFront(int direction) {
        return Math.min(direction, 8 - direction);
    }

    static boolean isCloserToFront(int d1, int d2) {
        return distanceToFront(d2) < distanceToFront(d1);
    }

    static int checkpointInterval(int episode) {
        if (episode < 10_000)  return 100;
        if (episode < 100_000) return 1_000;
        return 10_000;
    }
}