package net.nerol;

import net.nerol.bot_simulator.*;
import java.io.FileWriter;
import java.io.PrintWriter;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    private static Action prevAction;

    public static void main(String[] args) {

        QTable qTable = new QTable();
        Agent agent = new Agent(qTable);
        Environment env = new Environment();

        try
        {
            PrintWriter writer =
                    new PrintWriter(
                            new FileWriter("bot_replay.csv")
                    );
            writer.println(
                    "episode,step,"
                            + "playerX,playerY,playerZ,"
                            + "enemyX,enemyY,enemyZ,"
                            + "distance,direction,stateIndex,"
                            + "action,reward"
            );
            for (int episode = 0; episode < 100; episode++) {
                double totalReward = 0;
                State state = env.getCurrentState();

                for (int step = 0; step < 300; step++) {

                    Action action = agent.chooseAction(state);

                    env.executeAction(action);

                    State nextState = env.getCurrentState();

                    double reward = computeReward(state, action, nextState, prevAction);

                    agent.learn(state, action, reward, nextState);

                    totalReward += reward;
                    System.out.printf(
                            "Action=%s reward=%.3f dist=%d nextDist=%d hit=%b%n",
                            action,
                            reward,
                            state.distance,
                            nextState.distance,
                            env.bot2.wasHit
                    );
                    writer.printf(
                            "%d,%d, %.3f,%.3f,%.3f, %.3f,%.3f,%.3f, %d,%d,%d, %s,%.3f%n",

                            episode, step,

                            env.bot1.Pos.x, env.bot1.Pos.y, env.bot1.Pos.z,

                            env.bot2.Pos.x, env.bot2.Pos.y, env.bot2.Pos.z,

                            nextState.distance,
                            nextState.direction,
                            nextState.toIndex(),

                            action,
                            reward
                            );
                    state = nextState;
                }
                System.out.println("Episode " + episode + " total reward = " + totalReward);
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    static double computeReward(State s, Action a, State s2, Action prevAction) {

        double reward = 0.0;

        // 1. Reward getting closer to FRONT
        if (isCloserToFront(s.direction, s2.direction)) {
            reward += 0.2;
        }

        // 2. Reward getting closer in distance
        if (s2.distance < s.distance) {
            reward += 0.3;
        }

        // 3. Big reward for correct attack
        if (a == Action.ATTACK &&
                s.distance == 0 && // NEAR
                isFacingFront(s.direction)) // FRONT
        {
            reward += 1.5;
            // Critical hit bonus
            if (prevAction == Action.JUMP) {
                reward += 0.7;
            }
        }

        // 4. Penalty for bad attack
        if (a == Action.ATTACK &&
                !(s.distance == 0 && isFacingFront(s.direction))) {
            reward -= 0.5;
        }

        // 5. Discourage spam jump
        if (a == Action.JUMP) {
            reward -= 0.02;
        }

        // 6. Small penalty for turning (encourage efficiency)
        if (a.name().startsWith("TURN")) {
            reward -= 0.05;
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

}