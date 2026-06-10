package net.nerol.bot_simulator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class QTable {
    private double[][] q;
    private final int numStates = 24;
    private final int numActions = Action.values().length;

    private final double learningRate = 0.1;
    private final double discountFactor = 0.9;

    private final Random random = new Random();

    public QTable() {
        q = new double[numStates][numActions];
    }

    /** Write the Q-table to a plain-text CSV.
     *  Header: numStates,numActions
     *  Then one row per state, each row has numActions comma-separated doubles.
     *  Action ordering follows Action.values(); the mod-side BotAction enum must match. */
    public void save(String path) {
        try (PrintWriter w = new PrintWriter(path)) {
            w.println(numStates + "," + numActions);
            for (int s = 0; s < numStates; s++) {
                StringBuilder sb = new StringBuilder();
                for (int a = 0; a < numActions; a++) {
                    if (a > 0) sb.append(',');
                    sb.append(q[s][a]);
                }
                w.println(sb);
            }
        } catch (IOException e) {
            System.out.printf("Could not save QTable to %s%n", path);
            e.printStackTrace();
        }
    }

    /** Counterpart to {@link #save}. Restores Q-values; the file's dimensions must
     *  match the current numStates/numActions. */
    public void load(String path) {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String[] header = r.readLine().split(",");
            int fileStates = Integer.parseInt(header[0]);
            int fileActions = Integer.parseInt(header[1]);
            if (fileStates != numStates || fileActions != numActions) {
                throw new IllegalStateException(
                        "QTable dimension mismatch: file " + fileStates + "x" + fileActions
                                + ", expected " + numStates + "x" + numActions);
            }
            for (int s = 0; s < numStates; s++) {
                String[] cells = r.readLine().split(",");
                for (int a = 0; a < numActions; a++) {
                    q[s][a] = Double.parseDouble(cells[a]);
                }
            }
        } catch (IOException e) {
            System.out.printf("Could not load QTable from %s%n", path);
            e.printStackTrace();
        }
    }

    public Action getBestAction(int stateIndex) {
        double maxQ = Double.NEGATIVE_INFINITY;
        int bestAction = 0;

        for (int a = 0; a < numActions; a++) {
            if (q[stateIndex][a] > maxQ) {
                maxQ = q[stateIndex][a];
                bestAction = a;
            }
        }
        return Action.values()[bestAction];
    }

    public Action getActionEpsilonGreedy(int stateIndex, double epsilon) {
        if (random.nextDouble() < epsilon) {
            return Action.values()[random.nextInt(numActions)];
        } else {
            return getBestAction(stateIndex);
        }
    }

    public void update(int state, int action, double reward, int nextState) {
        double maxNextQ = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < numActions; a++) {
            maxNextQ = Math.max(maxNextQ, q[nextState][a]);
        }

        q[state][action] =
                q[state][action]
                        + learningRate * (reward + discountFactor * maxNextQ - q[state][action]);
    }
}