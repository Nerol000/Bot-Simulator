package net.nerol.bot_simulator;

import java.util.Random;

public class QTable {
    private double[][] q;
    private int numStates = 24;
    private int numActions = Action.values().length;

    private double learningRate = 0.1;
    private double discountFactor = 0.9;

    private Random random = new Random();

    public QTable() {
        q = new double[numStates][numActions];
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
