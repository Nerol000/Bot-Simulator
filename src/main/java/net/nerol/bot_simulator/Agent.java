package net.nerol.bot_simulator;

public class Agent {
    private QTable qTable;

    // Exploration is annealed per episode: start almost fully random so even rarely-reached
    // states get visited and populated, then sharpen toward greedy as the policy converges.
    private double epsilon = 1.0;
    private static final double EPS_MIN = 0.05;
    // Multiplicative per-episode decay. Reaches EPS_MIN after ~ln(EPS_MIN)/ln(EPS_DECAY)
    // episodes (~30k at 0.9999); tune to your training length.
    private static final double EPS_DECAY = 0.9999;

    public Agent(QTable qTable) {
        this.qTable = qTable;
    }

    public Action chooseAction(State state) {
        int stateIndex = state.toIndex();
        return qTable.getActionEpsilonGreedy(stateIndex, epsilon);
    }

    /** Anneal exploration. Call once per episode. */
    public void decayEpsilon() {
        epsilon = Math.max(EPS_MIN, epsilon * EPS_DECAY);
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void learn(State currentState, Action action, double reward, State nextState) {
        int s = currentState.toIndex();
        int a = action.ordinal();
        int s2 = nextState.toIndex();

        qTable.update(s, a, reward, s2);
    }
}

