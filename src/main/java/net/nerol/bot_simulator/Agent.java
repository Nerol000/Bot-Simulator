package net.nerol.bot_simulator;

public class Agent {
    private QTable qTable;
    private double epsilon = 0.2;

    public Agent(QTable qTable) {
        this.qTable = qTable;
    }

    public Action chooseAction(State state) {
        int stateIndex = state.toIndex();
        return qTable.getActionEpsilonGreedy(stateIndex, epsilon);
    }

    public void learn(State currentState, Action action, double reward, State nextState) {
        int s = currentState.toIndex();
        int a = action.ordinal();
        int s2 = nextState.toIndex();

        qTable.update(s, a, reward, s2);
    }
}

