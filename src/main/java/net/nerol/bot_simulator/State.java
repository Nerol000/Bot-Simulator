package net.nerol.bot_simulator;

public class State {
    public int distance;   // 0=NEAR, 1=MID, 2=FAR
    public int direction;  // 0–7 (FRONT, FRONT_RIGHT, ...)

    public State(int distance, int direction) {
        this.distance = distance;
        this.direction = direction;
    }

    public int toIndex() {
        return distance * 8 + direction; // 0–23
    }
}

