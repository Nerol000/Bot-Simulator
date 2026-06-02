package net.nerol.bot_simulator.minecraft.util;

public class Vec3 {
    public double x, y, z;

    public Vec3() {
        this(0, 0, 0);
    }

    public Vec3(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public double horizontalLength() {
        return Math.sqrt(x * x + z * z);
    }

    public void scale(double s) {
        x *= s;
        y *= s;
        z *= s;
    }
}

