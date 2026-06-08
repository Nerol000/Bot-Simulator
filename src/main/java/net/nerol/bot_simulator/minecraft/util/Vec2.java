package net.nerol.bot_simulator.minecraft.util;

public class Vec2 {
    public static final Vec2 ZERO_VEC2 = new Vec2();

    public double x;
    public double y;

    public Vec2() {
        this(0, 0);
    }

    public Vec2(final double x, final double y) {
        this.x = x;
        this.y = y;
    }

    public double magnitude() {
        return Math.sqrt(x*x + y*y);
    }

    public Vec2 scale(final float s) {
        return new Vec2(this.x * s, this.y * s);
    }

    public double dot(final Vec2 v) {
        return this.x * v.x + this.y * v.y;
    }

    public Vec2 add(final Vec2 rhs) {
        return new Vec2(this.x + rhs.x, this.y + rhs.y);
    }

    public Vec2 add(final float v) {
        return new Vec2(this.x + v, this.y + v);
    }

    public boolean equals(final Vec2 rhs) {
        return this.x == rhs.x && this.y == rhs.y;
    }

    public Vec2 negated() {
        return new Vec2(-this.x, -this.y);
    }

    public String to_string() {
        return "(" + this.x + ", " + this.y + ")";
    }
}
