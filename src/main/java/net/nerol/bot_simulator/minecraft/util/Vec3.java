package net.nerol.bot_simulator.minecraft.util;

public class Vec3 {
    public static final Vec3 ZERO_VEC3 = new Vec3();

    public double x, y, z;

    public Vec3() {
        this(0, 0, 0);
    }

    public Vec3(final double x, final double y, final double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public double horizontalLength() {
        return Math.sqrt(x * x + z * z);
    }

    public double magnitude() {
        return Math.sqrt(x*x + y*y + z*z);
    }

    public void scale(double s) {
        x *= s;
        y *= s;
        z *= s;
    }

    public String to_string() {
        return "(" + this.x + ", " + this.y + ", " + this.z + ")";
    }



    public double dot(final Vec3 vec) {
        return this.x * vec.x + this.y * vec.y + this.z * vec.z;
    }

    public Vec3 cross(final Vec3 vec) {
        return new Vec3(this.y * vec.z - this.z * vec.y, this.z * vec.x - this.x * vec.z, this.x * vec.y - this.y * vec.x);
    }

    public Vec3 subtract(final Vec3 vec) {
        return this.subtract(vec.x, vec.y, vec.z);
    }

    public Vec3 subtract(final double value) {
        return this.subtract(value, value, value);
    }

    public Vec3 subtract(final double x, final double y, final double z) {
        return this.add(-x, -y, -z);
    }

    public Vec3 add(final double value) {
        return this.add(value, value, value);
    }

    public Vec3 add(final Vec3 vec) {
        return this.add(vec.x, vec.y, vec.z);
    }

    public Vec3 add(final double x, final double y, final double z) {
        return new Vec3(this.x + x, this.y + y, this.z + z);
    }

    public Vec3 negated() {
        return new Vec3(-this.x, -this.y, -this.z);
    }
}

