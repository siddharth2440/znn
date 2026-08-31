const std = @import("std");
const Io = std.Io;

pub fn perceptron() !void {
    const x1: u32 = 2;
    const x2: u32 = 3;

    const w1: f32 = 0.5;
    const w2: f32 = -0.3;

    const b: f32 = 0.1;

    const y = x1 * w1 + x2 * w2 + b;

    const activatn_fn = 1 / (1 + std.math.exp(-y));

    std.debug.print("\nop = {any}\n", .{activatn_fn});
}
