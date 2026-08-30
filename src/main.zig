const std = @import("std");
const Io = std.Io;

const nn = @import("nn");
const Matrix = nn.Matrix;

pub fn main(init: std.process.Init) !void {
    // try nn.perceptron();

    var arena = init.arena;
    const allocator = arena.allocator();

    var i_mat = try Matrix.identity_matrix(4, allocator);
    std.debug.print("{any}\n", .{i_mat.data});

    i_mat.multiply_by(2);
    std.debug.print("{any}\n", .{i_mat.data});

    var mat = try Matrix.init(2, 3, allocator);

    mat.set_value(0, 0, 11);
    mat.set_value(0, 1, 12);
    mat.set_value(1, 0, 13);
    mat.set_value(1, 1, 14);
    mat.set_value(2, 0, 15);
    mat.set_value(2, 1, 16);

    mat.print_matrix();

    var transpose = try mat.matrix_transpose();
    transpose.print_matrix();
}
