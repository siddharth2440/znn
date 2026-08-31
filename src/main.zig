const std = @import("std");
const Io = std.Io;

const nn = @import("nn");
const Matrix = @import("matrix.zig").Matrix;

pub fn main(init: std.process.Init) !void {
    // try nn.perceptron();

    var arena = init.arena;
    const allocator = arena.allocator();

    // var i_mat = try Matrix.identity_matrix(4, allocator);
    // std.debug.print("{any}\n", .{i_mat.data});

    // i_mat.multiply_by(2);
    // std.debug.print("{any}\n", .{i_mat.data});

    var mat = try Matrix.init(2, 3, allocator);

    mat.set_value(0, 0, 11);
    mat.set_value(0, 1, 12);
    mat.set_value(1, 0, 13);
    mat.set_value(1, 1, 14);
    mat.set_value(2, 0, 15);
    mat.set_value(2, 1, 16);

    // std.debug.print("Mat1: \n", .{});
    // mat.print_matrix();
    // std.debug.print("\n\n", .{});

    // var transpose = try mat.matrix_transpose();
    // transpose.print_matrix();

    var mat2 = try Matrix.init(2, 3, allocator);

    mat2.set_value(0, 0, 11);
    mat2.set_value(0, 1, 12);
    mat2.set_value(1, 0, 13);
    mat2.set_value(1, 1, 14);
    mat2.set_value(2, 0, 15);
    mat2.set_value(2, 1, 16);

    // std.debug.print("Mat2: \n", .{});
    // mat2.print_matrix();
    // std.debug.print("\n\n", .{});

    // var mat_addition_result = try mat.matrix_addition(&mat2) orelse return error.InvalidMatrixDimension;
    // std.debug.print("Matrix Addition: \n", .{});
    // mat_addition_result.print_matrix();

    var mmat1 = try Matrix.init(3, 2, allocator);
    mmat1.set_value(0, 0, 11);
    mmat1.set_value(0, 1, 12);
    mmat1.set_value(0, 2, 13);
    mmat1.set_value(1, 0, 14);
    mmat1.set_value(1, 1, 15);
    mmat1.set_value(1, 2, 16);

    std.debug.print("MMat1 dim: ({d} x {d})\n", .{ mmat1.rows, mmat1.cols });
    mmat1.print_matrix();
    std.debug.print("\n\n", .{});

    var mmat2 = try Matrix.init(2, 3, allocator);

    mmat2.set_value(0, 0, 21);
    mmat2.set_value(0, 1, 22);
    mmat2.set_value(1, 0, 23);
    mmat2.set_value(1, 1, 24);
    mmat2.set_value(2, 0, 25);
    mmat2.set_value(2, 1, 26);

    std.debug.print("MMat2 dim: ({d} x {d})\n", .{ mmat2.rows, mmat2.cols });
    mmat2.print_matrix();
    std.debug.print("\n\n", .{});

    var mat_multiplication_result = try mmat1.matrix_multiplication(&mmat2) orelse return error.InvalidMatrixDimension;
    std.debug.print("Matrix Multiplication: \n", .{});
    mat_multiplication_result.print_matrix();
}
