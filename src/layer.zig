const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

// Layer: 2(inputs) - 3(neurons) - 1(output)
pub const Layer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,

    inputs_size: usize,
    output_size: usize,

    weights: Matrix,
    biases: Matrix,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, input_size: usize, output_size: usize) !Layer {
        var weights = try Matrix.init(input_size, output_size, allocator);
        weights.fill_random_values(io);

        var biases = try Matrix.init(1, output_size, allocator);
        biases.fill(0.0);

        return .{
            .allocator = allocator,
            .io = io,
            .inputs_size = input_size,
            .output_size = output_size,
            .weights = weights,
            .biases = biases,
        };
    }

    pub fn forward_pass(self: *Layer, input: *Matrix) !Matrix {
        var multiplication_result = try self.weights.matrix_multiplication(input) orelse return error.MatrixMultiplicationError;
        var output = try multiplication_result.matrix_addition(&self.biases) orelse return error.MatrixAdditionError;
        output.apply_activation_function(.relu);

        return output;
    }
};
