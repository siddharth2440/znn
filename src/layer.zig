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

        std.debug.print("\nweights: {any}\n", .{weights.data});

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

    pub fn forward_pass(self: *Layer, input: *Matrix) !ForwardResult {

        // z1 = w * ip
        var multiplication_result = try self.weights.matrix_multiplication(input) orelse return error.MatrixMultiplicationError;

        // z = z1 + b
        const output = try multiplication_result.matrix_addition(&self.biases) orelse return error.MatrixAdditionError;

        // A = activation(z)
        var activations = try Matrix.init(output.cols, output.rows, self.allocator);
        for (output.data, activations.data) |z_value, *activation_val| {
            activation_val.* = z_value;
        }

        activations.apply_activation_function(.relu);

        // output.apply_activation_function(.relu);

        std.debug.print("Outputs: {any}", .{output.data});

        return .{
            .activations = activations,
            .z_values = output,
        };
    }
};

pub const ForwardResult = struct {
    activations: Matrix,
    z_values: Matrix,
};
