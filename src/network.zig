const std = @import("std");
const Layer = @import("layer.zig").Layer;
const Matrix = @import("matrix.zig").Matrix;

pub const NeuralNetwork = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    io: std.Io,

    const architectures = [_]usize{
        2, 3, 4, 1,
    };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, arch: []const usize) !NeuralNetwork {
        const no_of_layers = arch.len - 1;
        var layers = try allocator.alloc(Layer, no_of_layers);

        for (0..no_of_layers) |id| {
            layers[id] = try Layer.init(allocator, io, arch[id], arch[id + 1]);
        }

        return .{
            .allocator = allocator,
            .io = io,
            .layers = layers,
        };
    }

    pub fn forward(self: *NeuralNetwork, input: *Matrix) !Matrix {
        var current = input;

        for (self.layers) |*layer| {
            var output = try layer.forward_pass(current);

            current = &output;
        }

        return current.*;
    }
};

pub const LossFunction = enum {
    mse,
    // binary_cross_entropy,
    // categorical_cross_entropy,
};

pub fn mse(prediction: *const Matrix, target: *const Matrix) !f32 {
    if ((prediction.cols != target.cols) || (prediction.rows != target.rows)) {
        return error.InvalidMatrixDimensions;
    }

    // mse = prediction - output
    var total: f32 = 0.0;
    for (prediction.data, target.data) |prediction_val, target_val| {
        const error_val = prediction_val - target_val;
        total += std.math.pow(f32, error_val, 2);
    }

    return @as(f32, @floatFromInt(total / prediction.data.len));
}

pub fn calculate_loss_func(pred: *const Matrix, target: *const Matrix, loss: LossFunction) !f32 {
    return switch (loss) {
        .mse => mse(pred, target),
        else => unreachable,
    };
}
