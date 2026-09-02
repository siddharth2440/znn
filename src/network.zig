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
