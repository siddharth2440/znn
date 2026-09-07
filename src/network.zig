const std = @import("std");
const Layer = @import("layer.zig").Layer;
const Matrix = @import("matrix.zig").Matrix;
const FowardResult = @import("layer.zig").ForwardResult;

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

    pub fn forward(self: *NeuralNetwork, input: *Matrix) !FowardResult {
        var current = input;

        const activations = try self.allocator.alloc(Matrix, self.layers.len + 1);
        const z_values = try self.allocator.alloc(Matrix, self.layers.len);

        activations[0] = input.*;

        for (self.layers, 0..) |*layer, id| {
            const forward_pass_result = try layer.forward_pass(current);
            z_values[id] = forward_pass_result.z_values;
            activations[id + 1] = forward_pass_result.activations;
            current = &activations[id + 1];
        }

        return .{
            .activations = activations,
            .z_values = z_values,
        };
    }
};
