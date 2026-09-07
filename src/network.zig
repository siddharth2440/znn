const std = @import("std");
const Layer = @import("layer.zig").Layer;
const Matrix = @import("matrix.zig").Matrix;
const BackwardResult = @import("layer.zig").BackwardResult;

// Whole-network forward result: one copy of the input, plus each layer's
// z-values and activations, indexed like the forward() loop below.
pub const NetworkForward = struct {
    activations: []Matrix,
    z_values: []Matrix,
};

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

    pub fn forward(self: *NeuralNetwork, input: *Matrix) !NetworkForward {
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

    // Backward propagation: run the forward pass, then push the loss gradient
    // back through every layer (last -> first), updating all weights and biases.
    //
    //   target : the desired output (same shape as the network output).
    //   learning_rate : eta.
    //
    // Returns the per-layer gradients, in case the caller wants to inspect them.
    pub fn backward(self: *NeuralNetwork, input: *Matrix, target: *Matrix, learning_rate: f32) ![]BackwardResult {
        const L = self.layers.len;

        // Recompute forward pass so we have activations and z-values available.
        const fwd = try self.forward(input);
        const activations = fwd.activations; // [0..L], a[L] is the prediction
        const z_values = fwd.z_values;       // [0..L-1]

        // Kick off the error at the output: dL/dy = 2*(y_hat - y)/n  (MSE derivative).
        const output = &activations[L];
        const n: f32 = @floatFromInt(output.data.len);
        var delta_next = try Matrix.init(output.cols, output.rows, self.allocator);
        for (output.data, target.data, delta_next.data) |pred, tgt, *d| {
            d.* = (2.0 / n) * (pred - tgt);
        }

        const gradients = try self.allocator.alloc(BackwardResult, L);

        // Walk the layers from output to input.
        for (0..L) |rev| {
            const layer_id = L - 1 - rev;

            // prev = activations[layer_id]  (the input feeding this layer)
            // z    = z_values[layer_id]     (this layer's pre-activation)
            gradients[layer_id] = try self.layers[layer_id].backward_pass(
                &activations[layer_id],
                &delta_next,
                &z_values[layer_id],
                learning_rate,
            );
            delta_next = gradients[layer_id].delta_prev;
        }

        return gradients;
    }
};
