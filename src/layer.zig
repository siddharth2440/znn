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

    // Backward pass for one layer.
    //
    //   delta_next : error arriving at THIS layer's z, from the layer in front.
    //                For the output layer this is dL/dy = 2*(y_hat - y)/n.
    //   prev       : this layer's input activations (a^(l-1)).
    //   z_this     : this layer's pre-activation values, needed to open the
    //                activation gate (relu prime = 1 if z > 0 else 0).
    //   learning_rate : eta, used to update weights and biases in place.
    //
    // Returns the error signal that continues flowing toward the previous layer
    // (in BackwardResult.delta_prev), plus the computed gradients.
    pub fn backward_pass(self: *Layer, prev: *Matrix, delta_next: *Matrix, z_this: *Matrix, learning_rate: f32) !BackwardResult {
        const allocator = self.allocator;

        // 1) Open the activation gate:
        //    delta = delta_next (x) relu'(z_this)
        //    relu'(z) = 1 if z > 0, else 0.
        var delta = try Matrix.init(delta_next.cols, delta_next.rows, allocator);
        for (delta_next.data, z_this.data, delta.data) |dn, z, *d| {
            d.* = if (z > 0.0) dn else 0.0;
        }

        // 2) Weight gradient:  dW = delta (x) prev^T          (out x in)
        var prev_T = try prev.matrix_transpose();
        const weight_gradient = try delta.matrix_multiplication(&prev_T) orelse return error.MatrixMultiplicationError;

        // 3) Bias gradient is simply delta.
        const bias_gradient = delta;

        // 4) Update parameters in place:  theta -= learning_rate * grad
        for (self.weights.data, weight_gradient.data) |*w, dw| {
            w.* -= learning_rate * dw;
        }
        for (self.biases.data, delta.data) |*b, db| {
            b.* -= learning_rate * db;
        }

        // 5) Propagate the error one layer deeper:
        //    delta_prev = W^T (x) delta          (in x 1, before relu gate)
        var W_T = try self.weights.matrix_transpose();
        const delta_prev = try W_T.matrix_multiplication(&delta) orelse return error.MatrixMultiplicationError;

        return .{
            .delta_prev = delta_prev,
            .weight_gradient = weight_gradient,
            .bias_gradient = bias_gradient,
        };
    }
};

pub const ForwardResult = struct {
    activations: Matrix,
    z_values: Matrix,
};

pub const BackwardResult = struct {
    // Error signal propagated one layer deeper (into the previous layer's z),
    // BEFORE that layer's activation gate is applied.
    delta_prev: Matrix,
    // Gradient of the loss with respect to this layer's weights (out x in).
    weight_gradient: Matrix,
    // Gradient of the loss with respect to this layer's biases (out x 1).
    bias_gradient: Matrix,
};
