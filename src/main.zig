const std = @import("std");
const Io = std.Io;

const nn = @import("network.zig").NeuralNetwork;
const Matrix = @import("matrix.zig").Matrix;
const Layer = @import("layer.zig").Layer;
const Loss = @import("loss.zig");

pub fn main(init: std.process.Init) !void {
    var arena = init.arena;
    const allocator = arena.allocator();

    const io = init.io;
    var input = try Matrix.init(1, 2, allocator);
    input.set_value(0, 0, 2.0);
    input.set_value(1, 0, 3.0);

    var target = try Matrix.init(1, 1, allocator);
    target.set_value(0, 0, 10.0);

    //  This means:
    //      2 input values
    //      First layer has 3 neurons
    //      Second layer has 4 neurons
    //      Output layer has 1 neuron
    var neural_network = try nn.init(allocator, io, &[_]usize{ 2, 3, 4, 1 });

    const learning_rate: f32 = 0.003;
    const epochs: usize = 5;

    // --- Forward pass before training ---
    const before = try neural_network.forward(&input);
    const loss_before = try Loss.calculate_loss_func(&before.activations[before.activations.len - 1], &target, .mse);
    std.debug.print("\nPrediction before training : {any}", .{before.activations[before.activations.len - 1].data});
    std.debug.print("\nLoss before training       : {any}", .{loss_before});

    // --- Train: repeat forward + backward, updating all weights each step ---
    var prev_prediction: f32 = before.activations[before.activations.len - 1].get_value(0, 0);
    for (0..epochs) |epoch| {
        _ = try neural_network.backward(&input, &target, learning_rate);

        const curr = try neural_network.forward(&input);
        const loss = try Loss.calculate_loss_func(&curr.activations[curr.activations.len - 1], &target, .mse);
        const prediction = curr.activations[curr.activations.len - 1].get_value(0, 0);

        std.debug.print("\nepoch {} -> prediction: {d}, loss: {d}", .{ epoch + 1, prediction, loss });
        prev_prediction = prediction;
    }

    const after = try neural_network.forward(&input);
    const loss_after = try Loss.calculate_loss_func(&after.activations[after.activations.len - 1], &target, .mse);
    std.debug.print("\nLoss decreased             : {any}", .{loss_after < loss_before});
    std.debug.print("\nLoss decreased by          : {any} {any}", .{ loss_after, loss_before });
}
