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
    const forward_result = try neural_network.forward(&input);

    std.debug.print("\n-- {any} --\n", .{forward_result.activations});
    std.debug.print("\n-- {any} --\n", .{forward_result.z_values});

    const loss = try Loss.calculate_loss_func(
        &forward_result.activations,
        &target,
        .mse,
    );
    std.debug.print("\n-- Loss: {any} --\n", .{loss});
}
