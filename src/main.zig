const std = @import("std");
const Io = std.Io;

const nn = @import("network.zig").NeuralNetwork;
const Matrix = @import("matrix.zig").Matrix;
const Layer = @import("layer.zig").Layer;

pub fn main(init: std.process.Init) !void {
    var arena = init.arena;
    const allocator = arena.allocator();

    const io = init.io;
    var input = try Matrix.init(1, 2, allocator);
    input.set_value(0, 0, 2.0);
    input.set_value(1, 0, 3.0);

    // var single_layer = try Layer.init(allocator, io, 2, 3);
    // var output = try single_layer.forward_pass(&input);

    // output.print_matrix();

    //  This means:
    //      2 input values
    //      First layer has 3 neurons
    //      Second layer has 4 neurons
    //      Output layer has 1 neuron
    var neural_network = try nn.init(allocator, io, &[_]usize{ 2, 3, 4, 1 });
    const output_matrix = try neural_network.forward(&input);

    std.debug.print("{any}", .{output_matrix.data});
}
