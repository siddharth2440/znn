const std = @import("std");
const matrix = @import("matrix.zig");

pub const LossFunction = enum {
    mse,
    // binary_cross_entropy,
    // categorical_cross_entropy,
};

pub fn mse(prediction: *const matrix.Matrix, target: *const matrix.Matrix) !f32 {
    if ((prediction.cols != target.cols) or (prediction.rows != target.rows)) {
        return error.InvalidMatrixDimensions;
    }

    // mse = prediction - output
    var total: f32 = 0.0;
    for (prediction.data, target.data) |prediction_val, target_val| {
        const error_val = prediction_val - target_val;
        total += error_val * error_val;
    }

    return total / @as(f32, @floatFromInt(prediction.data.len));
}

pub fn calculate_loss_func(pred: *const matrix.Matrix, target: *const matrix.Matrix, loss: LossFunction) !f32 {
    return switch (loss) {
        .mse => try mse(pred, target),
    };
}

pub fn derivative(prediction: *const matrix.Matrix, target: *const matrix.Matrix, allocator: std.mem.Allocator) !void {
    if (prediction.cols != target.cols or target.rows != prediction.rows) {
        return error.InvliadMatrixDimensions;
    }

    var gradient = try matrix.Matrix.init(prediction.cols, prediction.rows, allocator);
    try gradient.fill(0);

    const n: f32 = @floatFromInt(prediction.data.len);

    for (prediction.data, target.data, gradient.data) |prediction_val, target_val, *gradient_val| {
        gradient_val.* = (2.0 / n) * (prediction_val - target_val);
    }

    return gradient;
}
