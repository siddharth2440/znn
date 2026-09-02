const std = @import("std");

pub const Matrix = struct {
    rows: usize,
    cols: usize,
    data: []f32,
    allocator: std.mem.Allocator,

    pub fn init(cols: usize, rows: usize, allocator: std.mem.Allocator) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);

        return .{
            .cols = cols,
            .rows = rows,
            .data = data,
            .allocator = allocator,
        };
    }

    pub inline fn get_value(self: *Matrix, row: usize, col: usize) f32 {
        return self.data[(self.cols * row) + col];
    }

    pub inline fn set_value(self: *Matrix, row: usize, col: usize, value: f32) void {
        self.data[(self.cols * row) + col] = value;
    }

    pub fn multiply_by(self: *Matrix, num: usize) void {
        for (self.data) |*value| {
            value.* *= num;
        }
    }

    pub fn zero_matrix(rows: usize, cols: usize, allocator: std.mem.Allocator) !void {
        const z_mat = try Matrix.init(cols, rows, allocator);
        @memset(z_mat.data, 0.0);
    }

    pub fn fill(self: *Matrix, val: f32) void {
        @memset(self.data, val);
    }

    pub fn fill_random_values(self: *Matrix, io: std.Io) void {
        const rng_impl = std.Random.IoSource{ .io = io };
        var random = rng_impl.interface();

        for (self.data) |*data| {
            data.* = random.float(f32);
        }
    }

    pub fn identity_matrix(size: usize, allocator: std.mem.Allocator) !Matrix {
        var i_mat = try Matrix.init(size, size, allocator);
        @memset(i_mat.data, 0);

        var i: usize = 0;
        while (i < size) : (i += 1) {
            i_mat.set_value(i, i, 1);
        }

        return i_mat;
    }

    pub fn matrix_transpose(self: *Matrix) !Matrix {
        var transpose_matrix = try Matrix.init(self.rows, self.cols, self.allocator);

        var row: usize = 0;
        while (row < self.rows) : (row += 1) {
            var col: usize = 0;
            while (col < self.cols) : (col += 1) {
                const val = self.get_value(row, col);
                transpose_matrix.set_value(col, row, val);
            }
        }

        return transpose_matrix;
    }

    pub fn matrix_addition(self: *Matrix, with: *Matrix) !?Matrix {
        if (self.rows == with.rows and self.cols == with.cols) {
            var result_matrix = try Matrix.init(self.cols, self.rows, self.allocator);

            var row: usize = 0;
            while (row < self.rows) : (row += 1) {
                var col: usize = 0;
                while (col < self.cols) : (col += 1) {
                    const addtn_result = self.get_value(row, col) + with.get_value(row, col);

                    result_matrix.set_value(row, col, addtn_result);
                }
            }

            return result_matrix;
        } else {
            std.debug.print("Error: Dimensions should be identical. \n", .{});
            return null;
        }
    }

    pub fn matrix_multiplication(self: *Matrix, with: *Matrix) !?Matrix {
        if (self.cols == with.rows) {
            var resultant_matrix = try Matrix.init(with.cols, self.rows, self.allocator);
            @memset(resultant_matrix.data, 0);

            var row: usize = 0;
            while (row < self.rows) : (row += 1) {
                var col: usize = 0;
                while (col < with.cols) : (col += 1) {
                    var size: usize = 0;
                    while (size < self.cols) : (size += 1) {
                        const cur_val = self.get_value(row, size);
                        const other_matrix_val = with.get_value(size, col);
                        const product = cur_val * other_matrix_val;
                        resultant_matrix.set_value(row, col, resultant_matrix.get_value(row, col) + product);
                    }
                }
            }

            return resultant_matrix;
        } else {
            std.debug.print("Error: Matrix Dimension error", .{});
            return null;
        }
    }

    pub fn print_matrix(self: *Matrix) void {
        var row: usize = 0;
        while (row < self.rows) : (row += 1) {
            var col: usize = 0;
            while (col < self.cols) : (col += 1) {
                const val = self.get_value(row, col);
                std.debug.print("{d} \t", .{val});
            }

            std.debug.print("\n", .{});
        }
    }

    pub fn apply_activation_function(self: *Matrix, activation: ActivationType) void {
        for (self.data) |*val| {
            val.* = apply(activation, val.*);
        }
    }
};

// ----------------------   Activation Functions    ----------------------
pub const ActivationType = enum {
    none,

    relu,
    leaky_relu,

    sigmoid, // Used for Binary Classification Output.
    tanh,

    softmax,
};

pub fn apply(atype: ActivationType, data: f32) f32 {
    switch (atype) {
        .none => {
            return data;
        },
        .relu => {
            return @max(0.0, data);
        },
        .leaky_relu => {
            const alpha: f32 = 0.01;
            return if (data > 0) {
                return data;
            } else {
                return alpha * data;
            };
        },
        .sigmoid => {
            return 1.0 / 1.0 + std.math.exp(-data);
        },
        .tanh => {
            return std.math.tanh(data);
        },
        .softmax => unreachable,
    }
}
