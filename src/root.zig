const std = @import("std");
const Io = std.Io;

pub const Matrix = struct {
    rows: u8,
    cols: u8,
    data: []u8,
    allocator: std.mem.Allocator,

    pub fn init(cols: u8, rows: u8, allocator: std.mem.Allocator) !Matrix {
        const data = try allocator.alloc(u8, rows * cols);

        return .{
            .cols = cols,
            .rows = rows,
            .data = data,
            .allocator = allocator,
        };
    }

    pub inline fn get_value(self: *Matrix, row: u32, col: u32) u8 {
        return self.data[(self.cols * row) + col];
    }

    pub inline fn set_value(self: *Matrix, row: u32, col: u32, value: u8) void {
        self.data[(self.cols * row) + col] = value;
    }

    pub fn multiply_by(self: *Matrix, num: u8) void {
        for (self.data) |*value| {
            value.* *= num;
        }
    }

    pub fn zero_matrix(rows: u8, cols: u8, allocator: std.mem.Allocator) !void {
        const z_mat = try Matrix.init(cols, rows, allocator);
        @memset(z_mat.data, 0);
    }

    pub fn identity_matrix(size: u8, allocator: std.mem.Allocator) !Matrix {
        var i_mat = try Matrix.init(size, size, allocator);
        @memset(i_mat.data, 0);

        var i: u32 = 0;
        while (i < size) : (i += 1) {
            i_mat.set_value(i, i, 1);
        }

        return i_mat;
    }

    pub fn matrix_transpose(self: *Matrix) !Matrix {
        var transpose_matrix = try Matrix.init(self.rows, self.cols, self.allocator);

        var row: u32 = 0;
        while (row < self.rows) : (row += 1) {
            var col: u32 = 0;
            while (col < self.cols) : (col += 1) {
                const val = self.get_value(row, col);
                transpose_matrix.set_value(col, row, val);
            }
        }

        return transpose_matrix;
    }

    pub fn print_matrix(self: *Matrix) void {
        var row: u32 = 0;
        while (row < self.rows) : (row += 1) {
            var col: u32 = 0;
            while (col < self.cols) : (col += 1) {
                self.get_value(row, col);
            }

            std.debug.print("\n", .{});
        }
    }
};

pub fn perceptron() !void {
    const x1: u32 = 2;
    const x2: u32 = 3;

    const w1: f32 = 0.5;
    const w2: f32 = -0.3;

    const b: f32 = 0.1;

    const y = x1 * w1 + x2 * w2 + b;

    const activatn_fn = 1 / (1 + std.math.exp(-y));

    std.debug.print("\nop = {any}\n", .{activatn_fn});
}
