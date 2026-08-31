const std = @import("std");

pub const Matrix = struct {
    rows: u32,
    cols: u32,
    data: []u32,
    allocator: std.mem.Allocator,

    pub fn init(cols: u32, rows: u32, allocator: std.mem.Allocator) !Matrix {
        const data = try allocator.alloc(u32, rows * cols);

        return .{
            .cols = cols,
            .rows = rows,
            .data = data,
            .allocator = allocator,
        };
    }

    pub inline fn get_value(self: *Matrix, row: u32, col: u32) u32 {
        return self.data[(self.cols * row) + col];
    }

    pub inline fn set_value(self: *Matrix, row: u32, col: u32, value: u32) void {
        self.data[(self.cols * row) + col] = value;
    }

    pub fn multiply_by(self: *Matrix, num: u32) void {
        for (self.data) |*value| {
            value.* *= num;
        }
    }

    pub fn zero_matrix(rows: u32, cols: u32, allocator: std.mem.Allocator) !void {
        const z_mat = try Matrix.init(cols, rows, allocator);
        @memset(z_mat.data, 0);
    }

    pub fn identity_matrix(size: u32, allocator: std.mem.Allocator) !Matrix {
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

    pub fn matrix_addition(self: *Matrix, with: *Matrix) !?Matrix {
        if (self.rows == with.rows and self.cols == with.cols) {
            var result_matrix = try Matrix.init(self.cols, self.rows, self.allocator);

            var row: u32 = 0;
            while (row < self.rows) : (row += 1) {
                var col: u32 = 0;
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

            var row: u32 = 0;
            while (row < self.rows) : (row += 1) {
                var col: u32 = 0;
                while (col < with.cols) : (col += 1) {
                    var size: u32 = 0;
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
        var row: u32 = 0;
        while (row < self.rows) : (row += 1) {
            var col: u32 = 0;
            while (col < self.cols) : (col += 1) {
                const val = self.get_value(row, col);
                std.debug.print("{d} \t", .{val});
            }

            std.debug.print("\n", .{});
        }
    }
};
