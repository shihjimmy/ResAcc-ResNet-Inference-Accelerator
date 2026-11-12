`timescale 1ns/1ps
//============================================================================
// pe_array -- 16 x 8 PEs, SPEC 6.12
//
//   row r    = output channel r of the current batch (weight broadcast along the row)
//   column p = output pixel p of the current batch   (activation broadcast down the column)
//   read port: one whole row of both banks, for the requantizer
//============================================================================
`default_nettype none
module pe_array (
    input  wire          i_clk,
    input  wire          i_en,
    input  wire          i_bank,
    input  wire          i_first,
    input  wire [127:0]  i_weight,        // 16 x int8, row r = [8r +: 8]
    input  wire [71:0]   i_act,           // 8 x 9-bit signed, column p = [9p +: 9]
    input  wire [319:0]  i_bias,          // 16 x 20-bit signed, row r = [20r +: 20]
    input  wire [3:0]    i_rd_row,        // row read by the requantizer
    output wire [159:0]  o_rd_bank0,      // 8 x 20-bit, column p = [20p +: 20]
    output wire [159:0]  o_rd_bank1
);
    wire [19:0] acc0 [0:127];             // index = r*8 + p
    wire [19:0] acc1 [0:127];

    genvar r, p;
    generate
        for (r = 0; r < 16; r = r + 1) begin : g_row
            for (p = 0; p < 8; p = p + 1) begin : g_col
                pe u_pe (
                    .i_clk    (i_clk),
                    .i_en     (i_en),
                    .i_bank   (i_bank),
                    .i_first  (i_first),
                    .i_weight (i_weight[8*r +: 8]),
                    .i_act    (i_act[9*p +: 9]),
                    .i_bias   (i_bias[20*r +: 20]),
                    .o_acc0   (acc0[r*8+p]),
                    .o_acc1   (acc1[r*8+p])
                );
            end
        end

        for (p = 0; p < 8; p = p + 1) begin : g_rd
            assign o_rd_bank0[20*p +: 20] = acc0[i_rd_row*8 + p];
            assign o_rd_bank1[20*p +: 20] = acc1[i_rd_row*8 + p];
        end
    endgenerate
endmodule
`default_nettype wire
