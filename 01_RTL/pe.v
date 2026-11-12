`timescale 1ns/1ps
//============================================================================
// pe -- one processing element: 8 x 9 signed multiply, accumulate into one of
//       two accumulator banks, SPEC 6.12
//
//   i_first = 1 : bank <= bias + product   (no separate clear)
//   i_first = 0 : bank <= bank + product
// Accumulators are data registers (no reset): every batch starts with i_first.
//============================================================================
`default_nettype none
module pe (
    input  wire               i_clk,
    input  wire               i_en,           // valid cycle
    input  wire               i_bank,         // accumulator bank to update
    input  wire               i_first,        // first cycle of a batch: start from the bias
    input  wire signed [7:0]  i_weight,
    input  wire signed [8:0]  i_act,
    input  wire signed [19:0] i_bias,
    output wire signed [19:0] o_acc0,
    output wire signed [19:0] o_acc1
);
    reg  signed [19:0] acc0_r, acc0_w;
    reg  signed [19:0] acc1_r, acc1_w;

    wire signed [16:0] prod = i_weight * i_act;
    wire signed [19:0] base = i_first ? i_bias : (i_bank ? acc1_r : acc0_r);   // select the bank first
    wire signed [19:0] sum  = base + prod;                                       // one adder

    always @(*) begin
        acc0_w = acc0_r;
        acc1_w = acc1_r;
        if (i_en) begin
            if (i_bank) acc1_w = sum;
            else        acc0_w = sum;
        end
    end

    always @(posedge i_clk) begin
        acc0_r <= acc0_w;
        acc1_r <= acc1_w;
    end

    assign o_acc0 = acc0_r;
    assign o_acc1 = acc1_r;
endmodule
`default_nettype wire
