`timescale 1ns/1ps
//============================================================================
// requant_lane -- one requantization lane (one multiplier), SPEC 6.14
//
//   plain layer   : one cycle per output
//                     y = clamp((acc * m + 2^(s-1)) >>> s, 0, 255)
//   residual layer: two cycles per output
//                     cycle 1: part = acc_sc * m_sc + 2^(s-1)      (i_out = 0)
//                     cycle 2: y = clamp((acc_b * m_b + part) >>> s)   (i_add = 1, i_out = 1)
//   s is 23 or 24 only (i_s24), so the rounding shift is a 2-way choice.
//
// Latency: 2 cycles (stage 1 multiply, stage 2 add / round / clamp).
//============================================================================
`default_nettype none
module requant_lane (
    input  wire               i_clk,
    input  wire               i_rst_n,
    input  wire               i_valid,
    input  wire signed [19:0] i_acc,
    input  wire        [15:0] i_m,
    input  wire               i_add,          // add the stored partial product
    input  wire               i_out,          // produce an output (else store the partial)
    input  wire               i_s24,          // shift = 24 (else 23)
    output wire               o_valid,
    output wire        [7:0]  o_y
);
    //------------------------------------------------------------------------
    // stage 1: multiply
    //------------------------------------------------------------------------
    reg               valid_1_r;
    reg               add_1_r, out_1_r, s24_1_r;
    reg signed [36:0] prod_1_r;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) valid_1_r <= 1'b0;
        else          valid_1_r <= i_valid;
    end

    always @(posedge i_clk) begin
        add_1_r  <= i_add;
        out_1_r  <= i_out;
        s24_1_r  <= i_s24;
        prod_1_r <= i_acc * $signed({1'b0, i_m});
    end

    //------------------------------------------------------------------------
    // stage 2: add the partial, round, shift, clamp
    //------------------------------------------------------------------------
    reg               valid_2_r;
    reg        [7:0]  y_2_r;
    reg signed [37:0] part_r;

    // The rounding constant 2^(s-1) is folded into the single adder:
    //   plain / residual phase 1 : sum = prod + rnd          (part stores prod_sc + rnd)
    //   residual phase 2         : sum = prod + part         (rnd already inside part)
    // so stage 2 has only ONE 38-bit adder; the >>> s is plain wiring.
    wire signed [37:0] rnd     = s24_1_r ? 38'sd8388608 : 38'sd4194304;  // 2^23 or 2^22
    wire signed [37:0] sum     = prod_1_r + (add_1_r ? part_r : rnd);
    wire signed [37:0] shifted = s24_1_r ? (sum >>> 24) : (sum >>> 23);
    wire               neg     = sum[37];
    wire               big     = s24_1_r ? (|sum[36:32]) : (|sum[36:31]);  // value > 255
    wire        [7:0]  clamped = neg ? 8'd0 : (big ? 8'd255 : shifted[7:0]);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) valid_2_r <= 1'b0;
        else          valid_2_r <= valid_1_r && out_1_r;
    end

    always @(posedge i_clk) begin
        if (valid_1_r && !out_1_r) part_r <= sum;
        y_2_r <= clamped;
    end

    assign o_valid = valid_2_r;
    assign o_y     = y_2_r;
endmodule
`default_nettype wire
