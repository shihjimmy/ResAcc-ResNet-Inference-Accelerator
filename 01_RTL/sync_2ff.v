`timescale 1ns/1ps
//============================================================================
// sync_2ff -- two-flop synchronizer
//
// Only for single-bit levels or Gray-coded values (at most one bit changes
// per source clock). Latency: 2 destination clock cycles.
//============================================================================
`default_nettype none
module sync_2ff #(
    parameter WIDTH = 1
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire [WIDTH-1:0] i_d,          // from another clock domain
    output wire [WIDTH-1:0] o_q           // synchronized to i_clk
);
    reg [WIDTH-1:0] meta_r;               // may go metastable; settles within one cycle
    reg [WIDTH-1:0] sync_r;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            meta_r <= {WIDTH{1'b0}};
            sync_r <= {WIDTH{1'b0}};
        end else begin
            meta_r <= i_d;
            sync_r <= meta_r;
        end
    end

    assign o_q = sync_r;
endmodule
`default_nettype wire
