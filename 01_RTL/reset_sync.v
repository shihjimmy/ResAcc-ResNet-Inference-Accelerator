`timescale 1ns/1ps
//============================================================================
// reset_sync -- reset synchronizer (asynchronous assert, synchronous release)
//
// One per clock domain, so all flops of a domain leave reset on the same edge.
//============================================================================
`default_nettype none
module reset_sync (
    input  wire i_clk,
    input  wire i_rst_n,                  // external asynchronous reset
    output wire o_rst_n                   // reset for this clock domain
);
    reg [1:0] rst_r;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) rst_r <= 2'b00;
        else          rst_r <= {rst_r[0], 1'b1};
    end

    assign o_rst_n = rst_r[1];
endmodule
`default_nettype wire
