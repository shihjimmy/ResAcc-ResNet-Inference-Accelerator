`timescale 1ns/1ps
//============================================================================
// sram_sp -- single-port SRAM behavioral model
//
// Mimics a compiled SRAM macro. Every buffer in the design goes through this
// module so it can later be replaced by real macros without touching the rest.
//   * read latency : 1 cycle (address in cycle t, o_rdata valid in cycle t+1)
//   * single port  : one read OR one write per cycle
//   * byte-enable write
//   * contents are not reset (X until written, which exposes read-before-write)
//============================================================================
`default_nettype none
module sram_sp #(
    parameter DEPTH = 64,
    parameter WIDTH = 512,                         // multiple of 8
    parameter AW    = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  wire               i_clk,
    input  wire               i_cs,                // access this cycle
    input  wire               i_we,                // 1: write, 0: read
    input  wire [AW-1:0]      i_addr,
    input  wire [WIDTH-1:0]   i_wdata,
    input  wire [WIDTH/8-1:0] i_wbe,               // byte enable
    output reg  [WIDTH-1:0]   o_rdata
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    // Simulation model only (replaced by a macro in a real flow), so one process
    // with a loop over the bytes is fine and keeps the simulation fast.
    integer b;

    always @(posedge i_clk) begin
        if (i_cs) begin
            if (i_we) begin
                for (b = 0; b < WIDTH/8; b = b + 1)
                    if (i_wbe[b]) mem[i_addr][b*8 +: 8] <= i_wdata[b*8 +: 8];
            end else begin
                o_rdata <= mem[i_addr];
            end
        end
    end

`ifndef SYNTHESIS
    always @(posedge i_clk) begin
        if (i_cs && i_addr >= DEPTH) begin
            $display("[%0t] ERROR %m: addr %0d >= DEPTH %0d", $time, i_addr, DEPTH);
            $stop;
        end
    end
`endif
endmodule
`default_nettype wire
