// Black-box stand-in for sram_sp, used only by Yosys.
// SRAMs become memory macros in a real flow, so their area is counted separately (SPEC 8.2).
(* blackbox *)
module sram_sp #(parameter DEPTH = 64, parameter WIDTH = 512, parameter AW = 6) (
    input  wire             i_clk,
    input  wire             i_cs,
    input  wire             i_we,
    input  wire [AW-1:0]    i_addr,
    input  wire [WIDTH-1:0] i_wdata,
    input  wire [WIDTH/8-1:0] i_wbe,
    output wire [WIDTH-1:0] o_rdata
);
endmodule
