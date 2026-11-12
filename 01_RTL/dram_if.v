`timescale 1ns/1ps
//============================================================================
// dram_if -- AXI4 read master (clk_dram domain), SPEC 6.5
//
//   S_IDLE --(request FIFO not empty)--> S_ADDR --(arready)--> S_DATA --(rlast)--> S_IDLE
//
//   * request FIFO entry = {byte address[18:0], arlen[7:0]}, pushed by the loader
//   * every received beat is pushed into the data FIFO; rready = data FIFO not full
//   * one outstanding read at a time; DRAM is never written (tied off in resacc)
//============================================================================
`default_nettype none
module dram_if (
    input  wire         i_clk,
    input  wire         i_rst_n,
    // request FIFO (read side)
    input  wire         i_req_empty,
    input  wire [26:0]  i_req_data,          // {addr[18:0], len[7:0]}
    output wire         o_req_pop,
    // data FIFO (write side)
    input  wire         i_dat_full,
    output wire         o_dat_push,
    output wire [255:0] o_dat_data,
    // AXI AR channel
    output wire [18:0]  o_araddr,
    output wire [7:0]   o_arlen,
    output wire         o_arvalid,
    input  wire         i_arready,
    // AXI R channel
    input  wire [255:0] i_rdata,
    input  wire         i_rlast,
    input  wire         i_rvalid,
    output wire         o_rready
);
    localparam S_IDLE = 2'd0;
    localparam S_ADDR = 2'd1;
    localparam S_DATA = 2'd2;

    reg [1:0]  state_r,   state_w;
    reg [18:0] araddr_r,  araddr_w;
    reg [7:0]  arlen_r,   arlen_w;

    wire req_pop  = (state_r == S_IDLE) && !i_req_empty;
    wire rready   = (state_r == S_DATA) && !i_dat_full;
    wire dat_push = i_rvalid && rready;

    //------------------------------------------------------------------------
    // next state
    //------------------------------------------------------------------------
    always @(*) begin
        state_w  = state_r;
        araddr_w = araddr_r;
        arlen_w  = arlen_r;
        case (state_r)
            S_IDLE: if (req_pop) begin
                        state_w  = S_ADDR;
                        araddr_w = i_req_data[26:8];
                        arlen_w  = i_req_data[7:0];
                    end
            S_ADDR: if (i_arready)          state_w = S_DATA;
            S_DATA: if (dat_push && i_rlast) state_w = S_IDLE;
            default:                         state_w = S_IDLE;
        endcase
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r  <= S_IDLE;
            araddr_r <= 19'd0;
            arlen_r  <= 8'd0;
        end else begin
            state_r  <= state_w;
            araddr_r <= araddr_w;
            arlen_r  <= arlen_w;
        end
    end

    //------------------------------------------------------------------------
    // outputs
    //------------------------------------------------------------------------
    assign o_req_pop  = req_pop;
    assign o_araddr   = araddr_r;
    assign o_arlen    = arlen_r;
    assign o_arvalid  = (state_r == S_ADDR);
    assign o_rready   = rready;
    assign o_dat_push = dat_push;
    assign o_dat_data = i_rdata;
endmodule
`default_nettype wire
