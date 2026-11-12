`timescale 1ns/1ps
//============================================================================
// async_fifo -- dual-clock FIFO
//
// Only the read/write pointers cross clock domains:
//   * pointers are one bit wider than the address (to tell full from empty)
//   * pointers are converted to Gray code, then synchronized with sync_2ff
//   * empty: read pointer == synchronized write pointer
//   * full : write pointer == synchronized read pointer with the top two bits inverted
// The synchronized pointer is always old, so full/empty are conservative:
// the FIFO may look full/empty a little too long, but never overflows or underflows.
//
// Read side is first-word-fall-through: when o_rempty = 0, o_rdata already holds
// the oldest entry; pulse i_rinc for one cycle to consume it.
//============================================================================
`default_nettype none
module async_fifo #(
    parameter WIDTH  = 256,
    parameter ADDR_W = 4                       // depth = 2^ADDR_W
) (
    // write side
    input  wire              i_wclk,
    input  wire              i_wrst_n,
    input  wire              i_winc,           // caller guarantees o_wfull = 0
    input  wire [WIDTH-1:0]  i_wdata,
    output wire              o_wfull,
    // read side
    input  wire              i_rclk,
    input  wire              i_rrst_n,
    input  wire              i_rinc,           // caller guarantees o_rempty = 0
    output wire [WIDTH-1:0]  o_rdata,
    output wire              o_rempty
);
    localparam DEPTH = 1 << ADDR_W;

    reg [WIDTH-1:0] mem [0:DEPTH-1];

    //------------------------------------------------------------------------
    // write side
    //------------------------------------------------------------------------
    reg  [ADDR_W:0] wbin_r,  wbin_w;
    reg  [ADDR_W:0] wgray_r, wgray_w;
    reg             wfull_r, wfull_w;
    wire [ADDR_W:0] rgray_sync;                // read pointer seen by the write side
    wire            wpush = i_winc && !wfull_r;

    always @(*) begin
        wbin_w  = wbin_r + {{ADDR_W{1'b0}}, wpush};
        wgray_w = (wbin_w >> 1) ^ wbin_w;
        wfull_w = (wgray_w == {~rgray_sync[ADDR_W:ADDR_W-1], rgray_sync[ADDR_W-2:0]});
    end

    always @(posedge i_wclk or negedge i_wrst_n) begin
        if (!i_wrst_n) begin
            wbin_r  <= {(ADDR_W+1){1'b0}};
            wgray_r <= {(ADDR_W+1){1'b0}};
            wfull_r <= 1'b0;
        end else begin
            wbin_r  <= wbin_w;
            wgray_r <= wgray_w;
            wfull_r <= wfull_w;
        end
    end

    always @(posedge i_wclk) begin
        if (wpush) mem[wbin_r[ADDR_W-1:0]] <= i_wdata;
    end

    assign o_wfull = wfull_r;

    //------------------------------------------------------------------------
    // read side
    //------------------------------------------------------------------------
    reg  [ADDR_W:0] rbin_r,  rbin_w;
    reg  [ADDR_W:0] rgray_r, rgray_w;
    reg             rempty_r, rempty_w;
    wire [ADDR_W:0] wgray_sync;                // write pointer seen by the read side
    wire            rpop = i_rinc && !rempty_r;

    always @(*) begin
        rbin_w   = rbin_r + {{ADDR_W{1'b0}}, rpop};
        rgray_w  = (rbin_w >> 1) ^ rbin_w;
        rempty_w = (rgray_w == wgray_sync);
    end

    always @(posedge i_rclk or negedge i_rrst_n) begin
        if (!i_rrst_n) begin
            rbin_r   <= {(ADDR_W+1){1'b0}};
            rgray_r  <= {(ADDR_W+1){1'b0}};
            rempty_r <= 1'b1;
        end else begin
            rbin_r   <= rbin_w;
            rgray_r  <= rgray_w;
            rempty_r <= rempty_w;
        end
    end

    assign o_rdata  = mem[rbin_r[ADDR_W-1:0]];
    assign o_rempty = rempty_r;

    //------------------------------------------------------------------------
    // pointer synchronizers
    //------------------------------------------------------------------------
    sync_2ff #(.WIDTH(ADDR_W+1)) u_sync_r2w (
        .i_clk   (i_wclk),
        .i_rst_n (i_wrst_n),
        .i_d     (rgray_r),
        .o_q     (rgray_sync)
    );

    sync_2ff #(.WIDTH(ADDR_W+1)) u_sync_w2r (
        .i_clk   (i_rclk),
        .i_rst_n (i_rrst_n),
        .i_d     (wgray_r),
        .o_q     (wgray_sync)
    );

`ifndef SYNTHESIS
    always @(posedge i_wclk) begin
        if (i_winc && wfull_r) begin $display("[%0t] ERROR %m: write when full", $time); $stop; end
    end
    always @(posedge i_rclk) begin
        if (i_rinc && rempty_r) begin $display("[%0t] ERROR %m: read when empty", $time); $stop; end
    end
`endif
endmodule
`default_nettype wire
