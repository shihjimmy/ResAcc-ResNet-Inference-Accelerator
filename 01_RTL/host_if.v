`timescale 1ns/1ps
//============================================================================
// host_if -- AXI4-Lite slave on the host side (clk_host), SPEC 4.2-4.4 / 6.3
//
// Registers
//   0x00 W : [3:0] = number of images - 1; writing it starts a batch
//   0x04 R : [0]   = finish
//   0x08+4k R : result of image k (read directly from the core's result registers)
//
// Clock domain crossing
//   * start  : every write to 0x00 toggles o_start_tgl; the core detects the toggle
//              after a 2-FF synchronizer. o_n_minus_1 is written in the same cycle
//              and stays stable, so the core samples it only after seeing the toggle.
//   * done   : level from the core, 2-FF synchronized, rising edge sets finish.
//              Writing 0x00 clears finish immediately (never report a stale finish).
//   * result : intentionally NOT synchronized. The core fixes all results before
//              raising done, and the host only reads them after seeing finish.
//              (Needs a false-path constraint in synthesis.)
//============================================================================
`default_nettype none
module host_if (
    input  wire          i_clk,
    input  wire          i_rst_n,
    // AXI4-Lite slave
    input  wire [7:0]    i_awaddr,
    input  wire          i_awvalid,
    output wire          o_awready,
    input  wire [31:0]   i_wdata,
    input  wire [3:0]    i_wstrb,              // ignored: always a full 32-bit write
    input  wire          i_wvalid,
    output wire          o_wready,
    output wire [1:0]    o_bresp,
    output wire          o_bvalid,
    input  wire          i_bready,
    input  wire [7:0]    i_araddr,
    input  wire          i_arvalid,
    output wire          o_arready,
    output wire [31:0]   o_rdata,
    output wire [1:0]    o_rresp,
    output wire          o_rvalid,
    input  wire          i_rready,
    // to / from the core
    output wire          o_start_tgl,          // to core (synchronized there)
    output wire [3:0]    o_n_minus_1,          // to core (stable when start_tgl toggles)
    input  wire          i_core_done,          // from core (synchronized here)
    input  wire [511:0]  i_result_vec          // from core (quasi-static, not synchronized)
);
    //------------------------------------------------------------------------
    // registers
    //------------------------------------------------------------------------
    reg        aw_got_r,  aw_got_w;            // write address received
    reg        w_got_r,   w_got_w;             // write data received
    reg [7:0]  waddr_r,   waddr_w;
    reg [31:0] wdata_r,   wdata_w;
    reg        bvalid_r,  bvalid_w;
    reg        rvalid_r,  rvalid_w;
    reg [31:0] rdata_r,   rdata_w;
    reg        start_tgl_r, start_tgl_w;
    reg [3:0]  n_minus_1_r, n_minus_1_w;
    reg        finish_r,  finish_w;
    reg        done_d_r;                       // previous synchronized done (edge detect)

    wire done_sync;

    //------------------------------------------------------------------------
    // write channel
    //------------------------------------------------------------------------
    wire        awready  = !aw_got_r && !bvalid_r;
    wire        wready   = !w_got_r  && !bvalid_r;
    wire        aw_hs    = i_awvalid && awready;
    wire        w_hs     = i_wvalid  && wready;
    wire        do_write = (aw_got_r || aw_hs) && (w_got_r || w_hs) && !bvalid_r;
    wire [7:0]  wr_addr  = aw_got_r ? waddr_r : i_awaddr;
    wire [31:0] wr_data  = w_got_r  ? wdata_r : i_wdata;
    wire        wr_start = do_write && (wr_addr == 8'h00);

    always @(*) begin
        aw_got_w    = aw_got_r;
        w_got_w     = w_got_r;
        waddr_w     = waddr_r;
        wdata_w     = wdata_r;
        bvalid_w    = bvalid_r;
        start_tgl_w = start_tgl_r;
        n_minus_1_w = n_minus_1_r;

        if (do_write) begin
            aw_got_w = 1'b0;
            w_got_w  = 1'b0;
            bvalid_w = 1'b1;
            if (wr_start) begin
                n_minus_1_w = wr_data[3:0];
                start_tgl_w = ~start_tgl_r;
            end
        end else begin
            if (aw_hs) begin aw_got_w = 1'b1; waddr_w = i_awaddr; end
            if (w_hs)  begin w_got_w  = 1'b1; wdata_w = i_wdata;  end
        end
        if (bvalid_r && i_bready) bvalid_w = 1'b0;
    end

    //------------------------------------------------------------------------
    // finish: cleared by a new start, set by the rising edge of done
    //------------------------------------------------------------------------
    always @(*) begin
        finish_w = finish_r;
        if (wr_start)                    finish_w = 1'b0;
        else if (done_sync && !done_d_r) finish_w = 1'b1;
    end

    //------------------------------------------------------------------------
    // read channel
    //------------------------------------------------------------------------
    wire       arready = !rvalid_r;
    wire [7:0] ridx    = i_araddr - 8'h08;           // byte offset inside the result area

    always @(*) begin
        rvalid_w = rvalid_r;
        rdata_w  = rdata_r;
        if (i_arvalid && arready) begin
            rvalid_w = 1'b1;
            if (i_araddr == 8'h04)
                rdata_w = {31'd0, finish_r};
            else if (i_araddr >= 8'h08 && i_araddr <= 8'h44 && i_araddr[1:0] == 2'b00)
                rdata_w = i_result_vec[ridx[5:2]*32 +: 32];
            else
                rdata_w = 32'd0;
        end else if (rvalid_r && i_rready) begin
            rvalid_w = 1'b0;
        end
    end

    //------------------------------------------------------------------------
    // sequential
    //------------------------------------------------------------------------
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            aw_got_r    <= 1'b0;
            w_got_r     <= 1'b0;
            waddr_r     <= 8'd0;
            wdata_r     <= 32'd0;
            bvalid_r    <= 1'b0;
            rvalid_r    <= 1'b0;
            rdata_r     <= 32'd0;
            start_tgl_r <= 1'b0;
            n_minus_1_r <= 4'd0;
            finish_r    <= 1'b0;
            done_d_r    <= 1'b0;
        end else begin
            aw_got_r    <= aw_got_w;
            w_got_r     <= w_got_w;
            waddr_r     <= waddr_w;
            wdata_r     <= wdata_w;
            bvalid_r    <= bvalid_w;
            rvalid_r    <= rvalid_w;
            rdata_r     <= rdata_w;
            start_tgl_r <= start_tgl_w;
            n_minus_1_r <= n_minus_1_w;
            finish_r    <= finish_w;
            done_d_r    <= done_sync;
        end
    end

    sync_2ff #(.WIDTH(1)) u_done_sync (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_d     (i_core_done),
        .o_q     (done_sync)
    );

    //------------------------------------------------------------------------
    // outputs
    //------------------------------------------------------------------------
    assign o_awready   = awready;
    assign o_wready    = wready;
    assign o_bresp     = 2'b00;
    assign o_bvalid    = bvalid_r;
    assign o_arready   = arready;
    assign o_rdata     = rdata_r;
    assign o_rresp     = 2'b00;
    assign o_rvalid    = rvalid_r;
    assign o_start_tgl = start_tgl_r;
    assign o_n_minus_1 = n_minus_1_r;
endmodule
`default_nettype wire
