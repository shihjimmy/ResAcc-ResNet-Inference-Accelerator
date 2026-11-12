`timescale 1ns/1ps
//============================================================================
// resacc -- top level, SPEC 6.1
//
// Only wiring: one reset synchronizer per clock domain, the two async FIFOs,
// host_if (clk_host), dram_if (clk_dram) and core (clk_dut).
// Port names follow the course TESTBED and must not change.
//============================================================================
`default_nettype none
module resacc (
    input  wire          clk_dut,
    input  wire          clk_host,
    input  wire          clk_dram,
    input  wire          rst_n,
    // host AXI4-Lite
    input  wire [7:0]    host_axi_awaddr,
    input  wire          host_axi_awvalid,
    output wire          host_axi_awready,
    input  wire [31:0]   host_axi_wdata,
    input  wire [3:0]    host_axi_wstrb,
    input  wire          host_axi_wvalid,
    output wire          host_axi_wready,
    output wire [1:0]    host_axi_bresp,
    output wire          host_axi_bvalid,
    input  wire          host_axi_bready,
    input  wire [7:0]    host_axi_araddr,
    input  wire          host_axi_arvalid,
    output wire          host_axi_arready,
    output wire [31:0]   host_axi_rdata,
    output wire [1:0]    host_axi_rresp,
    output wire          host_axi_rvalid,
    input  wire          host_axi_rready,
    // DRAM AXI4
    output wire [7:0]    dram_axi_awid,
    output wire [18:0]   dram_axi_awaddr,
    output wire [7:0]    dram_axi_awlen,
    output wire [2:0]    dram_axi_awsize,
    output wire [1:0]    dram_axi_awburst,
    output wire          dram_axi_awvalid,
    input  wire          dram_axi_awready,
    output wire [255:0]  dram_axi_wdata,
    output wire [31:0]   dram_axi_wstrb,
    output wire          dram_axi_wlast,
    output wire          dram_axi_wvalid,
    input  wire          dram_axi_wready,
    input  wire [7:0]    dram_axi_bid,
    input  wire [1:0]    dram_axi_bresp,
    input  wire          dram_axi_bvalid,
    output wire          dram_axi_bready,
    output wire [7:0]    dram_axi_arid,
    output wire [18:0]   dram_axi_araddr,
    output wire [7:0]    dram_axi_arlen,
    output wire [2:0]    dram_axi_arsize,
    output wire [1:0]    dram_axi_arburst,
    output wire          dram_axi_arvalid,
    input  wire          dram_axi_arready,
    input  wire [7:0]    dram_axi_rid,
    input  wire [255:0]  dram_axi_rdata,
    input  wire [1:0]    dram_axi_rresp,
    input  wire          dram_axi_rlast,
    input  wire          dram_axi_rvalid,
    output wire          dram_axi_rready
);
    //------------------------------------------------------------------------
    // wires
    //------------------------------------------------------------------------
    wire         rst_dut_n, rst_host_n, rst_dram_n;
    wire         start_tgl, core_done;
    wire [3:0]   n_minus_1;
    wire [511:0] result_vec;
    wire         req_full, req_push, req_empty, req_pop;
    wire [26:0]  req_wdata, req_rdata;
    wire         dat_full, dat_push, dat_empty, dat_pop;
    wire [255:0] dat_wdata, dat_rdata;

    //------------------------------------------------------------------------
    // DRAM write channel: unused
    //------------------------------------------------------------------------
    assign dram_axi_awid    = 8'd0;
    assign dram_axi_awaddr  = 19'd0;
    assign dram_axi_awlen   = 8'd0;
    assign dram_axi_awsize  = 3'b101;
    assign dram_axi_awburst = 2'b01;
    assign dram_axi_awvalid = 1'b0;
    assign dram_axi_wdata   = 256'd0;
    assign dram_axi_wstrb   = 32'd0;
    assign dram_axi_wlast   = 1'b0;
    assign dram_axi_wvalid  = 1'b0;
    assign dram_axi_bready  = 1'b1;
    assign dram_axi_arid    = 8'd0;
    assign dram_axi_arsize  = 3'b101;        // 32 bytes per beat
    assign dram_axi_arburst = 2'b01;         // INCR

    //------------------------------------------------------------------------
    // reset synchronizers
    //------------------------------------------------------------------------
    reset_sync u_rst_dut  (.i_clk(clk_dut),  .i_rst_n(rst_n), .o_rst_n(rst_dut_n));
    reset_sync u_rst_host (.i_clk(clk_host), .i_rst_n(rst_n), .o_rst_n(rst_host_n));
    reset_sync u_rst_dram (.i_clk(clk_dram), .i_rst_n(rst_n), .o_rst_n(rst_dram_n));

    //------------------------------------------------------------------------
    // host side
    //------------------------------------------------------------------------
    host_if u_host_if (
        .i_clk        (clk_host),
        .i_rst_n      (rst_host_n),
        .i_awaddr     (host_axi_awaddr),
        .i_awvalid    (host_axi_awvalid),
        .o_awready    (host_axi_awready),
        .i_wdata      (host_axi_wdata),
        .i_wstrb      (host_axi_wstrb),
        .i_wvalid     (host_axi_wvalid),
        .o_wready     (host_axi_wready),
        .o_bresp      (host_axi_bresp),
        .o_bvalid     (host_axi_bvalid),
        .i_bready     (host_axi_bready),
        .i_araddr     (host_axi_araddr),
        .i_arvalid    (host_axi_arvalid),
        .o_arready    (host_axi_arready),
        .o_rdata      (host_axi_rdata),
        .o_rresp      (host_axi_rresp),
        .o_rvalid     (host_axi_rvalid),
        .i_rready     (host_axi_rready),
        .o_start_tgl  (start_tgl),
        .o_n_minus_1  (n_minus_1),
        .i_core_done  (core_done),
        .i_result_vec (result_vec)
    );

    //------------------------------------------------------------------------
    // async FIFOs
    //------------------------------------------------------------------------
    async_fifo #(.WIDTH(27), .ADDR_W(2)) u_req_fifo (     // core -> dram_if: read requests
        .i_wclk   (clk_dut),
        .i_wrst_n (rst_dut_n),
        .i_winc   (req_push),
        .i_wdata  (req_wdata),
        .o_wfull  (req_full),
        .i_rclk   (clk_dram),
        .i_rrst_n (rst_dram_n),
        .i_rinc   (req_pop),
        .o_rdata  (req_rdata),
        .o_rempty (req_empty)
    );

    async_fifo #(.WIDTH(256), .ADDR_W(4)) u_dat_fifo (    // dram_if -> core: read data
        .i_wclk   (clk_dram),
        .i_wrst_n (rst_dram_n),
        .i_winc   (dat_push),
        .i_wdata  (dat_wdata),
        .o_wfull  (dat_full),
        .i_rclk   (clk_dut),
        .i_rrst_n (rst_dut_n),
        .i_rinc   (dat_pop),
        .o_rdata  (dat_rdata),
        .o_rempty (dat_empty)
    );

    //------------------------------------------------------------------------
    // DRAM side
    //------------------------------------------------------------------------
    dram_if u_dram_if (
        .i_clk       (clk_dram),
        .i_rst_n     (rst_dram_n),
        .i_req_empty (req_empty),
        .i_req_data  (req_rdata),
        .o_req_pop   (req_pop),
        .i_dat_full  (dat_full),
        .o_dat_push  (dat_push),
        .o_dat_data  (dat_wdata),
        .o_araddr    (dram_axi_araddr),
        .o_arlen     (dram_axi_arlen),
        .o_arvalid   (dram_axi_arvalid),
        .i_arready   (dram_axi_arready),
        .i_rdata     (dram_axi_rdata),
        .i_rlast     (dram_axi_rlast),
        .i_rvalid    (dram_axi_rvalid),
        .o_rready    (dram_axi_rready)
    );

    //------------------------------------------------------------------------
    // core
    //------------------------------------------------------------------------
    core u_core (
        .i_clk        (clk_dut),
        .i_rst_n      (rst_dut_n),
        .i_start_tgl  (start_tgl),
        .i_n_minus_1  (n_minus_1),
        .o_done       (core_done),
        .o_result_vec (result_vec),
        .i_req_full   (req_full),
        .o_req_push   (req_push),
        .o_req_data   (req_wdata),
        .i_dat_empty  (dat_empty),
        .i_dat_data   (dat_rdata),
        .o_dat_pop    (dat_pop)
    );
endmodule
`default_nettype wire
