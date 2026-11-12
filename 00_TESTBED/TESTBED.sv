// ============================================================================
// TESTBED.sv -- 系統 testbench（SPEC 9.1）
//   resacc（DUT）+ axi_dram（DRAM 模型）+ host_model（Host 模型）
//   參數用 +define+ 指定（見 01_RTL/run_*.sh），沒指定就用下面的預設值
// ============================================================================
`timescale 1ns/1ps
`ifndef TOTAL_IMAGE_NUMBER
`define TOTAL_IMAGE_NUMBER 64
`endif
`ifndef BATCH_IMAGE_NUMBER
`define BATCH_IMAGE_NUMBER 16
`endif
`ifndef READ_LATENCY
`define READ_LATENCY 12
`endif
`ifndef WRITE_LATENCY
`define WRITE_LATENCY 16
`endif
`ifndef DUT_CLK_PERIOD
`define DUT_CLK_PERIOD 2.0
`endif
`ifndef RAND_GAP
`define RAND_GAP 0
`endif
`ifndef CHECK_MODE
`define CHECK_MODE 0
`endif
`ifndef POLL_GAP
`define POLL_GAP 16
`endif
`ifndef DUMP_DIR
`define DUMP_DIR "build"
`endif
`ifndef PATTERN_DIR
`define PATTERN_DIR "../00_TESTBED/PATTERN"
`endif
`define DRAM_CLK_PERIOD 0.834
`define HOST_CLK_PERIOD 1.3
`define MAX_TIME 10000000

module TESTBED;
    reg clk_dram = 0, clk_host = 0, clk_dut = 0, rst_n = 1;

    // ---------------------------------------------------------------- 連線
    wire [7:0]   dram_axi_awid, dram_axi_arid, dram_axi_bid, dram_axi_rid;
    wire [18:0]  dram_axi_awaddr, dram_axi_araddr;
    wire [7:0]   dram_axi_awlen, dram_axi_arlen;
    wire [2:0]   dram_axi_awsize, dram_axi_arsize;
    wire [1:0]   dram_axi_awburst, dram_axi_arburst, dram_axi_bresp, dram_axi_rresp;
    wire         dram_axi_awvalid, dram_axi_awready, dram_axi_wlast, dram_axi_wvalid, dram_axi_wready;
    wire         dram_axi_bvalid, dram_axi_bready, dram_axi_arvalid, dram_axi_arready;
    wire         dram_axi_rlast, dram_axi_rvalid, dram_axi_rready;
    wire [255:0] dram_axi_wdata, dram_axi_rdata;
    wire [31:0]  dram_axi_wstrb;

    wire [7:0]   host_axi_awaddr, host_axi_araddr;
    wire [31:0]  host_axi_wdata, host_axi_rdata;
    wire [3:0]   host_axi_wstrb;
    wire [1:0]   host_axi_bresp, host_axi_rresp;
    wire         host_axi_awvalid, host_axi_awready, host_axi_wvalid, host_axi_wready;
    wire         host_axi_bvalid, host_axi_bready, host_axi_arvalid, host_axi_arready;
    wire         host_axi_rvalid, host_axi_rready;

    wire         ctrl_next_images, all_done;
    wire [7:0]   ctrl_image_start_num, ctrl_image_len_num;

    // ---------------------------------------------------------------- DUT
    resacc dut (
        .clk_dut(clk_dut), .clk_host(clk_host), .clk_dram(clk_dram), .rst_n(rst_n),
        .host_axi_awaddr(host_axi_awaddr), .host_axi_awvalid(host_axi_awvalid), .host_axi_awready(host_axi_awready),
        .host_axi_wdata(host_axi_wdata), .host_axi_wstrb(host_axi_wstrb), .host_axi_wvalid(host_axi_wvalid),
        .host_axi_wready(host_axi_wready), .host_axi_bresp(host_axi_bresp), .host_axi_bvalid(host_axi_bvalid),
        .host_axi_bready(host_axi_bready), .host_axi_araddr(host_axi_araddr), .host_axi_arvalid(host_axi_arvalid),
        .host_axi_arready(host_axi_arready), .host_axi_rdata(host_axi_rdata), .host_axi_rresp(host_axi_rresp),
        .host_axi_rvalid(host_axi_rvalid), .host_axi_rready(host_axi_rready),
        .dram_axi_awid(dram_axi_awid), .dram_axi_awaddr(dram_axi_awaddr), .dram_axi_awlen(dram_axi_awlen),
        .dram_axi_awsize(dram_axi_awsize), .dram_axi_awburst(dram_axi_awburst), .dram_axi_awvalid(dram_axi_awvalid),
        .dram_axi_awready(dram_axi_awready), .dram_axi_wdata(dram_axi_wdata), .dram_axi_wstrb(dram_axi_wstrb),
        .dram_axi_wlast(dram_axi_wlast), .dram_axi_wvalid(dram_axi_wvalid), .dram_axi_wready(dram_axi_wready),
        .dram_axi_bid(dram_axi_bid), .dram_axi_bresp(dram_axi_bresp), .dram_axi_bvalid(dram_axi_bvalid),
        .dram_axi_bready(dram_axi_bready), .dram_axi_arid(dram_axi_arid), .dram_axi_araddr(dram_axi_araddr),
        .dram_axi_arlen(dram_axi_arlen), .dram_axi_arsize(dram_axi_arsize), .dram_axi_arburst(dram_axi_arburst),
        .dram_axi_arvalid(dram_axi_arvalid), .dram_axi_arready(dram_axi_arready), .dram_axi_rid(dram_axi_rid),
        .dram_axi_rdata(dram_axi_rdata), .dram_axi_rresp(dram_axi_rresp), .dram_axi_rlast(dram_axi_rlast),
        .dram_axi_rvalid(dram_axi_rvalid), .dram_axi_rready(dram_axi_rready));

    // ---------------------------------------------------------------- DRAM
    axi_dram #(.R_LAT(`READ_LATENCY), .W_LAT(`WRITE_LATENCY), .RAND_GAP(`RAND_GAP)) u_dram (
        .clk(clk_dram), .rst_n(rst_n),
        .s_axi_awid(dram_axi_awid), .s_axi_awaddr(dram_axi_awaddr), .s_axi_awlen(dram_axi_awlen),
        .s_axi_awsize(dram_axi_awsize), .s_axi_awburst(dram_axi_awburst), .s_axi_awvalid(dram_axi_awvalid),
        .s_axi_awready(dram_axi_awready), .s_axi_wdata(dram_axi_wdata), .s_axi_wstrb(dram_axi_wstrb),
        .s_axi_wlast(dram_axi_wlast), .s_axi_wvalid(dram_axi_wvalid), .s_axi_wready(dram_axi_wready),
        .s_axi_bid(dram_axi_bid), .s_axi_bresp(dram_axi_bresp), .s_axi_bvalid(dram_axi_bvalid),
        .s_axi_bready(dram_axi_bready), .s_axi_arid(dram_axi_arid), .s_axi_araddr(dram_axi_araddr),
        .s_axi_arlen(dram_axi_arlen), .s_axi_arsize(dram_axi_arsize), .s_axi_arburst(dram_axi_arburst),
        .s_axi_arvalid(dram_axi_arvalid), .s_axi_arready(dram_axi_arready), .s_axi_rid(dram_axi_rid),
        .s_axi_rdata(dram_axi_rdata), .s_axi_rresp(dram_axi_rresp), .s_axi_rlast(dram_axi_rlast),
        .s_axi_rvalid(dram_axi_rvalid), .s_axi_rready(dram_axi_rready));

    // ---------------------------------------------------------------- Host
    host_model #(.TOTAL_IMAGE_NUMBER(`TOTAL_IMAGE_NUMBER), .BATCH_IMAGE_NUMBER(`BATCH_IMAGE_NUMBER),
                 .POLL_GAP(`POLL_GAP), .CHECK_MODE(`CHECK_MODE)) u_host (
        .clk(clk_host), .rst_n(rst_n),
        .awaddr(host_axi_awaddr), .awvalid(host_axi_awvalid), .awready(host_axi_awready),
        .wdata(host_axi_wdata), .wstrb(host_axi_wstrb), .wvalid(host_axi_wvalid), .wready(host_axi_wready),
        .bresp(host_axi_bresp), .bvalid(host_axi_bvalid), .bready(host_axi_bready),
        .araddr(host_axi_araddr), .arvalid(host_axi_arvalid), .arready(host_axi_arready),
        .rdata(host_axi_rdata), .rresp(host_axi_rresp), .rvalid(host_axi_rvalid), .rready(host_axi_rready),
        .ctrl_next_images(ctrl_next_images), .ctrl_image_start_num(ctrl_image_start_num),
        .ctrl_image_len_num(ctrl_image_len_num), .all_done(all_done));

    // ---------------------------------------------------------------- 時脈與 reset（跟助教的 TESTBED 一樣）
    initial begin #5.0; @(posedge rst_n); #5.0; forever #(`DRAM_CLK_PERIOD/2.0) clk_dram = ~clk_dram; end
    initial begin #5.0; @(posedge rst_n); #5.0; forever #(`HOST_CLK_PERIOD/2.0) clk_host = ~clk_host; end
    initial begin #5.0; @(posedge rst_n); #5.0; forever #(`DUT_CLK_PERIOD/2.0)  clk_dut  = ~clk_dut;  end
    initial begin
        $display("TESTBED: %0d 張、每批 %0d 張、READ_LATENCY %0d、RAND_GAP %0d%%、clk_dut 週期 %0.2f ns",
                 `TOTAL_IMAGE_NUMBER, `BATCH_IMAGE_NUMBER, `READ_LATENCY, `RAND_GAP, `DUT_CLK_PERIOD);
        #1.0 rst_n = 0;
        #50.0;
        $readmemh({`PATTERN_DIR, "/model.dat"}, u_dram.mem);
        rst_n = 1;
    end

    // ---------------------------------------------------------------- 每批開始時載入圖片
    reg [8*300-1:0] fname;
    integer i;
    always @(posedge ctrl_next_images) begin
        for (i = 0; i < ctrl_image_len_num; i = i + 1) begin
            $sformat(fname, "%0s/IMAGES/image_%0d.bin", `PATTERN_DIR, ctrl_image_start_num + i);
            u_dram.load_image(i, fname);
        end
    end

    // ---------------------------------------------------------------- 結束、逾時
    initial begin
        wait (all_done === 1'b1);
        #100 $finish;
    end
    initial begin
        #(`MAX_TIME);
        $display("**************** Simulation Timeout! ****************");
        $finish;
    end

`ifdef DUMPVCD
    initial begin $dumpfile("resacc.vcd"); $dumpvars(0, TESTBED); end
`endif

`ifdef DUMP_BUFFERS
    // ---------------------------------------------------------------- 周邊測試：把載入的記憶體內容寫成檔案
    reg [8*300-1:0] dname;
    integer fd;
    reg md_d = 0;
    always @(posedge clk_dut) md_d <= dut.u_core.ld_model_done;
    always @(posedge clk_dut) begin
        if (dut.u_core.ld_model_done && !md_d) begin
            $writememh({`DUMP_DIR, "/wbuf.hex"}, dut.u_core.u_wbuf.mem);
            $writememh({`DUMP_DIR, "/pbuf.hex"}, dut.u_core.u_pbuf.mem);
            $writememh({`DUMP_DIR, "/fcm.hex"}, dut.u_core.u_fcm.mem);
            fd = $fopen({`DUMP_DIR, "/header.txt"}, "w");
            $fdisplay(fd, "%h", dut.u_core.shift_vec);
            $fdisplay(fd, "%h", dut.u_core.conf_m);
            $fdisplay(fd, "%h", dut.u_core.conf_s);
            $fdisplay(fd, "%h", dut.u_core.bfc_vec);
            $fclose(fd);
            $display("[%0t] TB: 模型載入完成，已輸出 WBUF / PBUF / FCMEM / header", $time);
        end
        if (dut.u_core.ld_img_done) begin
            $sformat(dname, "%0s/img_%0d.hex", `DUMP_DIR, ctrl_image_start_num + dut.u_core.u_loader.slot_r);
            $writememh(dname, dut.u_core.u_img.mem);
        end
    end
`endif
`ifdef DUMP_LAYERS
    // ---------------------------------------------------------------- 逐層驗證：每一層寫回完成時輸出 FM / pool
    reg [8*300-1:0] lname;
    integer lfd, lk;
    always @(posedge clk_dut) begin
        if (dut.u_core.u_engine.u_ctrl.state_r == 3'd5 && !dut.u_core.u_engine.pipe_busy && !dut.u_core.u_engine.drain_busy) begin
            $sformat(lname, "%0s/L%0d_img%0d.hex", `DUMP_DIR, dut.u_core.u_engine.u_ctrl.step_r,
                     ctrl_image_start_num + dut.u_core.u_engine.u_ctrl.slot_r);
            case (dut.u_core.u_engine.u_ctrl.step_r)
            3'd0: $writememh(lname, dut.u_core.u_engine.u_dp.u_fm_a.mem);
            3'd1: $writememh(lname, dut.u_core.u_engine.u_dp.u_fm_b.mem);
            3'd2: $writememh(lname, dut.u_core.u_engine.u_dp.u_fm_c.mem);
            3'd3: $writememh(lname, dut.u_core.u_engine.u_dp.u_fm_b.mem);
            default: begin
                lfd = $fopen(lname, "w");
                for (lk = 0; lk < 32; lk = lk + 1) $fdisplay(lfd, "%h", dut.u_core.u_engine.u_dp.pool_r[lk]);
                $fclose(lfd);
            end
            endcase
        end
    end
`endif
endmodule
