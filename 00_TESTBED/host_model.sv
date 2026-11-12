// ============================================================================
// host_model.sv -- Host 行為模型（AXI4-Lite master，取代助教加密的 HOST.svp）
//
// 每一批：
//   1. 拉 ctrl_next_images，讓 testbench 把這批圖片載入 DRAM
//   2. 寫 0x00 = 張數 − 1
//   3. 每隔 POLL_GAP 個 cycle 讀 0x04，直到 finish = 1
//      （寫完 start 後第一次讀到的 finish 必須是 0，否則表示讀到上一批的舊值）
//   4. 讀 0x08 + 4k，跟預期值比對
// 比對方式 CHECK_MODE：
//   0 = 跟 golden_results.txt 比（真的 engine）；同時統計跟浮點模型（categories.csv）的一致率
//   1 = stub 模式：預期 {k, 4'h0, 圖片 3072 bytes 的總和}
// 輸出在負緣改變，正緣判斷 handshake。
// ============================================================================
`timescale 1ns/1ps
`ifndef PATTERN_DIR
`define PATTERN_DIR "../00_TESTBED/PATTERN"
`endif
// 註：資料夾路徑用 define 而不是 parameter：Icarus 會在 parameter 字串前面補空字元，組出錯的檔名
module host_model #(
    parameter TOTAL_IMAGE_NUMBER = 64,
    parameter BATCH_IMAGE_NUMBER = 16,
    parameter POLL_GAP           = 16,
    parameter CHECK_MODE         = 0
) (
    input  wire         clk,
    input  wire         rst_n,
    output reg  [7:0]   awaddr,
    output reg          awvalid,
    input  wire         awready,
    output reg  [31:0]  wdata,
    output reg  [3:0]   wstrb,
    output reg          wvalid,
    input  wire         wready,
    input  wire [1:0]   bresp,
    input  wire         bvalid,
    output reg          bready,
    output reg  [7:0]   araddr,
    output reg          arvalid,
    input  wire         arready,
    input  wire [31:0]  rdata,
    input  wire [1:0]   rresp,
    input  wire         rvalid,
    output reg          rready,
    output reg          ctrl_next_images,
    output reg  [7:0]   ctrl_image_start_num,
    output reg  [7:0]   ctrl_image_len_num,
    output reg          all_done
);
    // ------------------------------------------------------------ AXI4-Lite 交易
    task axi_write(input [7:0] a, input [31:0] d);
        reg aw_ok, w_ok, b_ok;
        begin
            @(negedge clk);
            awaddr <= a; awvalid <= 1; wdata <= d; wstrb <= 4'hF; wvalid <= 1; bready <= 1;
            aw_ok = 0; w_ok = 0; b_ok = 0;
            while (!b_ok) begin
                @(posedge clk);
                if (awvalid && awready) aw_ok = 1;
                if (wvalid && wready)   w_ok  = 1;
                if (bvalid && bready) begin
                    b_ok = 1;
                    if (bresp !== 2'b00) $display("[%0t] HOST ERROR: bresp = %b", $time, bresp);
                end
                @(negedge clk);
                if (aw_ok) awvalid <= 0;
                if (w_ok)  wvalid  <= 0;
            end
            bready <= 0;
        end
    endtask

    task axi_read(input [7:0] a, output [31:0] d);
        reg ar_ok, r_ok;
        begin
            @(negedge clk);
            araddr <= a; arvalid <= 1; rready <= 1;
            ar_ok = 0; r_ok = 0;
            while (!r_ok) begin
                @(posedge clk);
                if (arvalid && arready) ar_ok = 1;
                if (rvalid && rready) begin
                    r_ok = 1; d = rdata;
                    if (rresp !== 2'b00) $display("[%0t] HOST ERROR: rresp = %b", $time, rresp);
                end
                @(negedge clk);
                if (ar_ok) arvalid <= 0;
            end
            rready <= 0;
        end
    endtask

    // ------------------------------------------------------------ 預期值
    integer gold_cat [0:255];
    integer gold_conf[0:255];
    integer float_cat[0:255];

    task load_expect;
        integer fd, i, c, v, r;
        reg [8*300-1:0] fname;
        begin
            $sformat(fname, "%0s/golden_results.txt", `PATTERN_DIR);
            fd = $fopen(fname, "r");
            if (fd == 0) begin $display("HOST ERROR: 打不開 %0s", fname); $finish; end
            for (i = 0; i < TOTAL_IMAGE_NUMBER; i = i + 1) r = $fscanf(fd, "%d %d\n", gold_cat[i], gold_conf[i]);
            $fclose(fd);
            $sformat(fname, "%0s/categories.csv", `PATTERN_DIR);
            fd = $fopen(fname, "r");
            for (i = 0; i < TOTAL_IMAGE_NUMBER; i = i + 1) r = $fscanf(fd, "%d\n", float_cat[i]);
            $fclose(fd);
        end
    endtask

    function [23:0] image_sum(input integer id);
        integer fd, i, c;
        reg [8*300-1:0] fname;
        begin
            $sformat(fname, "%0s/IMAGES/image_%0d.bin", `PATTERN_DIR, id);
            fd = $fopen(fname, "rb");
            image_sum = 0;
            for (i = 0; i < 3072; i = i + 1) begin c = $fgetc(fd); image_sum = image_sum + c[7:0]; end
            $fclose(fd);
        end
    endfunction

    // ------------------------------------------------------------ 主流程
    integer start, len, k, polls, n_err, n_agree_float, n_ok;
    reg [31:0] d;
    reg [31:0] expv;
    integer    cat, conf;
    real       t_batch;

    initial begin
        awaddr = 0; awvalid = 0; wdata = 0; wstrb = 0; wvalid = 0; bready = 0;
        araddr = 0; arvalid = 0; rready = 0;
        ctrl_next_images = 0; ctrl_image_start_num = 0; ctrl_image_len_num = 0; all_done = 0;
        n_err = 0; n_agree_float = 0; n_ok = 0;
        if (CHECK_MODE == 0) load_expect;

        wait (rst_n === 1'b1);
        repeat (10) @(posedge clk);

        for (start = 0; start < TOTAL_IMAGE_NUMBER; start = start + BATCH_IMAGE_NUMBER) begin
            len = (TOTAL_IMAGE_NUMBER - start < BATCH_IMAGE_NUMBER) ? TOTAL_IMAGE_NUMBER - start : BATCH_IMAGE_NUMBER;
            // 1. 載入圖片
            @(negedge clk);
            ctrl_image_start_num <= start; ctrl_image_len_num <= len; ctrl_next_images <= 1;
            @(negedge clk);
            ctrl_next_images <= 0;
            // 2. start
            t_batch = $realtime;
            axi_write(8'h00, len - 1);
            // 3. 輪詢
            polls = 0;
            d = 0;
            while (d[0] !== 1'b1) begin
                axi_read(8'h04, d);
                if (polls == 0 && d[0] === 1'b1) begin
                    $display("[%0t] HOST ERROR: start 後第一次輪詢就讀到 finish = 1（舊的 finish）", $time);
                    n_err = n_err + 1;
                end
                polls = polls + 1;
                if (d[0] !== 1'b1) repeat (POLL_GAP) @(posedge clk);
            end
            // 4. 讀結果
            for (k = 0; k < len; k = k + 1) begin
                axi_read(8'h08 + 4 * k, d);
                cat  = d[31:28];
                conf = {{4{d[27]}}, d[27:0]};
                if (CHECK_MODE == 1) begin
                    expv = {k[3:0], 4'h0, image_sum(start + k)};
                    if (d !== expv) begin
                        n_err = n_err + 1;
                        $display("[%0t] HOST ERROR: image %0d 讀到 %h，預期 %h", $time, start + k, d, expv);
                    end else n_ok = n_ok + 1;
                end else begin
                    if (cat !== gold_cat[start + k] || conf !== gold_conf[start + k]) begin
                        n_err = n_err + 1;
                        $display("[%0t] HOST ERROR: image %0d 讀到 (%0d, %0d)，golden (%0d, %0d)",
                                 $time, start + k, cat, conf, gold_cat[start + k], gold_conf[start + k]);
                    end else n_ok = n_ok + 1;
                    if (cat == float_cat[start + k]) n_agree_float = n_agree_float + 1;
                end
            end
            $display("[%0t] HOST: 第 %0d 批（image %0d..%0d）完成，花 %0.1f ns，輪詢 %0d 次",
                     $time, start / BATCH_IMAGE_NUMBER, start, start + len - 1, $realtime - t_batch, polls);
        end

        $display("==================================================================");
        $display(" 圖片 %0d 張，正確 %0d，錯誤 %0d   %s", TOTAL_IMAGE_NUMBER, n_ok, n_err,
                 (n_err == 0 && n_ok == TOTAL_IMAGE_NUMBER) ? "PASS" : "FAIL");
        if (CHECK_MODE == 0)
            $display(" 與浮點模型類別一致：%0d / %0d", n_agree_float, TOTAL_IMAGE_NUMBER);
        $display(" 總時間：%0.1f ns", $realtime);
        $display("==================================================================");
        all_done = 1;
    end
endmodule
