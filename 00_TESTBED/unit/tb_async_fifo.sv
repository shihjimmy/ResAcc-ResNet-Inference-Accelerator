// ============================================================================
// tb_async_fifo.sv -- async_fifo 單元測試
//   兩個不相關的時脈、隨機的寫入/讀取節奏，送 N_ITEMS 筆遞增資料，
//   讀出端逐筆比對順序與內容；同時統計吞吐量。
//   參數由命令列 -P 指定：WPER / RPER（ns）、WPROB / RPROB（每個 cycle 想寫/讀的機率 %）
// ============================================================================
`timescale 1ns/1ps
module tb_async_fifo;
    parameter real WPER  = 0.834;     // 寫入端週期（預設：DRAM 1.2 GHz）
    parameter real RPER  = 2.0;       // 讀出端週期（預設：core 500 MHz）
    parameter      WPROB = 100;
    parameter      RPROB = 100;
    parameter      N_ITEMS = 20000;
    localparam W = 32;

    reg wclk = 0, rclk = 0, rst_n = 0;
    always #(WPER/2) wclk = ~wclk;
    initial begin #(RPER*0.37); forever #(RPER/2) rclk = ~rclk; end   // 故意錯開相位

    wire wrst_n, rrst_n;
    reset_sync u_rw (.i_clk(wclk), .i_rst_n(rst_n), .o_rst_n(wrst_n));
    reset_sync u_rr (.i_clk(rclk), .i_rst_n(rst_n), .o_rst_n(rrst_n));

    reg          winc = 0, rinc = 0;
    reg  [W-1:0] wdata = 0;
    wire [W-1:0] rdata;
    wire         wfull, rempty;

    async_fifo #(.WIDTH(W), .ADDR_W(4)) dut (
        .i_wclk(wclk), .i_wrst_n(wrst_n), .i_winc(winc), .i_wdata(wdata), .o_wfull(wfull),
        .i_rclk(rclk), .i_rrst_n(rrst_n), .i_rinc(rinc), .o_rdata(rdata), .o_rempty(rempty));

    // ---- 寫入端：在 negedge 決定下一個 cycle 要不要寫（避免和 posedge 賽跑）
    integer n_written = 0;
    always @(negedge wclk) begin
        if (wrst_n && !wfull && n_written < N_ITEMS && ($urandom % 100) < WPROB) begin
            winc  <= 1;
            wdata <= n_written;
        end else
            winc <= 0;
    end
    always @(posedge wclk) if (winc && !wfull) n_written = n_written + 1;

    // ---- 讀出端：比對
    integer n_read = 0, n_err = 0;
    real t_first, t_last;
    always @(negedge rclk) rinc <= (rrst_n && !rempty && ($urandom % 100) < RPROB);
    always @(posedge rclk) begin
        if (rinc && !rempty) begin
            if (rdata !== n_read) begin
                n_err = n_err + 1;
                if (n_err < 5) $display("[%0t] ERROR: 讀到 %0d，應該是 %0d", $time, rdata, n_read);
            end
            if (n_read == 0) t_first = $realtime;
            t_last = $realtime;
            n_read = n_read + 1;
        end
    end

    initial begin
        #20 rst_n = 1;
        wait (n_read == N_ITEMS);
        #50;
        $display("async_fifo  W=%0.3fns(%0d%%) R=%0.3fns(%0d%%): %0d 筆, 錯誤 %0d, 讀出速率 %0.2f 筆/讀端cycle  %s",
                 WPER, WPROB, RPER, RPROB, n_read, n_err,
                 (n_read-1) * RPER / (t_last - t_first), (n_err == 0 && rempty) ? "PASS" : "FAIL");
        $finish;
    end
    initial begin #(N_ITEMS * 20.0 + 1000); $display("TIMEOUT  FAIL"); $finish; end
endmodule
