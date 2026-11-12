// ============================================================================
// axi_dram.sv -- AXI4 slave 的 DRAM 行為模型（取代助教加密的 AXI_DRAM.svp）
//
//   * 512 KB = 16384 個 256-bit word；testbench 用 $readmemh 把 model.dat 載入 mem
//   * 讀取：收到 AR 後等 R_LAT 個 cycle，再一個 cycle 一個 beat 送出（遵守 rready 背壓）
//   * 寫入：收 AW、W，等 W_LAT 個 cycle 回 B（本設計不會寫，但照樣支援）
//   * 一次只處理一筆讀取、一筆寫入
//   * 輸出在「負緣」改變，內部在正緣判斷 handshake（跟原本助教的模型一樣）
//   * RAND_GAP > 0：讀取時隨機插入 rvalid = 0 的空檔（百分比），用來壓力測試
//   * 檢查：burst 跨 4 KB 邊界、非 INCR、arsize 不是 32 bytes、寫入受保護區 → 報錯
// ============================================================================
`timescale 1ns/1ps
module axi_dram #(
    parameter R_LAT    = 12,
    parameter W_LAT    = 16,
    parameter RAND_GAP = 0,
    parameter PROTECTED_ADDR = 19'h3FFFF       // 這個位址以下不可寫（圖片 + 模型）
) (
    input  wire         clk,
    input  wire         rst_n,
    // AW
    input  wire [7:0]   s_axi_awid,
    input  wire [18:0]  s_axi_awaddr,
    input  wire [7:0]   s_axi_awlen,
    input  wire [2:0]   s_axi_awsize,
    input  wire [1:0]   s_axi_awburst,
    input  wire         s_axi_awvalid,
    output reg          s_axi_awready,
    // W
    input  wire [255:0] s_axi_wdata,
    input  wire [31:0]  s_axi_wstrb,
    input  wire         s_axi_wlast,
    input  wire         s_axi_wvalid,
    output reg          s_axi_wready,
    // B
    output reg  [7:0]   s_axi_bid,
    output reg  [1:0]   s_axi_bresp,
    output reg          s_axi_bvalid,
    input  wire         s_axi_bready,
    // AR
    input  wire [7:0]   s_axi_arid,
    input  wire [18:0]  s_axi_araddr,
    input  wire [7:0]   s_axi_arlen,
    input  wire [2:0]   s_axi_arsize,
    input  wire [1:0]   s_axi_arburst,
    input  wire         s_axi_arvalid,
    output reg          s_axi_arready,
    // R
    output reg  [7:0]   s_axi_rid,
    output reg  [255:0] s_axi_rdata,
    output reg  [1:0]   s_axi_rresp,
    output reg          s_axi_rlast,
    output reg          s_axi_rvalid,
    input  wire         s_axi_rready
);
    reg [255:0] mem [0:16383];

    // ------------------------------------------------------------ 讀取
    // n_* = 下一個負緣要放到輸出的值；在正緣算好
    reg         n_arready, n_rvalid, n_rlast;
    reg [255:0] n_rdata;
    reg [7:0]   n_rid;
    integer     r_state;            // 0 閒置、1 等延遲、2 送資料
    integer     r_word, r_beat, r_len, r_cnt;
    reg         r_new;              // 1 = 這個 cycle 要放上新的 beat（或重新決定要不要空拍）
    integer     n_reads = 0;

    task check_burst(input [18:0] addr, input [7:0] len, input [2:0] size, input [1:0] burst, input [8*5-1:0] who);
        begin
            if (size !== 3'b101) $display("[%0t] DRAM ERROR %0s: size=%0d（應為 5 = 32 bytes）", $time, who, size);
            if (burst !== 2'b01) $display("[%0t] DRAM ERROR %0s: burst=%0d（應為 INCR）", $time, who, burst);
            if (addr[4:0] !== 0)  $display("[%0t] DRAM ERROR %0s: 位址 0x%0h 沒有對齊 32 bytes", $time, who, addr);
            if ((addr % 4096) + (len + 1) * 32 > 4096)
                $display("[%0t] DRAM ERROR %0s: burst 0x%0h + %0d beats 跨 4KB 邊界", $time, who, addr, len + 1);
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state   = 0; r_new = 0;
            n_arready = 1; n_rvalid = 0; n_rlast = 0; n_rdata = 0; n_rid = 0;
        end else begin
            case (r_state)
            0: if (s_axi_arvalid && s_axi_arready) begin
                   check_burst(s_axi_araddr, s_axi_arlen, s_axi_arsize, s_axi_arburst, "read");
                   r_word = s_axi_araddr >> 5; r_len = s_axi_arlen; r_beat = 0;
                   n_rid = s_axi_arid; r_cnt = R_LAT;
                   n_arready = 0;
                   n_reads = n_reads + 1;
                   r_state = 1;
               end
            1: if (r_cnt > 0) r_cnt = r_cnt - 1;
            2: if (s_axi_rvalid && s_axi_rready) begin    // 這個 beat 已經被收走
                   if (r_beat == r_len) begin              // 最後一個 beat 送完
                       n_rvalid = 0; n_rlast = 0; n_arready = 1;
                       r_state = 0;
                   end else begin
                       r_beat = r_beat + 1;
                       r_new = 1;
                   end
               end else
                   r_new = !s_axi_rvalid;                  // rvalid 已經拉高就必須保持（AXI 規則）
            endcase
            // 延遲結束 → 開始送資料
            if (r_state == 1 && r_cnt == 0) begin r_state = 2; r_new = 1; end
            // 需要放新的 beat 時，決定這個 cycle 送不送（可能隨機空一拍）
            if (r_state == 2 && r_new) begin
                n_rvalid = !(RAND_GAP > 0 && ($urandom % 100) < RAND_GAP);
                n_rdata  = mem[r_word + r_beat];
                n_rlast  = (r_beat == r_len);
            end
        end
    end

    always @(negedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 0; s_axi_rvalid <= 0; s_axi_rlast <= 0; s_axi_rdata <= 0;
            s_axi_rid <= 0; s_axi_rresp <= 0;
        end else begin
            s_axi_arready <= n_arready;
            s_axi_rvalid  <= n_rvalid;
            s_axi_rlast   <= n_rlast;
            s_axi_rdata   <= n_rdata;
            s_axi_rid     <= n_rid;
            s_axi_rresp   <= 2'b00;
        end
    end

    // ------------------------------------------------------------ 寫入
    reg         n_awready, n_wready, n_bvalid;
    reg [7:0]   n_bid;
    integer     w_state, w_word, w_beat, w_cnt, k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_state = 0; n_awready = 1; n_wready = 0; n_bvalid = 0; n_bid = 0;
        end else begin
            case (w_state)
            0: if (s_axi_awvalid && s_axi_awready) begin
                   check_burst(s_axi_awaddr, s_axi_awlen, s_axi_awsize, s_axi_awburst, "write");
                   if (s_axi_awaddr <= PROTECTED_ADDR)
                       $display("[%0t] DRAM ERROR: 寫入受保護區 0x%0h", $time, s_axi_awaddr);
                   w_word = s_axi_awaddr >> 5; w_beat = 0; n_bid = s_axi_awid;
                   n_awready = 0; n_wready = 1; w_state = 1;
               end
            1: if (s_axi_wvalid && s_axi_wready) begin
                   for (k = 0; k < 32; k = k + 1)
                       if (s_axi_wstrb[k]) mem[w_word + w_beat][k*8 +: 8] = s_axi_wdata[k*8 +: 8];
                   w_beat = w_beat + 1;
                   if (s_axi_wlast) begin n_wready = 0; w_cnt = W_LAT; w_state = 2; end
               end
            2: if (w_cnt > 0) w_cnt = w_cnt - 1; else begin n_bvalid = 1; w_state = 3; end
            3: if (s_axi_bvalid && s_axi_bready) begin n_bvalid = 0; n_awready = 1; w_state = 0; end
            endcase
        end
    end

    always @(negedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 0; s_axi_wready <= 0; s_axi_bvalid <= 0; s_axi_bid <= 0; s_axi_bresp <= 0;
        end else begin
            s_axi_awready <= n_awready; s_axi_wready <= n_wready; s_axi_bvalid <= n_bvalid;
            s_axi_bid <= n_bid; s_axi_bresp <= 2'b00;
        end
    end

    // ------------------------------------------------------------ testbench 用的 task
    // 把一張圖片（3072 bytes）從檔案讀到第 slot 個位置（byte 位址 0x1000·slot）
    task load_image(input integer slot, input [8*256-1:0] fname);
        integer fd, i, c;
        begin
            fd = $fopen(fname, "rb");
            if (fd == 0) begin $display("DRAM ERROR: 打不開 %0s", fname); $finish; end
            for (i = 0; i < 3072; i = i + 1) begin
                c = $fgetc(fd);
                mem[(slot * 4096 + i) >> 5][(i % 32) * 8 +: 8] = c[7:0];
            end
            $fclose(fd);
        end
    endtask
endmodule
