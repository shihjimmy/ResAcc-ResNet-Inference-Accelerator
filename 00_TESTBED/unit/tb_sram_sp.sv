// ============================================================================
// tb_sram_sp.sv -- sram_sp 單元測試：讀取延遲 1 cycle、byte enable、未寫入為 X
// ============================================================================
`timescale 1ns/1ps
module tb_sram_sp;
    reg clk = 0; always #1 clk = ~clk;
    reg cs = 0, we = 0;
    reg  [3:0]  addr = 0;
    reg  [63:0] wdata = 0;
    reg  [7:0]  wbe = 0;
    wire [63:0] rdata;
    integer err = 0;

    sram_sp #(.DEPTH(16), .WIDTH(64)) dut (.i_clk(clk), .i_cs(cs), .i_we(we), .i_addr(addr), .i_wdata(wdata), .i_wbe(wbe), .o_rdata(rdata));

    task wr(input [3:0] a, input [63:0] d, input [7:0] be);
        begin @(negedge clk); cs = 1; we = 1; addr = a; wdata = d; wbe = be; @(negedge clk); cs = 0; we = 0; end
    endtask
    task rd_check(input [3:0] a, input [63:0] exp, input [8*20-1:0] msg);
        begin
            @(negedge clk); cs = 1; we = 0; addr = a;
            @(posedge clk); #0.1;                        // 讀取在這個 posedge 發生
            @(negedge clk); cs = 0;
            if (rdata !== exp) begin err = err + 1; $display("ERROR %0s: got %h exp %h", msg, rdata, exp); end
        end
    endtask

    initial begin
        wr(3, 64'h1122334455667788, 8'hFF);
        rd_check(3, 64'h1122334455667788, "full write");
        wr(3, 64'hAAAAAAAAAAAAAAAA, 8'b0000_1111);       // 只寫低 4 byte
        rd_check(3, 64'h11223344AAAAAAAA, "byte enable");
        rd_check(5, 64'hxxxxxxxxxxxxxxxx, "unwritten = X");
        $display("sram_sp: 錯誤 %0d  %s", err, err == 0 ? "PASS" : "FAIL");
        $finish;
    end
endmodule
