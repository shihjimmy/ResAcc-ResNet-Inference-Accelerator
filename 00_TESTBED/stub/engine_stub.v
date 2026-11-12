`timescale 1ns/1ps
//============================================================================
// engine_stub -- fake engine for testing the peripheral path (same ports as engine)
//
// On i_start:
//   1. read all 48 IMG words and add up the 3072 bytes (checks the image load path)
//   2. pulse o_stem_done (the core then prefetches the next image)
//   3. wait TAIL cycles, write result {slot, 4'h0, sum[23:0]}, pulse o_done
//   4. model the FC: o_fc_busy stays high for FC_T cycles
//============================================================================
`default_nettype none
module engine #(
    parameter TAIL = 300,
    parameter FC_T = 60
) (
    input  wire          i_clk,
    input  wire          i_rst_n,
    input  wire          i_start,
    input  wire [3:0]    i_slot,
    output wire          o_stem_done,
    output wire          o_done,
    output wire          o_fc_busy,
    output wire          o_res_we,
    output wire [3:0]    o_res_idx,
    output wire [31:0]   o_res_data,
    output wire          o_img_cs,
    output wire [5:0]    o_img_addr,
    input  wire [511:0]  i_img_rdata,
    output wire          o_wbuf_cs,
    output wire [10:0]   o_wbuf_addr,
    input  wire [127:0]  i_wbuf_rdata,
    output wire          o_pbuf_cs,
    output wire [4:0]    o_pbuf_addr,
    input  wire [255:0]  i_pbuf_rdata,
    output wire          o_fcm_cs,
    output wire [3:0]    o_fcm_addr,
    input  wire [255:0]  i_fcm_rdata,
    input  wire [55:0]   i_shift_vec,
    input  wire [15:0]   i_conf_m,
    input  wire [7:0]    i_conf_s,
    input  wire [319:0]  i_bfc_vec
);
    localparam S_IDLE = 2'd0, S_READ = 2'd1, S_TAIL = 2'd2;

    reg [1:0]  state_r;
    reg [5:0]  addr_r;
    reg        rd_valid_r;            // a read was issued last cycle
    reg [23:0] sum_r;
    reg [3:0]  slot_r;
    reg [9:0]  cnt_r, fc_cnt_r;
    reg        stem_done_r, done_r, res_we_r;

    function [23:0] bytesum(input [511:0] w);
        integer i;
        begin
            bytesum = 24'd0;
            for (i = 0; i < 64; i = i + 1) bytesum = bytesum + w[i*8 +: 8];
        end
    endfunction

    wire img_cs = (state_r == S_READ) && (addr_r < 6'd48);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r <= S_IDLE; addr_r <= 6'd0; rd_valid_r <= 1'b0; sum_r <= 24'd0; slot_r <= 4'd0;
            cnt_r <= 10'd0; fc_cnt_r <= 10'd0; stem_done_r <= 1'b0; done_r <= 1'b0; res_we_r <= 1'b0;
        end else begin
            stem_done_r <= 1'b0;
            done_r      <= 1'b0;
            res_we_r    <= 1'b0;
            rd_valid_r  <= img_cs;
            if (fc_cnt_r != 10'd0) fc_cnt_r <= fc_cnt_r - 10'd1;
            if (rd_valid_r) sum_r <= sum_r + bytesum(i_img_rdata);
            case (state_r)
                S_IDLE: if (i_start) begin
                            slot_r <= i_slot; addr_r <= 6'd0; sum_r <= 24'd0; state_r <= S_READ;
                        end
                S_READ: if (addr_r < 6'd48) addr_r <= addr_r + 6'd1;
                        else if (!rd_valid_r) begin
                            stem_done_r <= 1'b1; cnt_r <= TAIL; state_r <= S_TAIL;
                        end
                S_TAIL: if (cnt_r != 10'd0) cnt_r <= cnt_r - 10'd1;
                        else begin
                            res_we_r <= 1'b1; done_r <= 1'b1; fc_cnt_r <= FC_T; state_r <= S_IDLE;
                        end
                default: state_r <= S_IDLE;
            endcase
        end
    end

    assign o_stem_done = stem_done_r;
    assign o_done      = done_r;
    assign o_fc_busy   = (fc_cnt_r != 10'd0);
    assign o_res_we    = res_we_r;
    assign o_res_idx   = slot_r;
    assign o_res_data  = {slot_r, 4'h0, sum_r};
    assign o_img_cs    = img_cs;
    assign o_img_addr  = addr_r;
    assign o_wbuf_cs   = 1'b0;
    assign o_wbuf_addr = 11'd0;
    assign o_pbuf_cs   = 1'b0;
    assign o_pbuf_addr = 5'd0;
    assign o_fcm_cs    = 1'b0;
    assign o_fcm_addr  = 4'd0;
endmodule
`default_nettype wire
