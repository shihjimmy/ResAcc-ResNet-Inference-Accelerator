`timescale 1ns/1ps
//============================================================================
// engine_dp -- data path of the engine, SPEC 6.11-6.14
//
// Does what the tokens from engine_ctrl say; makes no decisions of its own,
// except the small drain sequencer that is launched by a token.
//
// Compute pipeline (one token per cycle):
//   S0  (engine_ctrl) SRAM read addresses, token
//   S1  SRAM data arrives; window register (loaded when kx == 0);
//       8 x 64:1 byte select -> 8 operands
//   S2  PE array multiply-accumulate
// Drain pipeline (after the last token of a batch):
//   D0  select one PE row (and its m) into the lanes
//   D1  multiply   (requant_lane stage 1)
//   D2  round, shift, clamp (requant_lane stage 2)
//   D3  write 8 bytes into an FM SRAM, or add them to the pool
// The token travels with its data (tk1_* at S1, tk2_* at S2), so every stage
// uses the control bits that belong to the data it is processing.
//============================================================================
`default_nettype none
module engine_dp (
    input  wire          i_clk,
    input  wire          i_rst_n,
    // token from engine_ctrl (S0)
    input  wire          i_tk_valid,
    input  wire          i_tk_first,
    input  wire          i_tk_last,
    input  wire          i_tk_final,
    input  wire          i_tk_bank,
    input  wire          i_tk_bset,
    input  wire          i_tk_win_ld,
    input  wire [1:0]    i_tk_src,
    input  wire          i_tk_stem,
    input  wire [47:0]   i_tk_sel,
    input  wire [7:0]    i_tk_vld,
    input  wire [1:0]    i_tk_dmode,
    input  wire [1:0]    i_tk_ddst,
    input  wire [5:0]    i_tk_dword,
    input  wire [5:0]    i_tk_dbyte,
    input  wire          i_tk_dpacked,
    input  wire [1:0]    i_tk_dwstep,
    input  wire          i_tk_dog,
    input  wire          i_tk_ds24,
    // FM read request (S0)
    input  wire          i_fm_rd,
    input  wire [1:0]    i_fm_rd_sel,
    input  wire [5:0]    i_fm_rd_addr,
    // SRAM data from core (arrives at S1)
    input  wire [511:0]  i_img_rdata,
    input  wire [127:0]  i_wbuf_rdata,
    // parameter load
    input  wire          i_par_we,
    input  wire          i_par_set,
    input  wire [1:0]    i_par_word,
    input  wire [255:0]  i_pbuf_rdata,
    // pool
    input  wire          i_pool_clr,
    output wire [383:0]  o_pool_vec,          // 32 x 12-bit
    // status
    output wire          o_pipe_busy,
    output wire          o_drain_busy
);
    localparam DM_PLAIN = 2'd0;
    localparam DM_RES   = 2'd1;
    localparam DM_POOL  = 2'd2;

    //========================================================================
    // FM SRAMs: read by S0 requests, written by D3
    //========================================================================
    wire         wr_v;                         // D3 write valid (declared below)
    wire [1:0]   wr_dst;
    wire [5:0]   wr_word;
    wire [511:0] wr_data;
    wire [63:0]  wr_be;
    wire [511:0] fma_rdata, fmb_rdata, fmc_rdata;

    wire rd_a = i_fm_rd && (i_fm_rd_sel == 2'd1);
    wire rd_b = i_fm_rd && (i_fm_rd_sel == 2'd2);
    wire rd_c = i_fm_rd && (i_fm_rd_sel == 2'd3);
    wire wr_a = wr_v && (wr_dst == 2'd1);
    wire wr_b = wr_v && (wr_dst == 2'd2);
    wire wr_c = wr_v && (wr_dst == 2'd3);

    sram_sp #(.DEPTH(64), .WIDTH(512)) u_fm_a (
        .i_clk   (i_clk),
        .i_cs    (rd_a | wr_a),
        .i_we    (wr_a),
        .i_addr  (wr_a ? wr_word : i_fm_rd_addr),
        .i_wdata (wr_data),
        .i_wbe   (wr_be),
        .o_rdata (fma_rdata)
    );

    sram_sp #(.DEPTH(16), .WIDTH(512)) u_fm_b (
        .i_clk   (i_clk),
        .i_cs    (rd_b | wr_b),
        .i_we    (wr_b),
        .i_addr  (wr_b ? wr_word[3:0] : i_fm_rd_addr[3:0]),
        .i_wdata (wr_data),
        .i_wbe   (wr_be),
        .o_rdata (fmb_rdata)
    );

    sram_sp #(.DEPTH(16), .WIDTH(512)) u_fm_c (
        .i_clk   (i_clk),
        .i_cs    (rd_c | wr_c),
        .i_we    (wr_c),
        .i_addr  (wr_c ? wr_word[3:0] : i_fm_rd_addr[3:0]),
        .i_wdata (wr_data),
        .i_wbe   (wr_be),
        .o_rdata (fmc_rdata)
    );

`ifndef SYNTHESIS
    always @(posedge i_clk) begin
        if ((rd_a & wr_a) | (rd_b & wr_b) | (rd_c & wr_c)) begin
            $display("[%0t] ERROR %m: FM read and write in the same cycle", $time);
            $stop;
        end
    end
`endif

    //========================================================================
    // S1: token, window register, input select
    //========================================================================
    reg         tk1_valid_r;
    reg         tk1_first_r, tk1_last_r, tk1_final_r, tk1_bank_r, tk1_bset_r, tk1_win_ld_r, tk1_stem_r;
    reg  [1:0]  tk1_src_r;
    reg  [47:0] tk1_sel_r;
    reg  [7:0]  tk1_vld_r;
    reg  [1:0]  tk1_dmode_r, tk1_ddst_r, tk1_dwstep_r;
    reg  [5:0]  tk1_dword_r, tk1_dbyte_r;
    reg         tk1_dpacked_r, tk1_dog_r, tk1_ds24_r;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) tk1_valid_r <= 1'b0;
        else          tk1_valid_r <= i_tk_valid;
    end

    always @(posedge i_clk) begin
        tk1_first_r   <= i_tk_first;
        tk1_last_r    <= i_tk_last;
        tk1_final_r   <= i_tk_final;
        tk1_bank_r    <= i_tk_bank;
        tk1_bset_r    <= i_tk_bset;
        tk1_win_ld_r  <= i_tk_win_ld;
        tk1_stem_r    <= i_tk_stem;
        tk1_src_r     <= i_tk_src;
        tk1_sel_r     <= i_tk_sel;
        tk1_vld_r     <= i_tk_vld;
        tk1_dmode_r   <= i_tk_dmode;
        tk1_ddst_r    <= i_tk_ddst;
        tk1_dwstep_r  <= i_tk_dwstep;
        tk1_dword_r   <= i_tk_dword;
        tk1_dbyte_r   <= i_tk_dbyte;
        tk1_dpacked_r <= i_tk_dpacked;
        tk1_dog_r     <= i_tk_dog;
        tk1_ds24_r    <= i_tk_ds24;
    end

    // window: newly read word when kx == 0, otherwise the held word
    reg  [511:0] window_r;
    reg  [511:0] src_data;

    always @(*) begin
        case (tk1_src_r)
            2'd0:    src_data = i_img_rdata;
            2'd1:    src_data = fma_rdata;
            2'd2:    src_data = fmb_rdata;
            default: src_data = fmc_rdata;
        endcase
    end

    wire [511:0] window = tk1_win_ld_r ? src_data : window_r;

    always @(posedge i_clk) begin
        if (tk1_valid_r) window_r <= window;
    end

    // 8 operands: 9-bit signed, 0 for padding, pixel - 128 for the stem
    wire [71:0] act;

    genvar p;
    generate
        for (p = 0; p < 8; p = p + 1) begin : g_sel
            wire [7:0] byte_val = window[8*tk1_sel_r[6*p +: 6] +: 8];
            assign act[9*p +: 9] = !tk1_vld_r[p] ? 9'd0 :
                                   tk1_stem_r    ? ({1'b0, byte_val} - 9'd128) :
                                                   {1'b0, byte_val};
        end
    endgenerate

    //========================================================================
    // S2: operands and token
    //========================================================================
    reg          tk2_valid_r;
    reg          tk2_first_r, tk2_last_r, tk2_final_r, tk2_bank_r, tk2_bset_r;
    reg  [1:0]   tk2_dmode_r, tk2_ddst_r, tk2_dwstep_r;
    reg  [5:0]   tk2_dword_r, tk2_dbyte_r;
    reg          tk2_dpacked_r, tk2_dog_r, tk2_ds24_r;
    reg  [127:0] weight_2_r;
    reg  [71:0]  act_2_r;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) tk2_valid_r <= 1'b0;
        else          tk2_valid_r <= tk1_valid_r;
    end

    always @(posedge i_clk) begin
        tk2_first_r   <= tk1_first_r;
        tk2_last_r    <= tk1_last_r;
        tk2_final_r   <= tk1_final_r;
        tk2_bank_r    <= tk1_bank_r;
        tk2_bset_r    <= tk1_bset_r;
        tk2_dmode_r   <= tk1_dmode_r;
        tk2_ddst_r    <= tk1_ddst_r;
        tk2_dwstep_r  <= tk1_dwstep_r;
        tk2_dword_r   <= tk1_dword_r;
        tk2_dbyte_r   <= tk1_dbyte_r;
        tk2_dpacked_r <= tk1_dpacked_r;
        tk2_dog_r     <= tk1_dog_r;
        tk2_ds24_r    <= tk1_ds24_r;
        weight_2_r    <= i_wbuf_rdata;
        act_2_r       <= act;
    end

    //========================================================================
    // parameters: set 0 = main / plain layer, set 1 = shortcut
    //========================================================================
    reg [19:0] bias0_r [0:15];
    reg [19:0] bias1_r [0:15];
    reg [15:0] m0_r    [0:15];
    reg [15:0] m1_r    [0:15];

    genvar k;
    generate
        for (k = 0; k < 8; k = k + 1) begin : g_par_bias
            always @(posedge i_clk) begin
                if (i_par_we && i_par_word == 2'd0 && !i_par_set) bias0_r[k]     <= i_pbuf_rdata[32*k +: 20];
                if (i_par_we && i_par_word == 2'd1 && !i_par_set) bias0_r[k + 8] <= i_pbuf_rdata[32*k +: 20];
                if (i_par_we && i_par_word == 2'd0 &&  i_par_set) bias1_r[k]     <= i_pbuf_rdata[32*k +: 20];
                if (i_par_we && i_par_word == 2'd1 &&  i_par_set) bias1_r[k + 8] <= i_pbuf_rdata[32*k +: 20];
            end
        end
        for (k = 0; k < 16; k = k + 1) begin : g_par_m
            always @(posedge i_clk) begin
                if (i_par_we && i_par_word == 2'd2 && !i_par_set) m0_r[k] <= i_pbuf_rdata[16*k +: 16];
                if (i_par_we && i_par_word == 2'd2 &&  i_par_set) m1_r[k] <= i_pbuf_rdata[16*k +: 16];
            end
        end
    endgenerate

    //========================================================================
    // PE array (S2)
    //========================================================================
    wire [319:0] bias_vec;
    wire [159:0] row0, row1;
    reg  [3:0]   dr_row_r;                    // row being drained (D0)

    generate
        for (k = 0; k < 16; k = k + 1) begin : g_bias
            assign bias_vec[20*k +: 20] = tk2_bset_r ? bias1_r[k] : bias0_r[k];
        end
    endgenerate

    pe_array u_pe_array (
        .i_clk      (i_clk),
        .i_en       (tk2_valid_r),
        .i_bank     (tk2_bank_r),
        .i_first    (tk2_first_r),
        .i_weight   (weight_2_r),
        .i_act      (act_2_r),
        .i_bias     (bias_vec),
        .i_rd_row   (dr_row_r),
        .o_rd_bank0 (row0),
        .o_rd_bank1 (row1)
    );

    //========================================================================
    // drain sequencer (D0), launched by the last token of a batch
    //   plain   : row r in one cycle (bank of the batch, m set 0)
    //   residual: row r in two cycles: phase 0 = shortcut (bank 1, m set 1),
    //                                   phase 1 = main (bank 0, m set 0) + partial
    //========================================================================
    wire launch = tk2_valid_r && tk2_last_r && tk2_final_r;

    reg        dr_act_r,   dr_act_w;
    reg        dr_ph_r,    dr_ph_w;
    reg  [3:0] dr_row_w;
    // descriptor of the batch being drained (data registers)
    reg  [1:0] ds_mode_r, ds_dst_r, ds_wstep_r;
    reg  [5:0] ds_word_r, ds_byte_r;
    reg        ds_packed_r, ds_og_r, ds_s24_r, ds_bank_r;

    wire ds_res = (ds_mode_r != DM_PLAIN);

    always @(*) begin
        dr_act_w = dr_act_r;
        dr_ph_w  = dr_ph_r;
        dr_row_w = dr_row_r;
        if (launch) begin
            dr_act_w = 1'b1;
            dr_ph_w  = 1'b0;
            dr_row_w = 4'd0;
        end else if (dr_act_r) begin
            if (ds_res && !dr_ph_r) begin
                dr_ph_w = 1'b1;
            end else begin
                dr_ph_w = 1'b0;
                if (dr_row_r == 4'd15) dr_act_w = 1'b0;
                else                   dr_row_w = dr_row_r + 4'd1;
            end
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            dr_act_r <= 1'b0;
            dr_ph_r  <= 1'b0;
            dr_row_r <= 4'd0;
        end else begin
            dr_act_r <= dr_act_w;
            dr_ph_r  <= dr_ph_w;
            dr_row_r <= dr_row_w;
        end
    end

    always @(posedge i_clk) begin
        if (launch) begin
            ds_mode_r   <= tk2_dmode_r;
            ds_dst_r    <= tk2_ddst_r;
            ds_wstep_r  <= tk2_dwstep_r;
            ds_word_r   <= tk2_dword_r;
            ds_byte_r   <= tk2_dbyte_r;
            ds_packed_r <= tk2_dpacked_r;
            ds_og_r     <= tk2_dog_r;
            ds_s24_r    <= tk2_ds24_r;
            ds_bank_r   <= tk2_bank_r;
        end
    end

`ifndef SYNTHESIS
    always @(posedge i_clk) begin
        if (launch && o_drain_busy) begin
            $display("[%0t] ERROR %m: drain launched while the previous one is busy", $time);
            $stop;
        end
    end
`endif

    //========================================================================
    // D0 -> lanes: register the selected row and m
    //========================================================================
    wire        use_sc   = ds_res && !dr_ph_r;                       // residual phase 0
    wire        use_b1   = use_sc || (!ds_res && ds_bank_r);         // read bank 1
    wire [159:0] row_sel = use_b1 ? row1 : row0;
    wire [15:0] m_sel    = use_sc ? m1_r[dr_row_r] : m0_r[dr_row_r];

    reg         ln_valid_r;
    reg         ln_add_r, ln_out_r;
    reg [159:0] ln_acc_r;
    reg [15:0]  ln_m_r;
    reg [3:0]   row_0_r, row_1_r, row_2_r;   // row index travelling with the lanes

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) ln_valid_r <= 1'b0;
        else          ln_valid_r <= dr_act_r;
    end

    always @(posedge i_clk) begin
        ln_add_r <= ds_res && dr_ph_r;
        ln_out_r <= !ds_res || dr_ph_r;
        ln_acc_r <= row_sel;
        ln_m_r   <= m_sel;
        row_0_r  <= dr_row_r;
        row_1_r  <= row_0_r;
        row_2_r  <= row_1_r;
    end

    //========================================================================
    // D1, D2: eight requantization lanes
    //========================================================================
    wire [7:0]  ln_ovalid;
    wire [63:0] y;

    generate
        for (p = 0; p < 8; p = p + 1) begin : g_lane
            requant_lane u_lane (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_valid (ln_valid_r),
                .i_acc   (ln_acc_r[20*p +: 20]),
                .i_m     (ln_m_r),
                .i_add   (ln_add_r),
                .i_out   (ln_out_r),
                .i_s24   (ds_s24_r),
                .o_valid (ln_ovalid[p]),
                .o_y     (y[8*p +: 8])
            );
        end
    endgenerate

    // valid bits of the drain pipeline (for the busy status)
    reg v1_r;
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) v1_r <= 1'b0;
        else          v1_r <= ln_valid_r;
    end

    //========================================================================
    // D3: write back or pool
    //   the 8 outputs are 8 consecutive bytes aligned to 8, so the data is the
    //   8 bytes repeated 8 times and the byte enable selects the right slot
    //========================================================================
    wire       out_v    = ln_ovalid[0];
    wire [5:0] row_word = ds_packed_r ? (ds_word_r + {4'd0, row_2_r[3:2]})
                                      : (ds_word_r + ({2'b00, row_2_r} << ds_wstep_r));
    wire [5:0] row_byte = ds_packed_r ? (ds_byte_r + {row_2_r[1:0], 4'b0000}) : ds_byte_r;

    assign wr_v    = out_v && (ds_mode_r != DM_POOL);
    assign wr_dst  = ds_dst_r;
    assign wr_word = row_word;
    assign wr_data = {8{y}};

    generate
        for (k = 0; k < 8; k = k + 1) begin : g_wbe
            assign wr_be[8*k +: 8] = {8{row_byte[5:3] == k}};
        end
    endgenerate

    // pool: 32 x 12-bit sums of the l2 block outputs
    reg  [11:0] pool_r [0:31];
    wire [10:0] y_sum  = y[7:0]   + y[15:8]  + y[23:16] + y[31:24]
                       + y[39:32] + y[47:40] + y[55:48] + y[63:56];
    wire [4:0]  pool_c = {ds_og_r, row_2_r};
    wire        pool_v = out_v && (ds_mode_r == DM_POOL);

    generate
        for (k = 0; k < 32; k = k + 1) begin : g_pool
            always @(posedge i_clk) begin
                if (i_pool_clr)                  pool_r[k] <= 12'd0;
                else if (pool_v && pool_c == k)  pool_r[k] <= pool_r[k] + {1'b0, y_sum};
            end
            assign o_pool_vec[12*k +: 12] = pool_r[k];
        end
    endgenerate

    //========================================================================
    // status
    //========================================================================
    assign o_pipe_busy  = tk1_valid_r || tk2_valid_r;
    assign o_drain_busy = dr_act_r || ln_valid_r || v1_r || out_v;
endmodule
`default_nettype wire
