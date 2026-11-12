`timescale 1ns/1ps
//============================================================================
// engine -- compute engine (wrapper), SPEC 6.9-6.17
//
//   engine_ctrl : decides what happens each cycle (FSM, counters, addresses, tokens)
//   engine_dp   : FM SRAMs, input select, PE array, requantizer, pool
//   fc_unit     : FC, argmax, confidence (overlaps with the next image)
// The shared SRAMs (IMG, WBUF, PBUF, FCMEM) live in core; their read ports
// are driven from here.
//============================================================================
`default_nettype none
module engine (
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
    //------------------------------------------------------------------------
    // wires
    //------------------------------------------------------------------------
    wire         fc_start, fc_busy;
    wire [3:0]   fc_slot;
    wire         fm_rd;
    wire [1:0]   fm_rd_sel;
    wire [5:0]   fm_rd_addr;
    wire         tk_valid, tk_first, tk_last, tk_final, tk_bank, tk_bset, tk_win_ld, tk_stem;
    wire [1:0]   tk_src;
    wire [47:0]  tk_sel;
    wire [7:0]   tk_vld;
    wire [1:0]   tk_dmode, tk_ddst, tk_dwstep;
    wire [5:0]   tk_dword, tk_dbyte;
    wire         tk_dpacked, tk_dog, tk_ds24;
    wire         par_we, par_set;
    wire [1:0]   par_word;
    wire         pool_clr;
    wire [383:0] pool_vec;
    wire         pipe_busy, drain_busy;

    //------------------------------------------------------------------------
    // control
    //------------------------------------------------------------------------
    engine_ctrl u_ctrl (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (i_start),
        .i_slot       (i_slot),
        .o_stem_done  (o_stem_done),
        .o_done       (o_done),
        .o_fc_start   (fc_start),
        .o_fc_slot    (fc_slot),
        .i_fc_busy    (fc_busy),
        .o_img_cs     (o_img_cs),
        .o_img_addr   (o_img_addr),
        .o_wbuf_cs    (o_wbuf_cs),
        .o_wbuf_addr  (o_wbuf_addr),
        .o_pbuf_cs    (o_pbuf_cs),
        .o_pbuf_addr  (o_pbuf_addr),
        .i_shift_vec  (i_shift_vec),
        .o_fm_rd      (fm_rd),
        .o_fm_rd_sel  (fm_rd_sel),
        .o_fm_rd_addr (fm_rd_addr),
        .o_tk_valid   (tk_valid),
        .o_tk_first   (tk_first),
        .o_tk_last    (tk_last),
        .o_tk_final   (tk_final),
        .o_tk_bank    (tk_bank),
        .o_tk_bset    (tk_bset),
        .o_tk_win_ld  (tk_win_ld),
        .o_tk_src     (tk_src),
        .o_tk_stem    (tk_stem),
        .o_tk_sel     (tk_sel),
        .o_tk_vld     (tk_vld),
        .o_tk_dmode   (tk_dmode),
        .o_tk_ddst    (tk_ddst),
        .o_tk_dword   (tk_dword),
        .o_tk_dbyte   (tk_dbyte),
        .o_tk_dpacked (tk_dpacked),
        .o_tk_dwstep  (tk_dwstep),
        .o_tk_dog     (tk_dog),
        .o_tk_ds24    (tk_ds24),
        .o_par_we     (par_we),
        .o_par_set    (par_set),
        .o_par_word   (par_word),
        .o_pool_clr   (pool_clr),
        .i_pipe_busy  (pipe_busy),
        .i_drain_busy (drain_busy)
    );

    //------------------------------------------------------------------------
    // data path
    //------------------------------------------------------------------------
    engine_dp u_dp (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_tk_valid   (tk_valid),
        .i_tk_first   (tk_first),
        .i_tk_last    (tk_last),
        .i_tk_final   (tk_final),
        .i_tk_bank    (tk_bank),
        .i_tk_bset    (tk_bset),
        .i_tk_win_ld  (tk_win_ld),
        .i_tk_src     (tk_src),
        .i_tk_stem    (tk_stem),
        .i_tk_sel     (tk_sel),
        .i_tk_vld     (tk_vld),
        .i_tk_dmode   (tk_dmode),
        .i_tk_ddst    (tk_ddst),
        .i_tk_dword   (tk_dword),
        .i_tk_dbyte   (tk_dbyte),
        .i_tk_dpacked (tk_dpacked),
        .i_tk_dwstep  (tk_dwstep),
        .i_tk_dog     (tk_dog),
        .i_tk_ds24    (tk_ds24),
        .i_fm_rd      (fm_rd),
        .i_fm_rd_sel  (fm_rd_sel),
        .i_fm_rd_addr (fm_rd_addr),
        .i_img_rdata  (i_img_rdata),
        .i_wbuf_rdata (i_wbuf_rdata),
        .i_par_we     (par_we),
        .i_par_set    (par_set),
        .i_par_word   (par_word),
        .i_pbuf_rdata (i_pbuf_rdata),
        .i_pool_clr   (pool_clr),
        .o_pool_vec   (pool_vec),
        .o_pipe_busy  (pipe_busy),
        .o_drain_busy (drain_busy)
    );

    //------------------------------------------------------------------------
    // fully connected layer
    //------------------------------------------------------------------------
    fc_unit u_fc (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_start    (fc_start),
        .i_slot     (fc_slot),
        .i_pool_vec (pool_vec),
        .o_busy     (fc_busy),
        .o_fcm_cs   (o_fcm_cs),
        .o_fcm_addr (o_fcm_addr),
        .i_fcm_rdata(i_fcm_rdata),
        .i_bfc_vec  (i_bfc_vec),
        .i_conf_m   (i_conf_m),
        .i_conf_s   (i_conf_s),
        .o_res_we   (o_res_we),
        .o_res_idx  (o_res_idx),
        .o_res_data (o_res_data)
    );

    assign o_fc_busy = fc_busy;
endmodule
`default_nettype wire
