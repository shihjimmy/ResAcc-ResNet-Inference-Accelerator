`timescale 1ns/1ps
//============================================================================
// core -- clk_dut domain shell, SPEC 6.1 / 6.8 / 6.17
//
//   * start detection (toggle from host_if after a 2-FF synchronizer), done output
//   * control FSM: load the model once after reset, then for each batch
//       load image -> run engine; prefetch the next image as soon as the
//       engine reports that the stem layer no longer needs IMG
//   * loader and the four shared memories (WBUF / PBUF / FCMEM / IMG):
//       loader writes, engine reads, never in the same cycle
//   * 16 x 32-bit result registers
//============================================================================
`default_nettype none
module core (
    input  wire          i_clk,
    input  wire          i_rst_n,
    // host_if
    input  wire          i_start_tgl,
    input  wire [3:0]    i_n_minus_1,
    output wire          o_done,
    output wire [511:0]  o_result_vec,
    // request FIFO (write side)
    input  wire          i_req_full,
    output wire          o_req_push,
    output wire [26:0]   o_req_data,
    // data FIFO (read side)
    input  wire          i_dat_empty,
    input  wire [255:0]  i_dat_data,
    output wire          o_dat_pop
);
    localparam S_LOAD_MODEL = 3'd0;
    localparam S_IDLE       = 3'd1;
    localparam S_WAIT_IMG   = 3'd2;
    localparam S_RUN        = 3'd3;
    localparam S_WAIT_FC    = 3'd4;

    //------------------------------------------------------------------------
    // wires between submodules
    //------------------------------------------------------------------------
    wire         start_sync;
    wire         ld_model_done, ld_img_done;
    wire         ld_wbuf_we, ld_pbuf_we, ld_fcm_we, ld_img_we;
    wire [10:0]  ld_wbuf_addr;
    wire [127:0] ld_wbuf_wdata;
    wire [4:0]   ld_pbuf_addr;
    wire [255:0] ld_pbuf_wdata;
    wire [3:0]   ld_fcm_addr;
    wire [255:0] ld_fcm_wdata;
    wire [5:0]   ld_img_addr;
    wire [511:0] ld_img_wdata;
    wire [63:0]  ld_img_wbe;
    wire [55:0]  shift_vec;
    wire [15:0]  conf_m;
    wire [7:0]   conf_s;
    wire [319:0] bfc_vec;

    wire         eng_stem_done, eng_done, eng_fc_busy;
    wire         res_we;
    wire [3:0]   res_idx;
    wire [31:0]  res_data;
    wire         en_img_cs, en_wbuf_cs, en_pbuf_cs, en_fcm_cs;
    wire [5:0]   en_img_addr;
    wire [10:0]  en_wbuf_addr;
    wire [4:0]   en_pbuf_addr;
    wire [3:0]   en_fcm_addr;
    wire [511:0] img_rdata;
    wire [127:0] wbuf_rdata;
    wire [255:0] pbuf_rdata;
    wire [255:0] fcm_rdata;

    //------------------------------------------------------------------------
    // control FSM
    //------------------------------------------------------------------------
    reg [2:0] state_r,      state_w;
    reg       start_d_r;                         // previous synchronized start toggle
    reg       start_pend_r, start_pend_w;        // start received, not yet served
    reg [3:0] nm1_r,        nm1_w;               // images in this batch - 1
    reg [3:0] n_r,          n_w;                 // current image
    reg       img_loaded_r, img_loaded_w;        // IMG holds the next image to run
    reg       done_r,       done_w;
    reg       cmd_model_r;
    reg       cmd_img_r,    cmd_img_w;
    reg [3:0] cmd_slot_r,   cmd_slot_w;
    reg       eng_start_r,  eng_start_w;

    wire start_pulse = start_sync ^ start_d_r;
    wire last_img    = (n_r == nm1_r);

    always @(*) begin
        state_w      = state_r;
        start_pend_w = start_pend_r;
        nm1_w        = nm1_r;
        n_w          = n_r;
        img_loaded_w = img_loaded_r;
        done_w       = done_r;
        cmd_img_w    = 1'b0;
        cmd_slot_w   = cmd_slot_r;
        eng_start_w  = 1'b0;

        // a start may arrive at any time (even while the model is loading)
        if (start_pulse) begin
            start_pend_w = 1'b1;
            nm1_w        = i_n_minus_1;
            done_w       = 1'b0;
        end
        if (ld_img_done) img_loaded_w = 1'b1;

        case (state_r)
            S_LOAD_MODEL: begin
                if (ld_model_done) state_w = S_IDLE;
            end
            S_IDLE: begin
                if (start_pend_r) begin
                    start_pend_w = 1'b0;
                    n_w          = 4'd0;
                    img_loaded_w = 1'b0;
                    cmd_img_w    = 1'b1;           // first image of the batch
                    cmd_slot_w   = 4'd0;
                    state_w      = S_WAIT_IMG;
                end
            end
            S_WAIT_IMG: begin
                if (img_loaded_r) begin
                    img_loaded_w = 1'b0;           // IMG now belongs to the engine
                    eng_start_w  = 1'b1;
                    state_w      = S_RUN;
                end
            end
            S_RUN: begin
                if (eng_stem_done && !last_img) begin   // IMG is free: prefetch the next image
                    cmd_img_w  = 1'b1;
                    cmd_slot_w = n_r + 4'd1;
                end
                if (eng_done) begin
                    if (!last_img) begin
                        n_w     = n_r + 4'd1;
                        state_w = S_WAIT_IMG;
                    end else begin
                        state_w = S_WAIT_FC;
                    end
                end
            end
            S_WAIT_FC: begin
                if (!eng_fc_busy) begin
                    done_w  = 1'b1;
                    state_w = S_IDLE;
                end
            end
            default: state_w = S_IDLE;
        endcase
    end

    // the model load command is issued once, in the first cycle after reset
    reg model_cmd_sent_r;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r          <= S_LOAD_MODEL;
            start_d_r        <= 1'b0;
            start_pend_r     <= 1'b0;
            nm1_r            <= 4'd0;
            n_r              <= 4'd0;
            img_loaded_r     <= 1'b0;
            done_r           <= 1'b0;
            cmd_model_r      <= 1'b0;
            cmd_img_r        <= 1'b0;
            cmd_slot_r       <= 4'd0;
            eng_start_r      <= 1'b0;
            model_cmd_sent_r <= 1'b0;
        end else begin
            state_r          <= state_w;
            start_d_r        <= start_sync;
            start_pend_r     <= start_pend_w;
            nm1_r            <= nm1_w;
            n_r              <= n_w;
            img_loaded_r     <= img_loaded_w;
            done_r           <= done_w;
            cmd_model_r      <= !model_cmd_sent_r;
            cmd_img_r        <= cmd_img_w;
            cmd_slot_r       <= cmd_slot_w;
            eng_start_r      <= eng_start_w;
            model_cmd_sent_r <= 1'b1;
        end
    end

    //------------------------------------------------------------------------
    // start synchronizer
    //------------------------------------------------------------------------
    sync_2ff #(.WIDTH(1)) u_start_sync (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_d     (i_start_tgl),
        .o_q     (start_sync)
    );

    //------------------------------------------------------------------------
    // loader
    //------------------------------------------------------------------------
    loader u_loader (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_cmd_model  (cmd_model_r),
        .i_cmd_img    (cmd_img_r),
        .i_img_slot   (cmd_slot_r),
        .o_model_done (ld_model_done),
        .o_img_done   (ld_img_done),
        .i_req_full   (i_req_full),
        .o_req_push   (o_req_push),
        .o_req_data   (o_req_data),
        .i_dat_empty  (i_dat_empty),
        .i_dat_data   (i_dat_data),
        .o_dat_pop    (o_dat_pop),
        .o_wbuf_we    (ld_wbuf_we),
        .o_wbuf_addr  (ld_wbuf_addr),
        .o_wbuf_wdata (ld_wbuf_wdata),
        .o_pbuf_we    (ld_pbuf_we),
        .o_pbuf_addr  (ld_pbuf_addr),
        .o_pbuf_wdata (ld_pbuf_wdata),
        .o_fcm_we     (ld_fcm_we),
        .o_fcm_addr   (ld_fcm_addr),
        .o_fcm_wdata  (ld_fcm_wdata),
        .o_img_we     (ld_img_we),
        .o_img_addr   (ld_img_addr),
        .o_img_wdata  (ld_img_wdata),
        .o_img_wbe    (ld_img_wbe),
        .o_shift_vec  (shift_vec),
        .o_conf_m     (conf_m),
        .o_conf_s     (conf_s),
        .o_bfc_vec    (bfc_vec)
    );

    //------------------------------------------------------------------------
    // engine
    //------------------------------------------------------------------------
    engine u_engine (
        .i_clk        (i_clk),
        .i_rst_n      (i_rst_n),
        .i_start      (eng_start_r),
        .i_slot       (n_r),
        .o_stem_done  (eng_stem_done),
        .o_done       (eng_done),
        .o_fc_busy    (eng_fc_busy),
        .o_res_we     (res_we),
        .o_res_idx    (res_idx),
        .o_res_data   (res_data),
        .o_img_cs     (en_img_cs),
        .o_img_addr   (en_img_addr),
        .i_img_rdata  (img_rdata),
        .o_wbuf_cs    (en_wbuf_cs),
        .o_wbuf_addr  (en_wbuf_addr),
        .i_wbuf_rdata (wbuf_rdata),
        .o_pbuf_cs    (en_pbuf_cs),
        .o_pbuf_addr  (en_pbuf_addr),
        .i_pbuf_rdata (pbuf_rdata),
        .o_fcm_cs     (en_fcm_cs),
        .o_fcm_addr   (en_fcm_addr),
        .i_fcm_rdata  (fcm_rdata),
        .i_shift_vec  (shift_vec),
        .i_conf_m     (conf_m),
        .i_conf_s     (conf_s),
        .i_bfc_vec    (bfc_vec)
    );

    //------------------------------------------------------------------------
    // shared memories: loader writes, engine reads
    //------------------------------------------------------------------------
    sram_sp #(.DEPTH(1227), .WIDTH(128)) u_wbuf (
        .i_clk   (i_clk),
        .i_cs    (ld_wbuf_we | en_wbuf_cs),
        .i_we    (ld_wbuf_we),
        .i_addr  (ld_wbuf_we ? ld_wbuf_addr : en_wbuf_addr),
        .i_wdata (ld_wbuf_wdata),
        .i_wbe   ({16{1'b1}}),
        .o_rdata (wbuf_rdata)
    );

    sram_sp #(.DEPTH(30), .WIDTH(256)) u_pbuf (
        .i_clk   (i_clk),
        .i_cs    (ld_pbuf_we | en_pbuf_cs),
        .i_we    (ld_pbuf_we),
        .i_addr  (ld_pbuf_we ? ld_pbuf_addr : en_pbuf_addr),
        .i_wdata (ld_pbuf_wdata),
        .i_wbe   ({32{1'b1}}),
        .o_rdata (pbuf_rdata)
    );

    sram_sp #(.DEPTH(10), .WIDTH(256)) u_fcm (
        .i_clk   (i_clk),
        .i_cs    (ld_fcm_we | en_fcm_cs),
        .i_we    (ld_fcm_we),
        .i_addr  (ld_fcm_we ? ld_fcm_addr : en_fcm_addr),
        .i_wdata (ld_fcm_wdata),
        .i_wbe   ({32{1'b1}}),
        .o_rdata (fcm_rdata)
    );

    sram_sp #(.DEPTH(48), .WIDTH(512)) u_img (
        .i_clk   (i_clk),
        .i_cs    (ld_img_we | en_img_cs),
        .i_we    (ld_img_we),
        .i_addr  (ld_img_we ? ld_img_addr : en_img_addr),
        .i_wdata (ld_img_wdata),
        .i_wbe   (ld_img_wbe),
        .o_rdata (img_rdata)
    );

`ifndef SYNTHESIS
    always @(posedge i_clk) begin
        if ((ld_wbuf_we & en_wbuf_cs) | (ld_pbuf_we & en_pbuf_cs) |
            (ld_fcm_we  & en_fcm_cs)  | (ld_img_we  & en_img_cs)) begin
            $display("[%0t] ERROR %m: loader and engine access the same memory", $time);
            $stop;
        end
    end
`endif

    //------------------------------------------------------------------------
    // result registers (data, no reset)
    //------------------------------------------------------------------------
    reg [31:0] result_r [0:15];

    always @(posedge i_clk) begin
        if (res_we) result_r[res_idx] <= res_data;
    end

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : g_result
            assign o_result_vec[gi*32 +: 32] = result_r[gi];
        end
    endgenerate

    assign o_done = done_r;
endmodule
`default_nettype wire
