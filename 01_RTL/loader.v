`timescale 1ns/1ps
//============================================================================
// loader -- writes DRAM read data into the on-chip memories (clk_dut), SPEC 6.6/6.7
//
// Two jobs, one at a time (the core FSM guarantees this):
//   J_MODEL : 6 read requests (WSEC: 5 bursts, PSEC: 1 burst), 657 beats
//               beat   0..613 -> WBUF (each beat = two 128-bit words, 2 cycles)
//               beat 614      -> header registers (shift per layer, conf_m, conf_s)
//               beat 615..644 -> PBUF[0..29]
//               beat 645..654 -> FCMEM[0..9]
//               beat 655..656 -> FC bias registers
//   J_IMG   : 1 read request (0x1000 * slot, 96 beats)
//               beat b = c*32 + y -> IMG word c*16 + y/2, low half (y even) or high half (y odd)
//============================================================================
`default_nettype none
module loader (
    input  wire          i_clk,
    input  wire          i_rst_n,
    // commands from the core FSM
    input  wire          i_cmd_model,          // pulse: load the model
    input  wire          i_cmd_img,            // pulse: load one image
    input  wire [3:0]    i_img_slot,           // image slot n (DRAM byte address 0x1000*n)
    output wire          o_model_done,         // level: model loaded
    output wire          o_img_done,           // pulse: image loaded
    // request FIFO (write side)
    input  wire          i_req_full,
    output wire          o_req_push,
    output wire [26:0]   o_req_data,
    // data FIFO (read side)
    input  wire          i_dat_empty,
    input  wire [255:0]  i_dat_data,
    output wire          o_dat_pop,
    // memory write ports
    output wire          o_wbuf_we,
    output wire [10:0]   o_wbuf_addr,
    output wire [127:0]  o_wbuf_wdata,
    output wire          o_pbuf_we,
    output wire [4:0]    o_pbuf_addr,
    output wire [255:0]  o_pbuf_wdata,
    output wire          o_fcm_we,
    output wire [3:0]    o_fcm_addr,
    output wire [255:0]  o_fcm_wdata,
    output wire          o_img_we,
    output wire [5:0]    o_img_addr,
    output wire [511:0]  o_img_wdata,
    output wire [63:0]   o_img_wbe,
    // header and FC bias registers
    output wire [55:0]   o_shift_vec,          // byte k = shift of layer k
    output wire [15:0]   o_conf_m,
    output wire [7:0]    o_conf_s,
    output wire [319:0]  o_bfc_vec             // FC bias k = o_bfc_vec[32k +: 32]
);
    localparam WSEC_BEATS = 10'd614;
    localparam PSEC_BEATS = 10'd43;
    localparam IMG_BEATS  = 10'd96;
    localparam WBUF_WORDS = 11'd1227;

    localparam J_NONE  = 2'd0;
    localparam J_MODEL = 2'd1;
    localparam J_IMG   = 2'd2;

    //------------------------------------------------------------------------
    // registers
    //------------------------------------------------------------------------
    reg [1:0]   job_r,        job_w;
    reg [9:0]   beat_r,       beat_w;        // beat being consumed
    reg         half_r,       half_w;        // WBUF: 0 = low 128 bits, 1 = high 128 bits
    reg [2:0]   req_idx_r,    req_idx_w;     // requests already pushed
    reg [3:0]   slot_r,       slot_w;
    reg         model_done_r, model_done_w;
    reg         img_done_r,   img_done_w;
    // data registers (no reset; written before use)
    reg [55:0]  shift_vec_r;
    reg [15:0]  conf_m_r;
    reg [7:0]   conf_s_r;
    reg [319:0] bfc_vec_r;

    //------------------------------------------------------------------------
    // read requests
    //------------------------------------------------------------------------
    wire [2:0] req_total = (job_r == J_MODEL) ? 3'd6 : 3'd1;
    reg  [26:0] req_data;

    always @(*) begin
        if (job_r == J_MODEL) begin
            case (req_idx_r)
                3'd0:    req_data = {19'h10000, 8'd127};
                3'd1:    req_data = {19'h11000, 8'd127};
                3'd2:    req_data = {19'h12000, 8'd127};
                3'd3:    req_data = {19'h13000, 8'd127};
                3'd4:    req_data = {19'h14000, 8'd101};      // 614 - 4*128 = 102 beats
                default: req_data = {19'h15000, 8'd42};       // PSEC, 43 beats
            endcase
        end else begin
            req_data = {3'b000, slot_r, 12'h000, 8'd95};      // 0x1000*slot, 96 beats
        end
    end

    wire req_push = (job_r != J_NONE) && (req_idx_r < req_total) && !i_req_full;

    //------------------------------------------------------------------------
    // beat routing
    //------------------------------------------------------------------------
    wire        have      = (job_r != J_NONE) && !i_dat_empty;
    wire        in_wsec   = (job_r == J_MODEL) && (beat_r < WSEC_BEATS);
    wire        in_psec   = (job_r == J_MODEL) && !in_wsec;
    wire [9:0]  pbeat     = beat_r - WSEC_BEATS;               // beat index inside PSEC
    wire [9:0]  fbeat     = pbeat - 10'd31;                    // beat index inside the FC weights
    wire [10:0] wword     = {beat_r, half_r};                  // WBUF word = 2*beat + half
    wire [9:0]  last_beat = (job_r == J_MODEL) ? (WSEC_BEATS + PSEC_BEATS - 10'd1) : (IMG_BEATS - 10'd1);
    // WBUF beats take two cycles: the beat is consumed after the high half
    wire        beat_end  = have && !(in_wsec && !half_r);

    //------------------------------------------------------------------------
    // next state
    //------------------------------------------------------------------------
    always @(*) begin
        job_w        = job_r;
        beat_w       = beat_r;
        half_w       = half_r;
        req_idx_w    = req_idx_r;
        slot_w       = slot_r;
        model_done_w = model_done_r;
        img_done_w   = 1'b0;

        if (job_r == J_NONE) begin
            beat_w    = 10'd0;
            half_w    = 1'b0;
            req_idx_w = 3'd0;
            if (i_cmd_model) begin
                job_w = J_MODEL;
            end else if (i_cmd_img) begin
                job_w  = J_IMG;
                slot_w = i_img_slot;
            end
        end else begin
            if (req_push) req_idx_w = req_idx_r + 3'd1;
            if (have && in_wsec && !half_r) half_w = 1'b1;
            if (beat_end) begin
                half_w = 1'b0;
                if (beat_r == last_beat) begin
                    job_w = J_NONE;
                    if (job_r == J_MODEL) model_done_w = 1'b1;
                    else                  img_done_w   = 1'b1;
                end else begin
                    beat_w = beat_r + 10'd1;
                end
            end
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            job_r        <= J_NONE;
            beat_r       <= 10'd0;
            half_r       <= 1'b0;
            req_idx_r    <= 3'd0;
            slot_r       <= 4'd0;
            model_done_r <= 1'b0;
            img_done_r   <= 1'b0;
        end else begin
            job_r        <= job_w;
            beat_r       <= beat_w;
            half_r       <= half_w;
            req_idx_r    <= req_idx_w;
            slot_r       <= slot_w;
            model_done_r <= model_done_w;
            img_done_r   <= img_done_w;
        end
    end

    // header and FC bias (data registers)
    always @(posedge i_clk) begin
        if (have && in_psec && pbeat == 10'd0) begin
            shift_vec_r <= i_dat_data[55:0];
            conf_m_r    <= i_dat_data[79:64];
            conf_s_r    <= i_dat_data[87:80];
        end
        if (have && in_psec && pbeat == 10'd41) bfc_vec_r[255:0]   <= i_dat_data;          // bias 0..7
        if (have && in_psec && pbeat == 10'd42) bfc_vec_r[319:256] <= i_dat_data[63:0];    // bias 8..9
    end

    //------------------------------------------------------------------------
    // outputs
    //------------------------------------------------------------------------
    assign o_req_push   = req_push;
    assign o_req_data   = req_data;
    assign o_dat_pop    = beat_end;

    assign o_wbuf_we    = have && in_wsec && (wword < WBUF_WORDS);   // last beat has no high word
    assign o_wbuf_addr  = wword;
    assign o_wbuf_wdata = half_r ? i_dat_data[255:128] : i_dat_data[127:0];

    assign o_pbuf_we    = have && in_psec && (pbeat >= 10'd1) && (pbeat <= 10'd30);
    assign o_pbuf_addr  = pbeat[4:0] - 5'd1;
    assign o_pbuf_wdata = i_dat_data;

    assign o_fcm_we     = have && in_psec && (pbeat >= 10'd31) && (pbeat <= 10'd40);
    assign o_fcm_addr   = fbeat[3:0];
    assign o_fcm_wdata  = i_dat_data;

    // image beat b = c*32 + y  ->  word = c*16 + y/2
    assign o_img_we     = have && (job_r == J_IMG);
    assign o_img_addr   = {beat_r[6:5], beat_r[4:1]};
    assign o_img_wdata  = {i_dat_data, i_dat_data};
    assign o_img_wbe    = beat_r[0] ? {{32{1'b1}}, {32{1'b0}}} : {{32{1'b0}}, {32{1'b1}}};

    assign o_model_done = model_done_r;
    assign o_img_done   = img_done_r;
    assign o_shift_vec  = shift_vec_r;
    assign o_conf_m     = conf_m_r;
    assign o_conf_s     = conf_s_r;
    assign o_bfc_vec    = bfc_vec_r;

`ifndef SYNTHESIS
    // the engine's requantizer only supports shift values 23 and 24 (SPEC 6.14)
    integer k;
    always @(posedge i_clk) begin
        if (have && in_psec && pbeat == 10'd0) begin
            for (k = 0; k < 7; k = k + 1) begin
                if (i_dat_data[8*k +: 8] != 8'd23 && i_dat_data[8*k +: 8] != 8'd24) begin
                    $display("[%0t] ERROR %m: layer %0d shift %0d not supported (must be 23 or 24)",
                             $time, k, i_dat_data[8*k +: 8]);
                    $stop;
                end
            end
        end
    end
`endif
endmodule
`default_nettype wire
