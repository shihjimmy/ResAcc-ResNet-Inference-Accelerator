`timescale 1ns/1ps
//============================================================================
// engine_ctrl -- control path of the engine, SPEC 6.10-6.17
//
// Decides what happens every cycle; never touches feature data.
//
// One image = 5 steps:
//   step 0  stem            plain     IMG        -> FM_A
//   step 1  l1_a            plain     FM_A       -> FM_B
//   step 2  l1_sc + l1_b    residual  FM_A, FM_B -> FM_C
//   step 3  l2_a            plain     FM_C       -> FM_B
//   step 4  l2_sc + l2_b    residual  FM_C, FM_B -> pool
// A step is a list of batches (16 output channels x 8 output pixels).
// A batch is one pass (plain) or two passes (residual: shortcut, then main).
// A pass walks (i, ky, kx) and issues one token per cycle.
//
// Outputs at S0 (the issue cycle):
//   * SRAM read addresses (WBUF, IMG or FM)
//   * a token describing this cycle; engine_dp carries it down its pipeline
//     together with the data, so no control signal has to be delayed here
// Parameters (bias, m) are loaded into engine_dp before each (step, og);
// they are only changed when the pipeline and the drain are empty.
//
// Layer index L: 0 stem, 1 l1_a, 2 l1_sc, 3 l1_b, 4 l2_a, 5 l2_sc, 6 l2_b
//============================================================================
`default_nettype none
module engine_ctrl (
    input  wire          i_clk,
    input  wire          i_rst_n,
    // from / to core
    input  wire          i_start,
    input  wire [3:0]    i_slot,
    output wire          o_stem_done,          // pulse: IMG no longer needed
    output wire          o_done,               // pulse: all layers done, FC started
    // FC
    output wire          o_fc_start,
    output wire [3:0]    o_fc_slot,
    input  wire          i_fc_busy,
    // shared memory read ports (in core)
    output wire          o_img_cs,
    output wire [5:0]    o_img_addr,
    output wire          o_wbuf_cs,
    output wire [10:0]   o_wbuf_addr,
    output wire          o_pbuf_cs,
    output wire [4:0]    o_pbuf_addr,
    input  wire [55:0]   i_shift_vec,
    // FM read port (FM SRAMs are in engine_dp)
    output wire          o_fm_rd,
    output wire [1:0]    o_fm_rd_sel,          // 1: FM_A, 2: FM_B, 3: FM_C
    output wire [5:0]    o_fm_rd_addr,
    // token (S0)
    output wire          o_tk_valid,
    output wire          o_tk_first,           // first cycle of the pass: start from bias
    output wire          o_tk_last,            // last cycle of the pass
    output wire          o_tk_final,           // pass ends the batch: start the drain
    output wire          o_tk_bank,            // accumulator bank
    output wire          o_tk_bset,            // parameter set (0: main / plain, 1: shortcut)
    output wire          o_tk_win_ld,          // load the window register (kx == 0)
    output wire [1:0]    o_tk_src,             // 0: IMG, 1: FM_A, 2: FM_B, 3: FM_C
    output wire          o_tk_stem,            // operand = pixel - 128
    output wire [47:0]   o_tk_sel,             // byte index in the window, 6 bits per column
    output wire [7:0]    o_tk_vld,             // 0: padding
    // drain descriptor (used with last & final)
    output wire [1:0]    o_tk_dmode,           // 0: plain, 1: residual, 2: residual + pool
    output wire [1:0]    o_tk_ddst,            // 1: FM_A, 2: FM_B, 3: FM_C
    output wire [5:0]    o_tk_dword,           // FM word of output channel og*16
    output wire [5:0]    o_tk_dbyte,           // byte offset of the 8 outputs in that word
    output wire          o_tk_dpacked,         // 4 channels per word (4x4 maps)
    output wire [1:0]    o_tk_dwstep,          // log2(words per channel) when not packed
    output wire          o_tk_dog,
    output wire          o_tk_ds24,            // shift is 24 (else 23)
    // parameter load into engine_dp (aligned with i_pbuf_rdata)
    output wire          o_par_we,
    output wire          o_par_set,
    output wire [1:0]    o_par_word,           // 0: bias 0..7, 1: bias 8..15, 2: m 0..15
    output wire          o_pool_clr,
    // status from engine_dp
    input  wire          i_pipe_busy,
    input  wire          i_drain_busy
);
    localparam S_IDLE     = 3'd0;
    localparam S_PAR_WAIT = 3'd1;
    localparam S_PAR      = 3'd2;
    localparam S_BATCH    = 3'd3;
    localparam S_ISSUE    = 3'd4;
    localparam S_STEP_END = 3'd5;

    //------------------------------------------------------------------------
    // registers
    //------------------------------------------------------------------------
    reg [2:0]  state_r,   state_w;
    reg [2:0]  step_r,    step_w;
    reg        og_r,      og_w;
    reg [4:0]  g_r,       g_w;                 // pixel group
    reg        pass_sc_r, pass_sc_w;           // current pass is the shortcut
    reg        bank_r,    bank_w;              // bank of the current plain pass
    reg        bank_t_r,  bank_t_w;            // ping-pong toggle for plain layers
    reg [5:0]  i_r,       i_w;
    reg [1:0]  ky_r,      ky_w;
    reg [1:0]  kx_r,      kx_w;
    reg [10:0] waddr_r,   waddr_w;
    reg [2:0]  pj_r,      pj_w;                // parameter word counter
    reg [3:0]  slot_r,    slot_w;
    reg        stem_done_r, stem_done_w;
    reg        done_r,      done_w;
    reg        fc_start_r,  fc_start_w;
    reg        pool_clr_r,  pool_clr_w;
    reg        par_we_r;                       // PBUF data arrives one cycle after the read
    reg        par_set_r;
    reg [1:0]  par_word_r;

    //------------------------------------------------------------------------
    // step table
    //------------------------------------------------------------------------
    reg [2:0] st_main, st_sc;
    reg       st_res, st_pool, st_nog_m1;
    reg [4:0] st_ng_m1;

    always @(*) begin
        st_sc     = 3'd0;
        st_res    = 1'b0;
        st_pool   = 1'b0;
        st_nog_m1 = 1'b0;
        case (step_r)
            3'd0:    begin st_main = 3'd0; st_ng_m1 = 5'd31; end
            3'd1:    begin st_main = 3'd1; st_ng_m1 = 5'd7;  end
            3'd2:    begin st_main = 3'd3; st_sc = 3'd2; st_res = 1'b1; st_ng_m1 = 5'd7; end
            3'd3:    begin st_main = 3'd4; st_nog_m1 = 1'b1; st_ng_m1 = 5'd1; end
            default: begin st_main = 3'd6; st_sc = 3'd5; st_res = 1'b1; st_pool = 1'b1;
                           st_nog_m1 = 1'b1; st_ng_m1 = 5'd1; end
        endcase
    end

    //------------------------------------------------------------------------
    // layer table (for the layer of the current pass)
    //------------------------------------------------------------------------
    wire [2:0] lay = pass_sc_r ? st_sc : st_main;

    reg [5:0]  ly_cin_m1;                      // input channels - 1
    reg        ly_k3;                          // 3x3 kernel (else 1x1)
    reg [1:0]  ly_src;
    reg [3:0]  ly_pblk;                        // parameter block of og 0

    always @(*) begin
        case (lay)
            3'd0:    begin ly_cin_m1 = 6'd2;  ly_k3 = 1'b1; ly_src = 2'd0; ly_pblk = 4'd0; end
            3'd1:    begin ly_cin_m1 = 6'd15; ly_k3 = 1'b1; ly_src = 2'd1; ly_pblk = 4'd1; end
            3'd2:    begin ly_cin_m1 = 6'd15; ly_k3 = 1'b0; ly_src = 2'd1; ly_pblk = 4'd2; end
            3'd3:    begin ly_cin_m1 = 6'd15; ly_k3 = 1'b1; ly_src = 2'd2; ly_pblk = 4'd3; end
            3'd4:    begin ly_cin_m1 = 6'd15; ly_k3 = 1'b1; ly_src = 2'd3; ly_pblk = 4'd4; end
            3'd5:    begin ly_cin_m1 = 6'd15; ly_k3 = 1'b0; ly_src = 2'd3; ly_pblk = 4'd6; end
            default: begin ly_cin_m1 = 6'd31; ly_k3 = 1'b1; ly_src = 2'd2; ly_pblk = 4'd8; end
        endcase
    end

    //------------------------------------------------------------------------
    // input address generation (S0), SPEC 6.11
    //   column p computes output pixel n = 8g + p
    //   (oy0, ox0) = output position of column 0
    //   (dy_p, dx_p) = input offset of column p relative to column 0
    //------------------------------------------------------------------------
    reg  [4:0] oy0, ox0;
    reg        ly_s2, ly_p1, ly_two_row;
    reg  [5:0] ly_hw;                          // input height = width

    always @(*) begin
        ly_two_row = 1'b0;
        case (lay)
            3'd0:    begin oy0 = {1'b0, g_r[4:1]}; ox0 = {1'b0, g_r[0], 3'b000}; ly_s2 = 1'b1; ly_p1 = 1'b1; ly_hw = 6'd32; end
            3'd1:    begin oy0 = g_r;              ox0 = 5'd0; ly_s2 = 1'b1; ly_p1 = 1'b1; ly_hw = 6'd16; end
            3'd2:    begin oy0 = g_r;              ox0 = 5'd0; ly_s2 = 1'b1; ly_p1 = 1'b0; ly_hw = 6'd16; end
            3'd3:    begin oy0 = g_r;              ox0 = 5'd0; ly_s2 = 1'b0; ly_p1 = 1'b1; ly_hw = 6'd8;  end
            3'd4:    begin oy0 = {g_r[3:0], 1'b0}; ox0 = 5'd0; ly_s2 = 1'b1; ly_p1 = 1'b1; ly_hw = 6'd8;  ly_two_row = 1'b1; end
            3'd5:    begin oy0 = {g_r[3:0], 1'b0}; ox0 = 5'd0; ly_s2 = 1'b1; ly_p1 = 1'b0; ly_hw = 6'd8;  ly_two_row = 1'b1; end
            default: begin oy0 = {g_r[3:0], 1'b0}; ox0 = 5'd0; ly_s2 = 1'b0; ly_p1 = 1'b1; ly_hw = 6'd4;  ly_two_row = 1'b1; end
        endcase
    end

    // input position of column 0 (signed: -1 is padding)
    wire signed [7:0] iy0 = $signed({2'b00, (ly_s2 ? {oy0, 1'b0} : {1'b0, oy0})}) + $signed({6'd0, ky_r}) - $signed({7'd0, ly_p1});
    wire signed [7:0] ix0 = $signed({2'b00, (ly_s2 ? {ox0, 1'b0} : {1'b0, ox0})}) + $signed({6'd0, kx_r}) - $signed({7'd0, ly_p1});

    // row used to pick the FM word: iy0 clamped to [0, H-1]
    wire [5:0] iy0c = (iy0 < 0) ? 6'd0 : (iy0 >= $signed({2'b00, ly_hw})) ? (ly_hw - 6'd1) : iy0[5:0];

    reg [5:0] fm_word;
    always @(*) begin
        case (lay)
            3'd0:              fm_word = {i_r[1:0], 4'b0000} + {1'b0, iy0c[5:1]};   // i*16 + iy/2 (2 rows per word)
            3'd1, 3'd2:        fm_word = {i_r[3:0], 2'b00}   + {2'b00, iy0c[5:2]};  // i*4  + iy/4 (4 rows per word)
            3'd3, 3'd4, 3'd5:  fm_word = i_r;                                       // one 8x8 channel per word
            default:           fm_word = {2'b00, i_r[5:2]};                         // four 4x4 channels per word
        endcase
    end

    // per-column offsets, valid flags and byte indices
    wire [47:0] sel;
    wire [7:0]  vld;

    genvar p;
    generate
        for (p = 0; p < 8; p = p + 1) begin : g_col
            // offset of column p from column 0, in output pixels
            wire              dy  = ly_two_row ? p[2] : 1'b0;
            wire [2:0]        dx  = ly_two_row ? {1'b0, p[1:0]} : p[2:0];
            // ... and in input pixels (stride 2 doubles it)
            wire signed [7:0] iy  = iy0 + (ly_s2 ? {6'd0, dy, 1'b0} : {7'd0, dy});
            wire signed [7:0] ix  = ix0 + (ly_s2 ? {4'd0, dx, 1'b0} : {5'd0, dx});
            reg         [5:0] byte_idx;

            always @(*) begin
                case (lay)
                    3'd0:             byte_idx = {iy[0],   5'b00000} + ix[5:0];               // (iy%2)*32 + ix
                    3'd1, 3'd2:       byte_idx = {iy[1:0], 4'b0000}  + ix[5:0];               // (iy%4)*16 + ix
                    3'd3, 3'd4, 3'd5: byte_idx = {iy[2:0], 3'b000}   + ix[5:0];               // iy*8 + ix
                    default:          byte_idx = {i_r[1:0], 4'b0000} + {iy[1:0], 2'b00} + ix[5:0]; // (i%4)*16 + iy*4 + ix
                endcase
            end

            assign vld[p]         = (iy >= 0) && (iy < $signed({2'b00, ly_hw})) &&
                                    (ix >= 0) && (ix < $signed({2'b00, ly_hw}));
            assign sel[6*p +: 6]  = byte_idx;
        end
    endgenerate

    //------------------------------------------------------------------------
    // drain descriptor for the main layer of the step (SPEC 6.14)
    //   output position of column 0 uses the same (oy0, ox0) as the main layer
    //------------------------------------------------------------------------
    reg [1:0] d_dst;
    reg [5:0] d_word, d_byte;
    reg       d_packed;
    reg [1:0] d_wstep;

    always @(*) begin
        d_packed = 1'b0;
        d_wstep  = 2'd0;
        d_word   = 6'd0;
        d_byte   = 6'd0;
        case (st_main)
            3'd0: begin d_dst = 2'd1; d_word = {4'b0000, g_r[4:3]};  d_byte = {g_r[2:1], g_r[0], 3'b000}; d_wstep = 2'd2; end // stem: 16x16, 4 rows/word, oy0 = g/2
            3'd1: begin d_dst = 2'd2; d_word = 6'd0;                d_byte = {g_r[2:0], 3'b000}; end                     // l1_a: 8x8 per word
            3'd3: begin d_dst = 2'd3; d_word = 6'd0;                d_byte = {g_r[2:0], 3'b000}; end                     // l1_b
            3'd4: begin d_dst = 2'd2; d_word = {3'b000, og_r, 2'b00}; d_byte = {2'b00, g_r[0], 3'b000}; d_packed = 1'b1; end // l2_a: 4x4 packed
            default: begin d_dst = 2'd0; end                                                                                 // l2_b: pool
        endcase
    end

    // weight bases of the step's layers (independent of the current pass)
    reg [10:0] main_waddr, sc_waddr;
    always @(*) begin
        case (step_r)
            3'd0:    begin main_waddr = 11'd0;                                  sc_waddr = 11'd0; end
            3'd1:    begin main_waddr = 11'd27;                                 sc_waddr = 11'd0; end
            3'd2:    begin main_waddr = 11'd187;                                sc_waddr = 11'd171; end
            3'd3:    begin main_waddr = og_r ? 11'd475 : 11'd331;               sc_waddr = 11'd0; end
            default: begin main_waddr = og_r ? 11'd939 : 11'd651;               sc_waddr = og_r ? 11'd635 : 11'd619; end
        endcase
    end

    //------------------------------------------------------------------------
    // weight addresses (SPEC 5.4 wbase; og 1 adds Cin*K*K)
    //------------------------------------------------------------------------
    // first weight address of the pass that starts next cycle
    reg [10:0] next_pass_waddr;
    always @(*) begin
        if (state_r == S_BATCH && st_res)
            next_pass_waddr = sc_waddr;                  // residual batch starts with the shortcut
        else if (state_r == S_BATCH)
            next_pass_waddr = main_waddr;
        else
            next_pass_waddr = main_waddr;                // shortcut -> main
    end

    //------------------------------------------------------------------------
    // control FSM
    //------------------------------------------------------------------------
    wire issuing   = (state_r == S_ISSUE);
    wire last_ikk  = (i_r == ly_cin_m1) && (!ly_k3 || (ky_r == 2'd2 && kx_r == 2'd2));
    wire quiet     = !i_pipe_busy && !i_drain_busy;      // nothing in flight
    wire [2:0] pj_n = st_res ? 3'd6 : 3'd3;

    always @(*) begin
        state_w     = state_r;
        step_w      = step_r;
        og_w        = og_r;
        g_w         = g_r;
        pass_sc_w   = pass_sc_r;
        bank_w      = bank_r;
        bank_t_w    = bank_t_r;
        i_w         = i_r;
        ky_w        = ky_r;
        kx_w        = kx_r;
        waddr_w     = waddr_r;
        pj_w        = pj_r;
        slot_w      = slot_r;
        stem_done_w = 1'b0;
        done_w      = 1'b0;
        fc_start_w  = 1'b0;
        pool_clr_w  = 1'b0;

        case (state_r)
            S_IDLE: begin
                if (i_start) begin
                    slot_w  = i_slot;
                    step_w  = 3'd0;
                    og_w    = 1'b0;
                    g_w     = 5'd0;
                    state_w = S_PAR_WAIT;
                end
            end

            // parameters change only when nothing is in flight
            S_PAR_WAIT: begin
                if (quiet) begin
                    pj_w    = 3'd0;
                    state_w = S_PAR;
                end
            end

            // read 3 (plain) or 6 (residual) PBUF words
            S_PAR: begin
                pj_w = pj_r + 3'd1;
                if (pj_r == pj_n) state_w = S_BATCH;
            end

            // start a batch
            S_BATCH: begin
                if (st_res) begin
                    if (quiet) begin                   // both banks are used: wait for the previous drain
                        pass_sc_w = 1'b1;
                        bank_w    = 1'b1;
                        state_w   = S_ISSUE;
                    end
                end else begin
                    pass_sc_w = 1'b0;
                    bank_w    = bank_t_r;
                    bank_t_w  = ~bank_t_r;
                    state_w   = S_ISSUE;
                end
                i_w  = 6'd0;
                ky_w = 2'd0;
                kx_w = 2'd0;
            end

            // issue one (i, ky, kx) per cycle
            S_ISSUE: begin
                waddr_w = waddr_r + 11'd1;
                if (ly_k3 && kx_r != 2'd2) begin
                    kx_w = kx_r + 2'd1;
                end else begin
                    kx_w = 2'd0;
                    if (ly_k3 && ky_r != 2'd2) ky_w = ky_r + 2'd1;
                    else begin ky_w = 2'd0; i_w = i_r + 6'd1; end
                end

                if (last_ikk) begin
                    i_w  = 6'd0;
                    ky_w = 2'd0;
                    kx_w = 2'd0;
                    if (st_res && pass_sc_r) begin       // shortcut done: main pass, no bubble
                        pass_sc_w = 1'b0;
                        bank_w    = 1'b0;
                    end else if (g_r != st_ng_m1) begin
                        g_w     = g_r + 5'd1;
                        state_w = S_BATCH;
                    end else begin
                        g_w = 5'd0;
                        if (og_r != st_nog_m1) begin
                            og_w    = 1'b1;
                            state_w = S_PAR_WAIT;
                        end else begin
                            state_w = S_STEP_END;
                        end
                    end
                end
            end

            // a step ends when all its outputs are written back
            S_STEP_END: begin
                if (quiet) begin
                    og_w = 1'b0;
                    if (step_r == 3'd0) stem_done_w = 1'b1;
                    if (step_r == 3'd4) begin
                        fc_start_w = 1'b1;
                        done_w     = 1'b1;
                        state_w    = S_IDLE;
                    end else if (step_r == 3'd3) begin
                        if (!i_fc_busy) begin           // the FC of the previous image still reads the pool
                            pool_clr_w = 1'b1;
                            step_w     = 3'd4;
                            state_w    = S_PAR_WAIT;
                        end
                    end else begin
                        step_w  = step_r + 3'd1;
                        state_w = S_PAR_WAIT;
                    end
                end
            end

            default: state_w = S_IDLE;
        endcase

        // the weight address restarts at every new pass
        if ((state_r == S_BATCH && state_w == S_ISSUE) ||
            (state_r == S_ISSUE && last_ikk && st_res && pass_sc_r)) begin
            waddr_w = next_pass_waddr;
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r     <= S_IDLE;
            step_r      <= 3'd0;
            og_r        <= 1'b0;
            g_r         <= 5'd0;
            pass_sc_r   <= 1'b0;
            bank_r      <= 1'b0;
            bank_t_r    <= 1'b0;
            i_r         <= 6'd0;
            ky_r        <= 2'd0;
            kx_r        <= 2'd0;
            waddr_r     <= 11'd0;
            pj_r        <= 3'd0;
            slot_r      <= 4'd0;
            stem_done_r <= 1'b0;
            done_r      <= 1'b0;
            fc_start_r  <= 1'b0;
            pool_clr_r  <= 1'b0;
            par_we_r    <= 1'b0;
            par_set_r   <= 1'b0;
            par_word_r  <= 2'd0;
        end else begin
            state_r     <= state_w;
            step_r      <= step_w;
            og_r        <= og_w;
            g_r         <= g_w;
            pass_sc_r   <= pass_sc_w;
            bank_r      <= bank_w;
            bank_t_r    <= bank_t_w;
            i_r         <= i_w;
            ky_r        <= ky_w;
            kx_r        <= kx_w;
            waddr_r     <= waddr_w;
            pj_r        <= pj_w;
            slot_r      <= slot_w;
            stem_done_r <= stem_done_w;
            done_r      <= done_w;
            fc_start_r  <= fc_start_w;
            pool_clr_r  <= pool_clr_w;
            par_we_r    <= pbuf_cs;
            par_set_r   <= (pj_r >= 3'd3);
            par_word_r  <= (pj_r >= 3'd3) ? (pj_r[1:0] + 2'd1) : pj_r[1:0];   // 3,4,5 -> 0,1,2
        end
    end

    //------------------------------------------------------------------------
    // parameter reads: words 0..2 of the main block (set 0), then of the shortcut block (set 1)
    //------------------------------------------------------------------------
    wire       pbuf_cs  = (state_r == S_PAR) && (pj_r < pj_n);
    wire [3:0] main_blk = (st_main == 3'd0) ? 4'd0 : (st_main == 3'd1) ? 4'd1 : (st_main == 3'd3) ? 4'd3 :
                          (st_main == 3'd4) ? (4'd4 + {3'd0, og_r}) : (4'd8 + {3'd0, og_r});
    wire [3:0] sc_blk   = (st_sc == 3'd2) ? 4'd2 : (4'd6 + {3'd0, og_r});
    wire [3:0] blk      = (pj_r < 3'd3) ? main_blk : sc_blk;
    wire [1:0] bword    = (pj_r < 3'd3) ? pj_r[1:0] : (pj_r[1:0] + 2'd1);
    wire [5:0] blk_x3   = {blk, 1'b0} + {2'b00, blk};                        // blk * 3

    //------------------------------------------------------------------------
    // outputs
    //------------------------------------------------------------------------
    assign o_stem_done  = stem_done_r;
    assign o_done       = done_r;
    assign o_fc_start   = fc_start_r;
    assign o_fc_slot    = slot_r;

    assign o_wbuf_cs    = issuing;
    assign o_wbuf_addr  = waddr_r;
    assign o_img_cs     = issuing && (kx_r == 2'd0) && (ly_src == 2'd0);
    assign o_img_addr   = fm_word;
    assign o_fm_rd      = issuing && (kx_r == 2'd0) && (ly_src != 2'd0);
    assign o_fm_rd_sel  = ly_src;
    assign o_fm_rd_addr = fm_word;
    assign o_pbuf_cs    = pbuf_cs;
    assign o_pbuf_addr  = blk_x3[4:0] + {3'b000, bword};

    assign o_tk_valid   = issuing;
    assign o_tk_first   = (i_r == 6'd0) && (ky_r == 2'd0) && (kx_r == 2'd0);
    assign o_tk_last    = last_ikk;
    assign o_tk_final   = !pass_sc_r;
    assign o_tk_bank    = bank_r;
    assign o_tk_bset    = pass_sc_r;
    assign o_tk_win_ld  = (kx_r == 2'd0);
    assign o_tk_src     = ly_src;
    assign o_tk_stem    = (lay == 3'd0);
    assign o_tk_sel     = sel;
    assign o_tk_vld     = vld;
    assign o_tk_dmode   = st_pool ? 2'd2 : st_res ? 2'd1 : 2'd0;
    assign o_tk_ddst    = d_dst;
    assign o_tk_dword   = d_word;
    assign o_tk_dbyte   = d_byte;
    assign o_tk_dpacked = d_packed;
    assign o_tk_dwstep  = d_wstep;
    assign o_tk_dog     = og_r;
    assign o_tk_ds24    = (i_shift_vec[8*st_main +: 8] == 8'd24);

    assign o_par_we     = par_we_r;
    assign o_par_set    = par_set_r;
    assign o_par_word   = par_word_r;
    assign o_pool_clr   = pool_clr_r;
endmodule
`default_nettype wire
