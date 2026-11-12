`timescale 1ns/1ps
//============================================================================
// fc_unit -- fully connected layer, argmax and confidence, SPEC 6.16
//
//   for k in 0..9:  read FCMEM[k] (32 weights of class k)
//                   acc = bfc[k]; for j in 0..31: acc += pool[j] * w[k][j]   (one 13x8 multiplier)
//                   if (k == 0 || acc > max) { max = acc; cat = k }       (ties keep the smaller k)
//   confidence = (max * conf_m + 2^(conf_s-1)) >>> conf_s, low 28 bits
//     * the product is built by shift-and-add, one bit of conf_m per cycle
//     * the final shift is one bit per cycle
//     so no wide multiplier or barrel shifter is needed (about 30 extra cycles)
//   result = {cat[3:0], confidence[27:0]}
//
// The pool sums are read directly from engine_dp; engine_ctrl does not clear
// them while this unit is busy. Takes about 380 cycles, overlapped with the
// next image.
//============================================================================
`default_nettype none
module fc_unit (
    input  wire          i_clk,
    input  wire          i_rst_n,
    input  wire          i_start,
    input  wire [3:0]    i_slot,               // result register to write
    input  wire [383:0]  i_pool_vec,           // 32 x 12-bit unsigned
    output wire          o_busy,
    output wire          o_fcm_cs,
    output wire [3:0]    o_fcm_addr,
    input  wire [255:0]  i_fcm_rdata,
    input  wire [319:0]  i_bfc_vec,
    input  wire [15:0]   i_conf_m,
    input  wire [7:0]    i_conf_s,
    output wire          o_res_we,
    output wire [3:0]    o_res_idx,
    output wire [31:0]   o_res_data
);
    localparam S_IDLE  = 3'd0;
    localparam S_READ  = 3'd1;   // read FCMEM[k]
    localparam S_LOAD  = 3'd2;   // weights arrive, acc = bias
    localparam S_MAC   = 3'd3;   // 32 multiply-accumulates
    localparam S_CMP   = 3'd4;   // argmax update
    localparam S_MUL   = 3'd5;   // max * conf_m, one bit per cycle
    localparam S_SHIFT = 3'd6;   // + rounding constant, then >>> 1 per cycle
    localparam S_OUT   = 3'd7;

    //------------------------------------------------------------------------
    // registers
    //------------------------------------------------------------------------
    reg [2:0]         state_r, state_w;
    reg [3:0]         k_r,     k_w;
    reg [4:0]         j_r,     j_w;          // MAC index, also bit / shift counter
    reg [3:0]         slot_r,  slot_w;
    reg               res_we_r, res_we_w;
    // data registers
    reg [3:0]         cat_r,   cat_w;
    reg [255:0]       wrow_r,  wrow_w;
    reg signed [31:0] acc_r,   acc_w;
    reg signed [31:0] max_r,   max_w;
    reg signed [47:0] prod_r,  prod_w;       // max * conf_m, then shifted
    reg signed [47:0] mcand_r, mcand_w;      // max << bit

    //------------------------------------------------------------------------
    // datapath helpers
    //------------------------------------------------------------------------
    wire        [11:0] pool_j = i_pool_vec[12*j_r +: 12];
    wire signed [7:0]  w_j    = wrow_r[8*j_r +: 8];
    wire signed [20:0] mac    = $signed({1'b0, pool_j}) * w_j;          // 13 x 8
    wire signed [31:0] bias_k = i_bfc_vec[32*k_r +: 32];
    wire signed [47:0] round  = 48'sd1 << (i_conf_s - 8'd1);            // single-bit decode

    //------------------------------------------------------------------------
    // next state
    //------------------------------------------------------------------------
    always @(*) begin
        state_w  = state_r;
        k_w      = k_r;
        j_w      = j_r;
        slot_w   = slot_r;
        res_we_w = 1'b0;
        cat_w    = cat_r;
        wrow_w   = wrow_r;
        acc_w    = acc_r;
        max_w    = max_r;
        prod_w   = prod_r;
        mcand_w  = mcand_r;

        case (state_r)
            S_IDLE: begin
                if (i_start) begin
                    slot_w  = i_slot;
                    k_w     = 4'd0;
                    state_w = S_READ;
                end
            end
            S_READ: begin
                state_w = S_LOAD;                          // FCMEM read latency is 1 cycle
            end
            S_LOAD: begin
                wrow_w  = i_fcm_rdata;
                acc_w   = bias_k;
                j_w     = 5'd0;
                state_w = S_MAC;
            end
            S_MAC: begin
                acc_w = acc_r + mac;
                j_w   = j_r + 5'd1;
                if (j_r == 5'd31) state_w = S_CMP;
            end
            S_CMP: begin
                if (k_r == 4'd0 || acc_r > max_r) begin
                    max_w = acc_r;
                    cat_w = k_r;
                end
                if (k_r == 4'd9) begin
                    // start the multiply with the final max
                    prod_w  = 48'sd0;
                    mcand_w = (k_r == 4'd0 || acc_r > max_r) ? {{16{acc_r[31]}}, acc_r} : {{16{max_r[31]}}, max_r};
                    j_w     = 5'd0;
                    state_w = S_MUL;
                end else begin
                    k_w     = k_r + 4'd1;
                    state_w = S_READ;
                end
            end
            S_MUL: begin
                if (i_conf_m[j_r[3:0]]) prod_w = prod_r + mcand_r;
                mcand_w = mcand_r <<< 1;
                j_w     = j_r + 5'd1;
                if (j_r == 5'd15) begin
                    j_w     = 5'd0;
                    state_w = S_SHIFT;
                end
            end
            S_SHIFT: begin
                // first cycle adds the rounding constant, then one arithmetic shift per cycle
                if (j_r == 5'd0) prod_w = prod_r + round;
                else             prod_w = prod_r >>> 1;
                j_w = j_r + 5'd1;
                if (j_r == i_conf_s[4:0]) state_w = S_OUT;
            end
            S_OUT: begin
                res_we_w = 1'b1;
                state_w  = S_IDLE;
            end
            default: state_w = S_IDLE;
        endcase
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state_r  <= S_IDLE;
            k_r      <= 4'd0;
            j_r      <= 5'd0;
            slot_r   <= 4'd0;
            res_we_r <= 1'b0;
        end else begin
            state_r  <= state_w;
            k_r      <= k_w;
            j_r      <= j_w;
            slot_r   <= slot_w;
            res_we_r <= res_we_w;
        end
    end

    always @(posedge i_clk) begin
        cat_r   <= cat_w;
        wrow_r  <= wrow_w;
        acc_r   <= acc_w;
        max_r   <= max_w;
        prod_r  <= prod_w;
        mcand_r <= mcand_w;
    end

    //------------------------------------------------------------------------
    // outputs
    //------------------------------------------------------------------------
    assign o_busy     = (state_r != S_IDLE) || res_we_r;
    assign o_fcm_cs   = (state_r == S_READ);
    assign o_fcm_addr = k_r;
    assign o_res_we   = res_we_r;
    assign o_res_idx  = slot_r;
    assign o_res_data = {cat_r, prod_r[27:0]};
endmodule
`default_nettype wire
