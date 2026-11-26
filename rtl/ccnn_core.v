// Streaming int8 causal CNN for coupled-trap RTN detection (Verilog-2001; Quartus / Icarus).
//
// One ADC sample per `in_valid` (when `ready`).  Six layers, 32 channels:
//   L0  7-tap input conv (taps packed as one 7-wide vector, K = 1)
//   L1..L4  K = 5 dilated (1, 2, 4, 8) residual convs, causal line buffers in one RAM
//   L5  1x1 conv
// Engine: for each output channel o and tap k, one cycle reads a 32-wide input slot and a
// 32-wide weight word and forms a 32-term int8 dot product (32 multipliers, pipelined
// adder tree).  Per-channel accumulation, ReLU, integer requantisation (m * acc >> SHIFT,
// round half up), residual add with requantised skip, saturation to [0, 127].
// After L5 the 32 outputs update running sums S1 = sum h, S2 = sum h^2 (48-bit).
// Bit-exact with rtnsim/quant.py::run_int.  ~790 clock cycles per sample.
`include "ccnn_params.vh"

module ccnn_core (
    input  wire               clk,
    input  wire               rst,
    input  wire               clear,        // start a new record (zero buffers and sums)
    input  wire               in_valid,
    input  wire signed [7:0]  in_x,
    output wire               ready,
    input  wire [4:0]         rd_ch,
    output wire [47:0]        s1_out,
    output wire [47:0]        s2_out,
    output reg  [31:0]        n_samples
);
    // ------------------------------------------------------------------ memories
    reg [255:0] wrom  [0:703];      // weight words (l, o, k): byte c = W[k][c][o]
    reg [31:0]  bias  [0:255];      // index l*32 + o
    reg [31:0]  mult  [0:255];
    reg [31:0]  mskip [0:255];
    initial begin
        $readmemh("ccnn_w.hex",  wrom);
        $readmemh("ccnn_b.hex",  bias);
        $readmemh("ccnn_m.hex",  mult);
        $readmemh("ccnn_ms.hex", mskip);
    end
    reg [255:0] abuf [0:63];        // line buffers of L1..L4 (bases 0, 5, 14, 31)
    reg [255:0] abuf_q, wrom_q;
    reg [5:0]   abuf_raddr, abuf_waddr;
    reg [9:0]   wrom_raddr;
    reg         abuf_we;
    reg [255:0] abuf_wdata;
    always @(posedge clk) begin
        if (abuf_we) abuf[abuf_waddr] <= abuf_wdata;
        abuf_q <= abuf[abuf_raddr];
        wrom_q <= wrom[wrom_raddr];
    end

    // ------------------------------------------------------------------ layer table
    function [2:0] f_K;     input [2:0] l; f_K     = (l >= 1 && l <= 4) ? 3'd5 : 3'd1; endfunction
    function [3:0] f_DIL;   input [2:0] l;
        case (l) 3'd2: f_DIL = 4'd2; 3'd3: f_DIL = 4'd4; 3'd4: f_DIL = 4'd8; default: f_DIL = 4'd1; endcase
    endfunction
    function [5:0] f_DEPTH; input [2:0] l;
        case (l) 3'd1: f_DEPTH = 6'd5; 3'd2: f_DEPTH = 6'd9; 3'd3: f_DEPTH = 6'd17; 3'd4: f_DEPTH = 6'd33; default: f_DEPTH = 6'd1; endcase
    endfunction
    function [5:0] f_BUFBASE; input [2:0] l;
        case (l) 3'd2: f_BUFBASE = 6'd5; 3'd3: f_BUFBASE = 6'd14; 3'd4: f_BUFBASE = 6'd31; default: f_BUFBASE = 6'd0; endcase
    endfunction
    function [9:0] f_ROMBASE; input [2:0] l;
        case (l) 3'd0: f_ROMBASE = 10'd0; 3'd1: f_ROMBASE = 10'd32; 3'd2: f_ROMBASE = 10'd192;
                 3'd3: f_ROMBASE = 10'd352; 3'd4: f_ROMBASE = 10'd512; default: f_ROMBASE = 10'd672; endcase
    endfunction
    function f_RES; input [2:0] l; f_RES = (l >= 1 && l <= 4); endfunction
    function f_BUF; input [2:0] l; f_BUF = (l >= 1 && l <= 4); endfunction

    // ------------------------------------------------------------------ state
    localparam S_IDLE = 3'd0, S_CLR = 3'd1, S_LSTART = 3'd2, S_ISSUE = 3'd3, S_DRAIN = 3'd4,
               S_LEND = 3'd5, S_ACC = 3'd6;
    reg [2:0]   state;
    reg [2:0]   l;
    reg [4:0]   o;
    reg [2:0]   k;
    reg [5:0]   cnt;
    reg [55:0]  tap;                // 7 most recent inputs, byte 6 = newest
    reg [255:0] cur_in, out_vec;
    reg [5:0]   head1, head2, head3, head4;
    reg [47:0]  s1 [0:31];
    reg [47:0]  s2 [0:31];
    assign s1_out = s1[rd_ch];
    assign s2_out = s2[rd_ch];
    assign ready  = (state == S_IDLE) && !clear;

    function [5:0] f_head; input [2:0] l; input [5:0] h1, h2, h3, h4;
        case (l) 3'd1: f_head = h1; 3'd2: f_head = h2; 3'd3: f_head = h3; default: f_head = h4; endcase
    endfunction

    // pipeline tags: valid, o, k==0, last
    reg        v1, vq, v2, v3;
    reg [4:0]  o1, oq, o2, o3;
    reg        f1, fq, f2, f3, z1, zq, z2, z3;
    reg        rv1, rv2;
    reg [4:0]  ro1, ro2;
    reg signed [15:0] prod [0:31];
    reg signed [31:0] dsum, acc;
    reg signed [63:0] pr_r, pr_s;
    reg               res1;

    // combinational read-address computation for the tap being issued
    wire [5:0] hcur  = f_head(l, head1, head2, head3, head4);
    wire [7:0] back  = (f_K(l) - 3'd1 - k) * f_DIL(l);
    wire [5:0] slot  = (hcur >= back[5:0]) ? (hcur - back[5:0]) : (hcur + f_DEPTH(l) - back[5:0]);

    integer i;
    reg signed [31:0] sum_t;
    reg signed [31:0] fin;
    reg signed [31:0] pos;
    reg signed [63:0] rq, sq;
    reg [31:0] rr, ss, vv;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; v1 <= 0; vq <= 0; v2 <= 0; v3 <= 0; rv1 <= 0; rv2 <= 0; abuf_we <= 0;
            n_samples <= 0; tap <= 0; head1 <= 0; head2 <= 0; head3 <= 0; head4 <= 0;
        end else begin
            abuf_we <= 1'b0;
            // ---------------- datapath pipeline (runs every cycle)
            // stage 1: memory read (abuf_q / wrom_q valid after this edge)
            vq <= v1; oq <= o1; fq <= f1; zq <= z1;
            // stage 2: products
            v2 <= vq; o2 <= oq; f2 <= fq; z2 <= zq;
            for (i = 0; i < 32; i = i + 1)
                prod[i] <= $signed((f_BUF(l) ? abuf_q[8*i +: 8] : cur_in[8*i +: 8])) * $signed(wrom_q[8*i +: 8]);
            // stage 3: adder tree
            v3 <= v2; o3 <= o2; f3 <= f2; z3 <= z2;
            sum_t = 0;
            for (i = 0; i < 32; i = i + 1) sum_t = sum_t + prod[i];
            dsum <= sum_t;
            // stage 4: accumulate; hand finished channel to requantisation
            rv1 <= 1'b0;
            if (v3) begin
                fin = (z3 ? $signed(bias[{l, o3}]) : acc) + dsum;
                acc <= fin;
                if (f3) begin
                    pos  = fin[31] ? 32'sd0 : fin;
                    pr_r <= $signed({1'b0, pos}) * $signed(mult[{l, o3}]);
                    pr_s <= $signed({24'd0, cur_in[8*o3 +: 8]}) * $signed(mskip[{l, o3}]);
                    res1 <= f_RES(l);
                    rv1 <= 1'b1; ro1 <= o3;
                end
            end
            // stage R2: round, clamp, residual, store
            rv2 <= rv1; ro2 <= ro1;
            if (rv1) begin
                rq = (pr_r + (64'sd1 <<< (`Q_SHIFT - 1))) >>> `Q_SHIFT;
                sq = (pr_s + (64'sd1 <<< (`Q_SHIFT - 1))) >>> `Q_SHIFT;
                rr = (rq < 0) ? 0 : ((rq > 127) ? 127 : rq[31:0]);
                ss = (sq < 0) ? 0 : ((sq > 127) ? 127 : sq[31:0]);
                vv = res1 ? ((rr + ss > 127) ? 127 : rr + ss) : rr;
                out_vec[8*ro1 +: 8] <= vv[7:0];
            end

            // ---------------- control
            v1 <= 1'b0;
            case (state)
            S_IDLE: begin
                if (clear) begin
                    state <= S_CLR; cnt <= 0;
                end else if (in_valid) begin
                    tap    <= {in_x, tap[55:8]};
                    cur_in <= {200'd0, in_x, tap[55:8]};
                    l <= 0; o <= 0; k <= 0;
                    state <= S_LSTART;
                end
            end
            S_CLR: begin
                abuf_we <= 1'b1; abuf_waddr <= cnt; abuf_wdata <= 256'd0;
                if (cnt < 32) begin s1[cnt[4:0]] <= 0; s2[cnt[4:0]] <= 0; end
                cnt <= cnt + 1'b1;
                if (cnt == 6'd63) begin
                    tap <= 0; head1 <= 0; head2 <= 0; head3 <= 0; head4 <= 0; n_samples <= 0;
                    state <= S_IDLE;
                end
            end
            S_LSTART: begin
                if (f_BUF(l)) begin
                    abuf_we <= 1'b1; abuf_waddr <= f_BUFBASE(l) + hcur; abuf_wdata <= cur_in;
                end
                o <= 0; k <= 0;
                state <= S_ISSUE;
            end
            S_ISSUE: begin
                abuf_raddr <= f_BUFBASE(l) + slot;
                wrom_raddr <= f_ROMBASE(l) + o * f_K(l) + k;
                v1 <= 1'b1; o1 <= o; z1 <= (k == 0); f1 <= (k == f_K(l) - 1);
                if (k == f_K(l) - 1) begin
                    k <= 0;
                    if (o == 5'd31) begin state <= S_DRAIN; cnt <= 0; end
                    else o <= o + 1'b1;
                end else k <= k + 1'b1;
            end
            S_DRAIN: begin
                cnt <= cnt + 1'b1;
                if (cnt == 6'd7) state <= S_LEND;
            end
            S_LEND: begin
                case (l)
                    3'd1: head1 <= (head1 == 6'd4)  ? 6'd0 : head1 + 1'b1;
                    3'd2: head2 <= (head2 == 6'd8)  ? 6'd0 : head2 + 1'b1;
                    3'd3: head3 <= (head3 == 6'd16) ? 6'd0 : head3 + 1'b1;
                    3'd4: head4 <= (head4 == 6'd32) ? 6'd0 : head4 + 1'b1;
                    default: ;
                endcase
                if (l == 3'd5) begin
                    state <= S_ACC; cnt <= 0;
                end else begin
                    cur_in <= out_vec; l <= l + 1'b1;
                    state <= S_LSTART;
                end
            end
            S_ACC: begin
                s1[cnt[4:0]] <= s1[cnt[4:0]] + out_vec[8*cnt[4:0] +: 8];
                s2[cnt[4:0]] <= s2[cnt[4:0]] + out_vec[8*cnt[4:0] +: 8] * out_vec[8*cnt[4:0] +: 8];
                cnt <= cnt + 1'b1;
                if (cnt == 6'd31) begin n_samples <= n_samples + 1; state <= S_IDLE; end
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
