// DE10-Nano on-board self-test of ccnn_core (Cyclone V 5CSEBA6U23I7, 50 MHz clock).
// Streams @NTR@ stored traces x @NT@ samples through the CNN, compares the 32x2 running sums
// of each trace with the stored golden values (from the numpy integer reference).
//   KEY0 (press)  : reset and re-run
//   LED0          : heartbeat (~1.5 Hz) - bitstream alive
//   LED1          : running
//   LED2          : done
//   LED3          : PASS (all sums bit-exact)
//   LED4          : FAIL
//   LED7..LED5    : index of the trace being processed (7 at the end)
// Speed measurement registers (kept with noprune, read with Signal Tap after `done`):
//   sample_period / sample_period_max : system-clock cycles between consecutive input samples
//   run_cycles                        : system-clock cycles from reset release to done
//   ref_cycles                        : 50 MHz reference-clock cycles while running (wall time)
// Define USE_PLL (and add the pll_sys IP) to run the system clock from a PLL instead of 50 MHz.
module de10_selftest (
    input  wire       FPGA_CLK1_50,
    input  wire [1:0] KEY,
    output wire [7:0] LED
);
    localparam NTR = @NTR@, NT = @NT@;
    wire clk, locked;
`ifdef USE_PLL
    pll_sys u_pll(.refclk(FPGA_CLK1_50), .rst(1'b0), .outclk_0(clk), .locked(locked));
`else
    assign clk = FPGA_CLK1_50;
    assign locked = 1'b1;
`endif

    // reset synchroniser (KEY is active low; also held until the PLL locks)
    reg [2:0] rs = 3'b111;
    always @(posedge clk) rs <= {rs[1:0], ~KEY[0] | ~locked};
    wire rst = rs[2];

    // test-vector ROMs (M10K)
    reg [7:0]  tv  [0:NTR*NT-1];
    reg [47:0] exp [0:NTR*64-1];
    initial begin
        $readmemh("tv_in.hex", tv);
        $readmemh("tv_exp.hex", exp);
    end
    reg [15:0] tv_addr;
    reg [9:0]  exp_addr;
    reg [7:0]  tv_q;
    reg [47:0] exp_q;
    always @(posedge clk) begin
        tv_q  <= tv[tv_addr];
        exp_q <= exp[exp_addr];
    end

    reg clear, in_valid;
    reg signed [7:0] in_x;
    reg [4:0] rd_ch;
    wire ready;
    wire [47:0] s1, s2;
    wire [31:0] n;
    ccnn_core core(.clk(clk), .rst(rst), .clear(clear), .in_valid(in_valid), .in_x(in_x),
                   .ready(ready), .rd_ch(rd_ch), .s1_out(s1), .s2_out(s2), .n_samples(n));

    localparam T_START = 4'd0, T_CLR = 4'd1, T_WAIT = 4'd2, T_FEED = 4'd3, T_FIN = 4'd4,
               T_C0 = 4'd5, T_C1 = 4'd6, T_C2 = 4'd7, T_C3 = 4'd8, T_C4 = 4'd9, T_DONE = 4'd10;
    reg [3:0]  st;
    reg [2:0]  tr;
    reg [15:0] t;
    reg [5:0]  ch;
    reg [3:0]  wcnt;
    reg        fail, done;
    reg [31:0] cyc, cyc_last;
    (* noprune *) reg [31:0] sample_period, sample_period_max, run_cycles;

    always @(posedge clk) begin
        if (rst) begin
            st <= T_START; tr <= 0; fail <= 0; done <= 0; clear <= 0; in_valid <= 0;
            tv_addr <= 0; exp_addr <= 0; rd_ch <= 0; cyc <= 0;
            sample_period <= 0; sample_period_max <= 0; run_cycles <= 0;
        end else begin
            cyc <= cyc + 1;
            clear <= 1'b0;
            case (st)
            T_START: begin clear <= 1'b1; st <= T_CLR; t <= 0; wcnt <= 0; end
            T_CLR:   if (wcnt == 4'd3) st <= T_WAIT; else wcnt <= wcnt + 1'b1;
            T_WAIT:  if (ready) st <= T_FEED;            // clear sequence finished, tv_q valid
            T_FEED: begin
                if (in_valid) in_valid <= 1'b0;
                else if (ready) begin
                    if (t == NT) st <= T_FIN;
                    else begin
                        in_x <= tv_q; in_valid <= 1'b1;
                        tv_addr <= tv_addr + 1'b1; t <= t + 1'b1;
                        cyc_last <= cyc;
                        if (t != 0) begin
                            sample_period <= cyc - cyc_last;
                            if (cyc - cyc_last > sample_period_max) sample_period_max <= cyc - cyc_last;
                        end
                    end
                end
            end
            T_FIN:  begin ch <= 0; st <= T_C0; end
            T_C0:   begin rd_ch <= ch[4:0]; exp_addr <= tr * 64 + ch; st <= T_C1; end
            T_C1:   st <= T_C2;                            // exp_q valid next cycle
            T_C2:   begin if (s1 != exp_q) fail <= 1'b1; exp_addr <= exp_addr + 6'd32; st <= T_C3; end
            T_C3:   st <= T_C4;
            T_C4: begin
                if (s2 != exp_q) fail <= 1'b1;
                if (ch == 6'd31) begin
                    if (tr == NTR - 1) st <= T_DONE;
                    else begin tr <= tr + 1'b1; st <= T_START; end
                end else begin ch <= ch + 1'b1; st <= T_C0; end
            end
            T_DONE: begin if (!done) run_cycles <= cyc; done <= 1'b1; end
            default: st <= T_START;
            endcase
        end
    end

    // wall-clock measurement in the 50 MHz reference domain (independent of the system clock)
    reg [1:0] run_s, key_s;
    (* noprune *) reg [31:0] ref_cycles;
    always @(posedge FPGA_CLK1_50) begin
        run_s <= {run_s[0], ~rst & ~done};
        key_s <= {key_s[0], ~KEY[0]};
        if (key_s[1]) ref_cycles <= 0;
        else if (run_s[1]) ref_cycles <= ref_cycles + 1'b1;
    end

    reg [24:0] hb;
    always @(posedge FPGA_CLK1_50) hb <= hb + 1'b1;
    assign LED = {tr, done & fail, done & ~fail, done, ~done, hb[24]};
endmodule
