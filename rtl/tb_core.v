// Icarus testbench: stream @NTR@ traces of @NT@ samples through ccnn_core, compare S1/S2.
`timescale 1ns/1ps
module tb_core;
    localparam NTR = @NTR@, NT = @NT@;
    reg clk = 0, rst = 1, clear = 0, in_valid = 0;
    reg signed [7:0] in_x = 0;
    reg [4:0] rd_ch = 0;
    wire ready;
    wire [47:0] s1, s2;
    wire [31:0] n;
    ccnn_core dut(.clk(clk), .rst(rst), .clear(clear), .in_valid(in_valid), .in_x(in_x),
                  .ready(ready), .rd_ch(rd_ch), .s1_out(s1), .s2_out(s2), .n_samples(n));
    always #5 clk = ~clk;

    reg [7:0]  tv  [0:NTR*NT-1];
    reg [47:0] exp [0:NTR*64-1];
    integer tr, t, c, errors, cyc0, cyc, cycles;
    always @(posedge clk) cyc <= cyc + 1;
    initial begin
        $readmemh("tv_in.hex", tv);
        $readmemh("tv_exp.hex", exp);
        errors = 0; cyc = 0; cycles = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        for (tr = 0; tr < NTR; tr = tr + 1) begin
            @(posedge clk); clear <= 1;
            @(posedge clk); clear <= 0;
            for (t = 0; t < NT; t = t + 1) begin
                @(posedge clk); while (!ready) @(posedge clk);
                if (t == 1) cyc0 = cyc;
                in_x <= tv[tr*NT + t]; in_valid <= 1;
                @(posedge clk); in_valid <= 0;
            end
            @(posedge clk); while (!ready) @(posedge clk);
            cycles = (cyc - cyc0) / (NT - 1);
            for (c = 0; c < 32; c = c + 1) begin
                rd_ch = c; #1;
                if (s1 !== exp[tr*64 + c] || s2 !== exp[tr*64 + 32 + c]) begin
                    if (errors < 10) $display("trace %0d ch %0d: S1 %0d (exp %0d)  S2 %0d (exp %0d)",
                                              tr, c, s1, exp[tr*64+c], s2, exp[tr*64+32+c]);
                    errors = errors + 1;
                end
            end
            $display("trace %0d done, n=%0d, %0d cycles/sample", tr, n, cycles);
        end
        if (errors == 0) $display("PASS: %0d traces x %0d samples bit-exact", NTR, NT);
        else             $display("FAIL: %0d mismatches", errors);
        $finish;
    end
endmodule
