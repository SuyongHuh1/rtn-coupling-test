// Gate-level (post-fit, functional) testbench for power estimation.
// Drives the 50 MHz clock and KEY0 for the de10_selftest netlist (ccnn_de10.vo); the
// on-chip self-test then streams its stored traces exactly as on the board.
`timescale 1ps/1ps
module tb_gate;
    reg        clk = 1'b0;
    reg  [1:0] KEY = 2'b10;            // KEY0 pressed (reset) at start
    wire [7:0] LED;
    de10_selftest dut(.FPGA_CLK1_50(clk), .KEY(KEY), .LED(LED));
    always #10000 clk = ~clk;          // 50 MHz
    initial begin
        #200000 KEY[0] = 1'b1;         // release reset after 10 cycles
    end
endmodule
