# Gate-level simulation for the Power Analyzer (ModelSim-Intel FPGA Starter, Quartus 20.1).
# Run inside <project>/simulation/modelsim after EDA Netlist Writer produced ccnn_de10.vo:
#   vsim -c -do ../../run_gate.do
vlib work
vlog -work work ccnn_de10.vo
vlog -work work ../../tb_gate.v
vsim -t 1ps -voptargs="+acc" -L altera_ver -L altera_lnsim_ver -L cyclonev_ver -L lpm_ver -L sgate_ver -L altera_mf_ver work.tb_gate
vcd file ccnn_de10.vcd
vcd add -r /tb_gate/dut/*
# 120 us = 6,000 clock cycles = about 7.5 samples of steady streaming (activity repeats every sample)
run 120 us
vcd flush
quit -f
