# Quartus Prime (Lite) project for the DE10-Nano CNN self-test.
# Usage (in this folder):  quartus_sh -t create_project.tcl
#                          quartus_sh --flow compile ccnn_de10
# or open Quartus -> Tools -> Tcl Scripts -> run this file, then Processing -> Start Compilation.
load_package flow
project_new ccnn_de10 -overwrite

set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEBA6U23I7
set_global_assignment -name TOP_LEVEL_ENTITY de10_selftest
set_global_assignment -name VERILOG_FILE de10_selftest.v
set_global_assignment -name VERILOG_FILE ccnn_core.v
set_global_assignment -name SEARCH_PATH .
set_global_assignment -name SDC_FILE ccnn_de10.sdc
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
# For a PLL system clock: generate IP 'pll_sys' (see README; Quartus offers to add pll_sys.qip), then uncomment:
# set_global_assignment -name VERILOG_MACRO "USE_PLL=1"
# set_global_assignment -name QIP_FILE pll_sys.qip
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"

# DE10-Nano pins (Terasic DE10-Nano user manual, 3.3-V LVTTL banks)
set_location_assignment PIN_V11  -to FPGA_CLK1_50
set_location_assignment PIN_AH17 -to KEY[0]
set_location_assignment PIN_AH16 -to KEY[1]
set_location_assignment PIN_W15  -to LED[0]
set_location_assignment PIN_AA24 -to LED[1]
set_location_assignment PIN_V16  -to LED[2]
set_location_assignment PIN_V15  -to LED[3]
set_location_assignment PIN_AF26 -to LED[4]
set_location_assignment PIN_AE26 -to LED[5]
set_location_assignment PIN_Y16  -to LED[6]
set_location_assignment PIN_AA23 -to LED[7]
foreach p {FPGA_CLK1_50 KEY[0] KEY[1] LED[0] LED[1] LED[2] LED[3] LED[4] LED[5] LED[6] LED[7]} {
    set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to $p
}

set f [open ccnn_de10.sdc w]
puts $f "create_clock -name clk50 -period 20.000 \[get_ports FPGA_CLK1_50\]"
puts $f "# JTAG clock used by Signal Tap / debug hub (asynchronous to the design)"
puts $f "create_clock -name altera_reserved_tck -period 33.333 \[get_ports altera_reserved_tck\]"
puts $f "set_clock_groups -asynchronous -group \[get_clocks altera_reserved_tck\]"
puts $f "derive_pll_clocks"
puts $f "derive_clock_uncertainty"
puts $f "set_false_path -from \[get_ports KEY*\]"
puts $f "set_false_path -to \[get_ports LED*\]"
puts $f "set_false_path -to \[get_registers *run_s\[0\]\]"
close $f

project_close
