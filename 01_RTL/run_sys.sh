#!/bin/sh
# 完整系統模擬：真的 engine。逐層比對 + host 比對 golden_results。
#   輸出資料夾：環境變數 BUILD（預設 ./build）
#   用法：./run_sys.sh [張數] [READ_LATENCY] [RAND_GAP%]
cd "$(dirname "$0")"
N=${1:-64}; RL=${2:-12}; GAP=${3:-0}
B=${BUILD:-build}; mkdir -p $B && rm -f $B/L*_img*.hex 2>/dev/null
iverilog -g2012 -DDUMP_DIR=\"$B\" -o $B/sys.vvp \
    -DTOTAL_IMAGE_NUMBER=$N -DREAD_LATENCY=$RL -DRAND_GAP=$GAP -DCHECK_MODE=0 -DDUMP_LAYERS \
    -f rtl.f -f engine.f ../10_AXI_DRAM/axi_dram.sv ../00_TESTBED/host_model.sv ../00_TESTBED/TESTBED.sv || exit 1
vvp -n $B/sys.vvp | grep -v "^VCD info" || exit 1
python3 ../00_TESTBED/layer/check_layers.py $B
