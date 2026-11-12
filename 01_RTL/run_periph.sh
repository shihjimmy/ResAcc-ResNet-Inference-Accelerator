#!/bin/sh
# 周邊電路測試：真的周邊 RTL + engine_stub。
#   輸出資料夾：環境變數 BUILD（預設 ./build）
#   用法：./run_periph.sh [張數] [READ_LATENCY] [RAND_GAP%]
cd "$(dirname "$0")"
N=${1:-64}; RL=${2:-12}; GAP=${3:-0}
B=${BUILD:-build}; mkdir -p $B && rm -f $B/img_*.hex 2>/dev/null
iverilog -g2012 -DDUMP_DIR=\"$B\" -o $B/periph.vvp \
    -DTOTAL_IMAGE_NUMBER=$N -DREAD_LATENCY=$RL -DRAND_GAP=$GAP -DCHECK_MODE=1 -DDUMP_BUFFERS \
    -f rtl.f ../00_TESTBED/stub/engine_stub.v \
    ../10_AXI_DRAM/axi_dram.sv ../00_TESTBED/host_model.sv ../00_TESTBED/TESTBED.sv || exit 1
vvp -n $B/periph.vvp | grep -v "^VCD info" || exit 1
python3 ../00_TESTBED/periph/check_buffers.py $B
