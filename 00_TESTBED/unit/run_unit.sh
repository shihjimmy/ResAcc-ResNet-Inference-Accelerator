#!/bin/sh
# 執行所有基礎元件的單元測試。需要 iverilog / vvp 在 PATH 上。
cd "$(dirname "$0")"
RTL=../../01_RTL
mkdir -p build
set -e
iverilog -g2012 -o build/sram.vvp tb_sram_sp.sv $RTL/sram_sp.v && vvp -n build/sram.vvp | grep -v '^VCD'
for cfg in "0.834 2.0 100 100" "0.834 2.0 100 30" "0.834 2.0 20 100" "2.0 0.834 100 100" "2.0 0.834 50 50" "1.3 2.0 70 70"; do
    set -- $cfg
    iverilog -g2012 -P tb_async_fifo.WPER=$1 -P tb_async_fifo.RPER=$2 -P tb_async_fifo.WPROB=$3 -P tb_async_fifo.RPROB=$4 \
        -o build/fifo.vvp tb_async_fifo.sv $RTL/async_fifo.v $RTL/sync_2ff.v $RTL/reset_sync.v
    vvp -n build/fifo.vvp
done
