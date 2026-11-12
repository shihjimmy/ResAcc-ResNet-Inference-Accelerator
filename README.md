# ResAcc: ResNet6 (CIFAR-10) Inference Accelerator

An end-to-end RTL accelerator for a small ResNet (ResNet6) on CIFAR-10, with a software
reference model, a bit-exact integer model, an architecture exploration model and a
full-system testbench. The design has three clock domains, an AXI4-Lite host interface and
an AXI4 DRAM interface. Simulation uses open-source tools only (Icarus Verilog, Yosys, Python).

## Highlights

| Item | Result |
|---|---|
| Network | ResNet6: 3x3 stem, two residual blocks (with 1x1 shortcut), global average pool, FC 32 -> 10; 651,584 MAC / image |
| Quantization | BN folded, int8 per-channel weights, uint8 activations, integer requantization |
| Dataflow | Output stationary, 16 output channels x 8 output pixels (128 PEs) |
| Accuracy check | 64 test images: all layer outputs and final results bit-exact against the integer golden model; 63/64 agree with the float model |
| Latency | 64 images in 739,541.5 ns (clk_dut 2.0 ns) |
| Logic size | 159,690 cells (Yosys generic cells, SRAM black-boxed) |

The full description (architecture comparison, block diagrams, verification, results) is in
[`docs/report.pdf`](docs/report.pdf). The bit-level specification is
[`docs/SPEC.md`](docs/SPEC.md) and the design decisions are in [`docs/ARCH.md`](docs/ARCH.md)
(both written in Traditional Chinese).

## Repository layout

| Path | Content |
|---|---|
| `docs/` | Report (PDF), specification, design decision log |
| `01_RTL/` | Design RTL (top: `resacc`), file lists, simulation scripts |
| `00_TESTBED/` | Testbed, behavioral AXI4-Lite host, unit tests, checkers, test patterns |
| `00_TESTBED/PATTERN/` | Test images, reference csv, per-layer golden data, `model.dat`, `golden_results.txt` |
| `10_AXI_DRAM/` | Behavioral AXI4 DRAM model |
| `02_SYN/` | Yosys area / logic-depth estimation scripts |
| `97_ARCH/` | Architecture exploration (dataflow comparison, cycle / traffic / energy model) |
| `98_ALGORITHM/` | Reference model (`resnet6.pth`, notebooks), integer golden model, `model.dat` generator |

## Requirements

- Icarus Verilog 11 or newer (`iverilog -g2012`, `vvp`)
- Python 3 with `numpy` (the golden model and checkers need nothing else)
- Optional: Yosys (or `pip install yowasp-yosys`) for area / depth estimates
- Optional: the conda environment `98_ALGORITHM/environment.yml` (PyTorch) only to rerun the notebooks

## Quick start

All generated test data is already in `00_TESTBED/PATTERN/`, so the simulations run directly.

```sh
# 1. unit tests (SRAM, async FIFO)
sh 00_TESTBED/unit/run_unit.sh

# 2. full-system simulation: layer-by-layer check against GOLDEN + host result check
cd 01_RTL
sh run_sys.sh            # 64 images, DRAM read latency 12
sh run_sys.sh 2          # quick run with 2 images
sh run_sys.sh 64 12 30   # arguments: images, READ_LATENCY, random DRAM gap %

# 3. peripherals only (loader / host interface / DRAM path, engine replaced by a stub)
sh run_periph.sh
```

Outputs go to `build/` (override with the `BUILD` environment variable). A passing run prints
the total time and PASS lines for every layer and for the host result check.

### Software model and test data

```sh
cd 98_ALGORITHM
python3 float_ref.py          # float ResNet6, checked against the reference csv
python3 quant.py              # integer model, writes per-layer golden hex + quant_params.npz
python3 gen_model_dat.py      # packs model.dat and golden_results.txt
python3 hw_model.py 64        # model that reads only model.dat + images; must match GOLDEN
python3 sweep.py              # accuracy versus bit width
```

### Architecture model and synthesis estimates

```sh
cd 97_ARCH && python3 compare.py        # dataflow A / B / C comparison
cd 02_SYN  && sh run_yosys.sh           # YOSYS=yowasp-yosys sh run_yosys.sh if only the pip package is installed
```

`run_yosys.sh` takes several minutes; the area table is printed first, followed by the
logic depth of the main modules.

## Notes

- The simulation scripts are plain `sh` and expect to be run from the repository layout above
  (relative paths between folders).
- `01_RTL/engine.f` is the engine file list, `01_RTL/rtl.f` lists everything else.
