# ResAcc architecture decisions

Numbers come from `97_ARCH/arch_model.py`.

## Round 0 - quantization (software)
- Weights: signed 8-bit, symmetric, per output channel (BN folded first)
- Activations: unsigned 8-bit, zero-point 0 (all post-ReLU)
- Requantize: `(acc * m + 2^(s-1)) >> s`, m 16-bit, s per layer; accumulators fit in 20 bits
- RTL keeps WEIGHT_BITS / ACT_BITS as parameters (6-bit weights possible later)

## Round 1 - system skeleton
- clk_dut = 500 MHz (2.0 ns), assumed (no synthesis available)
- Three clock domains: host AXI-Lite (0.77 GHz), DRAM AXI-Full (1.2 GHz), core (0.5 GHz)

## Round 2 - performance budget (in progress)
- 651,584 MAC / image, 41,701,376 MAC / 64 images
- Goal 1e6 ns @ 500 MHz -> 500,000 cycles -> >= 83.4 MAC/cycle at 100% utilization
- Target PE count ~128 (to be fixed after the dataflow round)

## Round 4 - dataflow: DECIDED = C (OS-pixel)
`97_ARCH/dataflow.py`, 128 multipliers in all cases, per image:

| dataflow | cycles | util | avg read/cycle | buffer traffic | accumulators | psum buffer |
|---|---|---|---|---|---|---|
| A OS-channel (16 out-ch x 8 in-ch, position outer) | 6,532 | 78% | 106 B | 685 KB | 16 | - |
| B WS-channel (16 out-ch x 8 in-ch, weights fixed) | 6,532 | 78% | 46 B | 535 KB | 16 | 4,096 x 20b |
| C OS-pixel (16 out-ch x 8 out-pixels) | 5,120 | 99% | 24 B | 128 KB | 128 | - |

Energy proxy per image (`97_ARCH/compare.py`, MAC=1, RF=1, buffer=6x; buffer=3x in brackets):
A 4.96M (2.86M), B 4.59M (2.95M), C 2.09M (1.70M) -> C is 1.7-2.4x lower, ranking stable.
Area differences: A 320b acc + 1024b weight port; B + 81,920b psum buffer; C 2,560b acc + 128b weight port.

## Round 3/5 - dataflow C detailed spec (draft, `97_ARCH/schedule_C.py`)

### PE array
- 16 rows (output channels of current og) x 8 columns (8 output pixels of current pixel group)
- Per cycle: 16 weights (int8, one per row, broadcast across row), 8 inputs (9-bit signed, one per column, broadcast down)
- Input operand 9-bit signed: stem input = pixel-128 (int8), other layers uint8 zero-extended
- Each PE: 8x9 signed multiplier, 20-bit accumulator x 2 banks; accumulator initialised with bias_q
- One batch = (og, pixel group); takes Cin*K*K cycles; loop inside batch: for i: for ky: for kx

### Pixel group (8 output pixels)
| layer | output | group |
|---|---|---|
| stem | 16x16 | half an output row |
| l1_a / l1_b / l1_sc | 8x8 | one output row |
| l2_a / l2_b / l2_sc | 4x4 | two output rows |

### On-chip memories
| name | bytes | word | content |
|---|---|---|---|
| WBUF | 19,632 | 128b (16 weights) | all conv weights, stored in exact consumption order |
| PARAM | ~680 | bias 18b + m 16b | 160 output channels; shift s per layer in regs |
| FCMEM | ~350 | - | fc weights/bias, conf_m/conf_s |
| IMG | 3,072 | 512b (2 rows) | input image; next image loaded after stem done |
| FM_A | 4,096 | 512b | stem out |
| FM_B | 1,024 | 512b (1 plane of 8x8) | l1_a out, later l2_a out |
| FM_C | 1,024 | 512b | l1 block out |
FM word = 64 bytes; every (i, ky) fetch of a pixel group hits exactly one word; window mux picks 8 bytes, zero for padding.

### Buffer rotation
| layer | reads | writes |
|---|---|---|
| stem | IMG | FM_A |
| l1_a | FM_A | FM_B |
| l1_sc then l1_b | FM_A, FM_B | FM_C |
| l2_a | FM_C | FM_B |
| l2_sc then l2_b | FM_C, FM_B | (pool sums, not stored) |

### Requantize / output
- 8 lanes; drain 128 outputs in 16 cycles; normal layers overlap drain with next batch (accumulator ping-pong)
- Residual: shortcut -> bank1, main -> bank0, then lane computes (acc_b*m_b + acc_sc*m_sc + rnd) >> s; array stalls 16 cycles
- Output write: 8 bytes (8 pixels of one channel) per cycle with byte enables
- l2 block outputs go straight into 32 pool sums (12-bit), never stored

### FC unit
- 1 MAC (12b x 8b), 320 cycles + argmax + confidence multiply; overlaps with next image's stem
- Results: 16 x 32-bit register file (category[31:28], confidence[27:0])

### Control / system
- Weights loaded once after reset (624 beats); per image: compute, prefetch next image after stem
- Estimated 5,353 cycles/image, 64 images ~ 687,000 ns @ 500 MHz (69% of goal), before host/batch overheads

## RTL notes
- Bug found in system test: in residual layers the next batch's shortcut pass started before the previous
  batch's drain had begun (drain starts 2 cycles after the last issue), overwriting bank1. Fix: wait for
  pipe_empty && !drain_busy before each residual batch.
- Measured: 714,738 ns for 64 images (5,563 cycles/image), bit-exact.

## Round 6 - RTL structure and area
- Style: two-process (`always @(*)` -> `_w`, sequential only `_r <= _w`), `i_`/`o_` ports, `S_` states,
  English comments, control regs reset / data regs not reset.
- Engine split into `engine_ctrl` (control path, produces one token per cycle = stage S0) and
  `engine_dp` (datapath; the token travels with the data through S1/S2 and launches the drain).
  Datapath never decodes "which layer am I in"; everything it needs is in the token / drain descriptor.
- Area changes:
  - requant: 1 multiplier per lane instead of 2; residual rows take 2 cycles (part = acc_sc*m_sc + rnd, then
    acc_b*m_b + part). s restricted to {23, 24} -> 2-way shift instead of a variable shifter.
    Rounding constant folded into the single stage-2 adder. Lane: 5,195 -> 2,578 cells.
  - fc_unit: reads pool directly (no 32x12 copy), confidence by 16-cycle shift-add instead of a 32x17
    multiplier, final shift 1 bit per cycle. 8,061 -> 4,392 cells.
  - PE single adder: almost no gain (Yosys already shared it): 105,073 -> 103,512 for the array.
- Measured: total 185,780 -> 159,690 cells (-14.0%), FF 15,165 -> 14,201.
  Time 714,738 -> 739,541.5 ns (+192 cycles/image, exactly the extra residual drain cycles). 64/64 bit-exact
  at READ_LATENCY 0 / 12 and 30% DRAM gaps.
- Depth (Yosys ltp, ripple adders, relative only): pe 44, engine_ctrl 45, requant_lane 77, engine_dp 77, fc_unit 109.
  If timing mattered, first candidates: split the requant multiply into 2 stages, and FC MAC.
