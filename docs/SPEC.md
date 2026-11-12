# ResAcc — ResNet6 加速器設計規格

| 項目 | 內容 |
|---|---|
| 狀態 | RTL 完成（control/datapath 分離、面積最佳化），64 張 bit-exact 通過（見 8.1、8.2） |
| 範圍 | 演算法（整數模型）、外部介面、`model.dat` 格式、硬體架構、驗證計畫 |
| 不在範圍內 | 合成、APR、功耗分析（沒有商用工具） |
| 決策紀錄 | `ARCH.md`（各輪討論的過程）；本文件以最後的決定為準 |
| 數字來源 | `98_ALGORITHM/quant.py`、`97_ARCH/*.py`（文件裡每個數字都可以由這些程式重算） |

---

## 目錄

1. 目標與限制
2. 系統概觀
3. 演算法規格：整數模型
4. 外部介面
5. DRAM 資料格式與 `model.dat`
6. 硬體架構
7. 位元寬度總表
8. 效能與面積估算
9. 驗證計畫
10. 檔案結構
11. 待確認事項與風險
附錄 A：位址計算的實例

---

## 1. 目標與限制

### 1.1 功能
輸入 CIFAR-10 的 32×32 彩色圖片，以 ResNet6 推論出 10 類中的一類，並輸出 confidence（最大 logit × 2^20）。
一批 1–16 張圖，由 host 透過 AXI4-Lite 下命令，圖片和模型存在外部 DRAM，由加速器透過 AXI4 讀取。

### 1.2 來自題目 spec 的限制
| 項目 | 限制 |
|---|---|
| top module | `resacc` |
| clk_dram | 1.2 GHz（0.834 ns），固定 |
| clk_host | 0.77 GHz（1.3 ns），固定 |
| clk_dut | 自訂（≥ 100 MHz） |
| rst_n | 非同步、低電位有效 |
| 錯誤率（64 張） | < 0.37 |
| 總時間（64 張） | Baseline < 5×10^6 ns，Goal < 10^6 ns |
| 2 張圖時間 | < 64 張時間 / 16 |
| 模型 | 不可重新訓練、不可修改；不可針對測試圖片最佳化 |

### 1.3 本設計的目標
| 項目 | 目標 |
|---|---|
| clk_dut | **500 MHz（2.0 ns）**，假設值 |
| 64 張總時間 | ≤ Goal 的 75%（估計約 69%） |
| 正確性 | RTL 與 `quant.py` 整數模型**每個 bit 都相同**（category 與 confidence） |

---

## 2. 系統概觀

```
 clk_host 0.77GHz          clk_dut 500MHz                                    clk_dram 1.2GHz
┌───────────────┐   ┌──────────────────────────────────────────────────┐   ┌────────────────┐
│ host_if       │   │ ctrl_fsm                                         │   │ dram_if        │
│ AXI4-Lite     │◄─►│                                                  │◄─►│ AXI4 master    │◄─► DRAM
│ slave + 暫存器 │CDC│ loader ─► WBUF PBUF FCMEM IMG FM_A FM_B FM_C       │CDC│ (只讀)          │
└───────────────┘   │            │                  │                  │   └────────────────┘
                    │            ▼                  ▼                  │
                    │       16 權重/cycle      in_sel（取 8 個輸入）     │
                    │            └──► PE 陣列 16×8 ◄──┘                 │
                    │                    │（2 組累加器）                  │
                    │                    ▼                             │
                    │        requant ×8 ─► 寫回 FM / pool 加總 ─► fc_unit ─► 結果暫存器 │
                    └──────────────────────────────────────────────────┘
```

| 方塊 | 時脈域 | 功能 |
|---|---|---|
| `host_if` | clk_host | AXI4-Lite slave、暫存器、start/finish 的 CDC |
| `dram_if` | clk_dram | AXI4 讀取（不寫入 DRAM） |
| `async_fifo` ×2 | 跨 clk_dut ↔ clk_dram | 讀取請求（dut→dram）、讀回資料（dram→dut） |
| `core` | clk_dut | 全部運算與 buffer |

**Dataflow**：Output Stationary，PE 陣列 = **16 個輸出通道 × 8 個輸出像素**（`ARCH.md` 第 4 輪的方案 C）。

---

## 3. 演算法規格：整數模型

**權威參考：`98_ALGORITHM/quant.py`（`int_forward()`）。RTL 必須與它 bit-exact。** 本章是它的文字版。

### 3.1 量化方式
| 項目 | 方式 |
|---|---|
| 實數與整數的關係 | `r = S × q`，zero-point 一律為 0 |
| BN | 先融合進 conv：`w' = w·γ/σ`、`b' = β − μ·γ/σ`，`σ = √(var + 1e-5)` |
| 權重 | int8 對稱，**每個輸出通道一個 scale**，範圍 [−127, 127] |
| activation | uint8，每個 tensor 一個 scale，範圍 [0, 255]（都在 ReLU 之後） |
| 輸入圖片 | `q_in = pixel − 128`（int8）；浮點的 `+0.5` 偏移已併進 stem 的 bias |
| calibration | 用第 64–191 張圖、第 99.99 百分位決定 activation scale；第 0–63 張只用來評估 |
| requantize | 浮點倍率 `M` 以 `m / 2^s` 近似，m 為 16-bit 無號整數，s 每層一個 |
| 捨入 | `round_shift(v, s) = (v + 2^(s−1)) >>> s`（算術右移，往 −∞） |

### 3.2 每一層的整數運算

符號：`conv(x, w)` = 整數 conv（zero padding 補 0）；`clamp8(v) = min(max(v, 0), 255)`。

```
x0     = pixel − 128                                              # 3×32×32，int8
stem   = clamp8( round_shift( (conv(x0, W_stem, s2, p1) + b_stem) × m_stem , s_stem ) )     # 16×16×16
# ---- residual block 1
l1_a   = clamp8( round_shift( (conv(stem, W_l1a, s2, p1) + b_l1a) × m_l1a , s_l1a ) )      # 16×8×8
acc_b  = conv(l1_a, W_l1b, s1, p1) + b_l1b
acc_sc = conv(stem, W_l1sc, 1×1, s2, p0) + b_l1sc
l1_out = clamp8( round_shift( acc_b × m_l1b + acc_sc × m_l1sc , s_l1 ) )                  # 16×8×8
# ---- residual block 2（同上，形狀不同）
l2_a   = clamp8( round_shift( (conv(l1_out, W_l2a, s2, p1) + b_l2a) × m_l2a , s_l2a ) )    # 32×4×4
l2_out = clamp8( round_shift( acc_b × m_l2b + acc_sc × m_l2sc , s_l2 ) )                  # 32×4×4
# ---- pool + fc
pool[c]   = Σ_{y,x} l2_out[c][y][x]                               # 32 個，12-bit 無號（不除以 16）
logit[k]  = Σ_j pool[j] × Wfc[k][j] + bfc[k]                      # 10 個，有號
category  = argmax(logit)                                          # 平手取 index 小的
confidence= round_shift( logit[category] × conf_m , conf_s )       # = 實數 logit × 2^20
```

`m`、`b` 都是**每個輸出通道一個**；`s` 每層一個（殘差的兩條路徑共用同一個 s）。

### 3.3 參數總表（8-bit 版本，由 `quant.py` 產生，存於 `quant_params.npz`）

| 層 | 權重形狀 | s | m 範圍 | \|b_q\| 最大 | 累加器需要（有號） |
|---|---|---|---|---|---|
| stem | 16×3×3×3 | 23 | 11,482 – 51,844 | 17,472 | 19 bit |
| l1_a | 16×16×3×3 | 23 | 23,548 – 64,493 | 30,709 | 18 bit |
| l1_sc | 16×16×1×1 | 23 | 13,874 – 56,098 | 9,769 | 17 bit |
| l1_b | 16×16×3×3 | 23 | 16,800 – 33,244 | 28,183 | 18 bit |
| l2_a | 32×16×3×3 | 23 | 15,038 – 41,881 | 72,749 | 19 bit |
| l2_sc | 32×16×1×1 | 24 | 20,395 – 48,057 | 22,494 | 17 bit |
| l2_b | 32×32×3×3 | 24 | 26,767 – 41,215 | 36,027 | 19 bit |
| fc | 10×32 | conf_s = 10 | conf_m = 33,509 | \|bfc\| ≤ 6,165 | 20 bit |

### 3.4 準確度（`quant.py` 實測）
| 項目 | 結果 |
|---|---|
| 與浮點模型類別一致（第 0–63 張） | 63 / 64（98.4%），不一致的是第 53 張 |
| 與浮點模型類別一致（全部 192 張） | 98.4% |
| confidence 平均誤差 | 0.08（logit 單位） |
| 浮點模型對真實標籤的錯誤率（64 張） | 0.328（來自助教 notebook） |

### 3.5 Golden 檔案
| 檔案 | 內容 |
|---|---|
| `00_TESTBED/PATTERN/GOLDEN/image_<i>/*.hex` | 每層中間結果（累加器 32-bit、activation 8-bit、pool 12-bit、logit 32-bit），順序 `c×H×W + y×W + x` |
| `00_TESTBED/PATTERN/GOLDEN/image_<i>/result.txt` | category、confidence |
| `98_ALGORITHM/quant_params.npz` | 全部整數參數 |
| `00_TESTBED/PATTERN/golden_results.txt` | **待產生**：每張圖一行 `category confidence`，給 HOST model 比對 |

---

## 4. 外部介面

### 4.1 Port
與題目 spec 完全相同（host AXI4-Lite 8-bit 位址 / 32-bit 資料；DRAM AXI4 19-bit 位址 / 256-bit 資料 / 8-bit ID）。

### 4.2 Host 暫存器
| 位址 | 存取 | 內容 |
|---|---|---|
| 0x00 | 寫 | `[3:0]` = 本批張數 − 1（N = 值 + 1，1–16）。寫入即開始這一批 |
| 0x04 | 讀 | `[0]` = 1 完成、0 忙碌；其他 bit 為 0 |
| 0x08 + 4k | 讀 | 第 k 張的結果（k = 0–15）：`[31:28]` category、`[27:0]` confidence（28-bit 有號，2 補數） |
| 其他 | | 讀回 0，寫入忽略 |

### 4.3 Host 協定（AXI4-Lite slave 的行為）
| 通道 | 行為 |
|---|---|
| AW / W | 可以分開或同時到達；兩者都收到後才執行寫入；`wstrb` 忽略（視為整個 32 bit 寫入） |
| B | 寫入後下一個 cycle 拉起 `bvalid`，`bresp = 2'b00`，等到 `bready` |
| AR / R | 收到 AR 後下一個 cycle 拉起 `rvalid`，`rresp = 2'b00`，等到 `rready` |
| 一次只處理一筆 | 寫入或讀取未完成前，`awready`/`arready` 拉低 |

### 4.4 Finish 旗標的規則（避免讀到上一批的舊 finish）
- host 端暫存器 `finish_h`：**host 寫 0x00 的那個 cycle 立刻清成 0**。
- core 端 `done`：收到 start 時清 0，整批（含最後一張的 FC）完成時設 1。
- `done` 經 2FF 同步到 host 端，**偵測到上升緣**才把 `finish_h` 設成 1。

### 4.5 DRAM 介面使用方式
| 項目 | 設定 |
|---|---|
| 寫入通道 | 不使用：`awvalid = 0`、`wvalid = 0`、`bready = 1` |
| `arid` | 固定 0 |
| `arsize` | 3'b101（每 beat 32 bytes） |
| `arburst` | 2'b01（INCR） |
| `arlen` | 該 burst 的 beat 數 − 1；**不跨 4 KB 邊界**（AXI 規定） |
| outstanding | 一次只有一筆讀取請求，收到 `rlast` 才發下一筆 |
| `rready` | = 資料 async FIFO 不滿 |

---

## 5. DRAM 資料格式與 `model.dat`

### 5.1 通則
- DRAM 每個 word（= 一個 beat）256 bit = 32 bytes；word 編號 = byte 位址 / 32。
- **byte k 放在 bit [8k+7 : 8k]**（AXI 的 byte lane 規則）。
- `model.dat`：每行 64 個 hex 字元 = 一個 word，**最高位元組（byte 31）寫在最左邊**；用 `@<word 編號>` 指定起點。

### 5.2 位址配置
| byte 位址 | word 編號 | 內容 | 大小 |
|---|---|---|---|
| 0x00000 + 0x1000·n | 0x080·n | 第 n 張圖（testbench 每批寫入） | 96 beats |
| **0x10000** | **0x800** | **WSEC：conv 權重** | 614 beats |
| **0x15000** | **0xA80** | **PSEC：參數、FC** | 43 beats |
| 0x40000 以上 | | 暫存區，本設計不使用 | |

### 5.3 圖片格式（題目給定）
3,072 bytes，CHW：beat `b = c×32 + y`（一個 beat = 一列 32 個像素），byte lane `x` = 第 x 行。

### 5.4 WSEC：conv 權重
- 以 **WBUF word**（128 bit = 16 個 int8 權重）為單位，共 1,227 個 word。
- WBUF word 內：**byte r = 該批第 r 個輸出通道（o = og×16 + r）的權重**。
- 每個 DRAM beat 放兩個 WBUF word：bit [127:0] = word 2j，bit [255:128] = word 2j+1；最後一個 beat 的高半部補 0。
- **排列順序 = 硬體使用順序**：

```
for layer in [stem, l1_a, l1_sc, l1_b, l2_a, l2_sc, l2_b]:     # 執行順序（shortcut 在主路徑前）
  for og in range(ceil(Cout/16)):                              # 輸出通道批
    for i in range(Cin):                                       # 輸入通道
      for ky in range(K):
        for kx in range(K):
          word = { W[og*16+15][i][ky][kx], ..., W[og*16+0][i][ky][kx] }   # byte 0 = 第 og*16+0 個
```

| 層 | 起始 word（`wbase`） | word 數 |
|---|---|---|
| stem | 0 | 27 |
| l1_a | 27 | 144 |
| l1_sc | 171 | 16 |
| l1_b | 187 | 144 |
| l2_a | 331 | 288 |
| l2_sc | 619 | 32 |
| l2_b | 651 | 576 |
| 合計 | | 1,227（614 beats，19,648 bytes） |

### 5.5 PSEC：參數與 FC
所有多 byte 數值皆為 **little-endian**（低位 byte 在小的 byte lane）。

| PSEC beat | 內容 |
|---|---|
| 0 | header：byte 0–6 = 7 層的 s（順序同 5.4）；byte 8–9 = conf_m（u16）；byte 10 = conf_s；其餘 0 |
| 1 – 30 | 10 個參數區塊，每個 3 beats（見下） |
| 31 – 40 | FC 權重：beat 31+k 的 byte j = `Wfc[k][j]`（int8），k = 0–9 |
| 41 – 42 | FC bias：`bfc[0..9]` 依序為 int32，共 40 bytes，其餘補 0 |

**參數區塊**（一個「層 × 輸出通道批」一個，3 beats）：
| 區塊內 beat | 內容 |
|---|---|
| 0 | `b_q[og*16 + 0..7]`，每個 int32 |
| 1 | `b_q[og*16 + 8..15]`，每個 int32 |
| 2 | `m[og*16 + 0..15]`，每個 u16 |

| 區塊編號 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 層, og | stem,0 | l1_a,0 | l1_sc,0 | l1_b,0 | l2_a,0 | l2_a,1 | l2_sc,0 | l2_sc,1 | l2_b,0 | l2_b,1 |

---

## 6. 硬體架構

### 6.1 模組階層（control / datapath 分離）
```
resacc                         (top；只有連線與 reset 同步器)
├── host_if                    (clk_host) AXI4-Lite slave、暫存器、start toggle、finish 同步
├── dram_if                    (clk_dram) AXI4 讀取狀態機
├── async_fifo  u_req_fifo     dut → dram：讀取請求 {addr[18:0], len[7:0]}，深度 4
├── async_fifo  u_dat_fifo     dram → dut：讀回資料 256 bit，深度 16
└── core                       (clk_dut) 整體流程 FSM、result 暫存器、WBUF/PBUF/FCMEM/IMG
    ├── loader                 FIFO 資料 → WBUF / PBUF / FCMEM / IMG / header 暫存器
    ├── sram_sp ×4             WBUF PBUF FCMEM IMG
    └── engine                 (只是包裝：連線三個子模組)
        ├── engine_ctrl        control path：步驟/批/迴圈計數器、位址、token 產生 (S0)
        ├── engine_dp          datapath：FM_A/B/C、window+選 byte (S1)、運算元暫存 (S2)、
        │   │                  參數暫存器、排出序列器 (D0–D3)、pool
        │   ├── sram_sp ×3     FM_A FM_B FM_C
        │   ├── pe_array       16×8 個 pe
        │   │   └── pe         8×9 有號乘法器 + 1 個加法器 + 2 組 20-bit 累加器
        │   └── requant_lane ×8  每條 1 個 20×16 乘法器（殘差層 2 cycle/列）
        └── fc_unit            FC（序列 MAC）、argmax、confidence（shift-add 乘法）
共用元件：async_fifo, sync_2ff, reset_sync, sram_sp
```
命名規則：子模組 port 用 `i_`/`o_` 前綴（top `resacc` 保留課程規定的名字）、狀態 `S_`、
localparam 大寫、instance `u_`；`always @(*)` 算 `_w`、時序區塊只做 `_r <= _w`；
控制暫存器有 reset、資料暫存器不 reset。

### 6.2 Reset 與時脈域
- 每個時脈域各自一個 `reset_sync`：**非同步拉低、同步放開**（2 級 flip-flop）。
- SRAM 內容不 reset。

### 6.3 CDC 一覽
| 訊號 | 方向 | 方式 |
|---|---|---|
| 讀取請求 | dut → dram | `async_fifo`（Gray code 指標 + 2FF），深度 4 |
| 讀回資料 | dram → dut | `async_fifo`，深度 16 × 256 bit；寫入端 `rready = !full` |
| start | host → dut | host 端每次寫 0x00 翻轉 `start_tgl`；dut 端 2FF 同步後偵測翻轉。`n_minus_1` 在翻轉前已寫好且保持不變，dut 偵測到翻轉後才取樣 |
| done | dut → host | 準位訊號，2FF 同步，host 端偵測上升緣（見 4.4） |
| 結果暫存器 | dut → host | **刻意不做同步**：資料在 done 拉起前已固定，host 只在看到 finish 後讀取。合成時要設 false path |

### 6.4 `async_fifo`
- 參數：`WIDTH`、`DEPTH`（2 的冪次）。
- 指標比位址多 1 bit；寫/讀指標轉 Gray code 後經 2FF 同步到對方。
- `empty`：讀出端的讀指標 == 同步過來的寫指標。
- `full`：寫入端的寫指標與同步過來的讀指標，最高兩位相反、其餘相同。
- 行為保守：可能晚 2–3 cycle 才看到新資料或空位，但不會讀錯或覆蓋。

### 6.5 `dram_if`（clk_dram）
1. 等 `u_req` 不空 → 取出 `{addr, len}`。
2. 拉起 `arvalid`（`araddr = addr`、`arlen = len`），等 `arready`。
3. 每收到一個 beat（`rvalid && rready`）寫進 `u_rdat`；直到 `rlast`。
4. 回到 1。

### 6.6 讀取請求清單
| 用途 | 起始位址 | burst 長度（beats） |
|---|---|---|
| 模型 WSEC | 0x10000、0x11000、0x12000、0x13000 | 128、128、128、128 |
| | 0x14000 | 102 |
| 模型 PSEC | 0x15000 | 43 |
| 第 n 張圖 | 0x1000·n | 96 |

### 6.7 `loader`（clk_dut）
依目前的載入目標，把 `u_rdat` 的 beat 寫到對應位置：
| 目標 | 每個 beat 的處理 | 每 beat 花費 |
|---|---|---|
| WBUF | 拆成兩個 128-bit word，依序寫入（WBUF 為單埠 128-bit） | 2 cycles |
| PSEC beat 0 | 寫進 header 暫存器：`s[0..6]`、`conf_m`、`conf_s` | 1 cycle |
| PSEC beat 1–30 | 寫進 PBUF[beat−1] | 1 cycle |
| PSEC beat 31–40 | 寫進 FCMEM[beat−31] | 1 cycle |
| PSEC beat 41–42 | 寫進 `bfc` 暫存器 | 1 cycle |
| IMG | beat `b`（= c×32 + y）寫進 IMG word `c×16 + y/2` 的半邊（y 偶數 → byte 0–31，奇數 → byte 32–63），用 byte enable | 1 cycle |

### 6.8 晶片內記憶體

所有 SRAM 都用 `sram_sp`：單埠、**讀取延遲 1 cycle**、byte enable 寫入、內容不 reset。

| 名稱 | 深度 × 寬度 | 容量 | 內容 |
|---|---|---|---|
| WBUF | 1,227 × 128 bit | 19,632 B | conv 權重（5.4 的順序） |
| PBUF | 30 × 256 bit | 960 B | 10 個參數區塊 |
| FCMEM | 10 × 256 bit | 320 B | FC 權重，一個 word = 一個類別的 32 個權重 |
| IMG | 48 × 512 bit | 3,072 B | 輸入圖片 |
| FM_A | 64 × 512 bit | 4,096 B | stem 輸出 |
| FM_B | 16 × 512 bit | 1,024 B | l1_a 輸出；之後放 l2_a 輸出（用 8 個 word） |
| FM_C | 16 × 512 bit | 1,024 B | l1 區塊輸出 |
| **合計** | | **30,128 B** | |

另有 flip-flop：header 暫存器、`bfc`、參數暫存器（2 組 × 16 × (18+16) bit）、累加器 128 × 2 × 20 bit、pool 32 × 12 bit、結果 16 × 32 bit。

### 6.9 Feature map 的存放格式（IMG、FM_A/B/C 共用）

一個 word = 64 bytes。tensor 形狀 C×H×W，`plane = H×W`：

```
若 plane ≥ 64：                          若 plane < 64：
  rows_per_word  = 64 / W                  ch_per_word = 64 / plane
  words_per_ch   = plane / 64              word = c / ch_per_word
  word = c × words_per_ch + y / rows_per_word
  byte = (y mod rows_per_word) × W + x     byte = (c mod ch_per_word) × plane + y × W + x
```

| tensor | 形狀 | 放在 | 每 word 裝 | 用到的 word |
|---|---|---|---|---|
| 圖片 | 3×32×32 | IMG | 2 列 | 48 |
| stem 輸出 | 16×16×16 | FM_A | 4 列 | 64 |
| l1_a 輸出 | 16×8×8 | FM_B | 1 個通道 | 16 |
| l1 區塊輸出 | 16×8×8 | FM_C | 1 個通道 | 16 |
| l2_a 輸出 | 32×4×4 | FM_B | 4 個通道 | 8 |
| l2 區塊輸出 | 32×4×4 | **不存**（直接進 pool） | | |

**性質（設計依據）**：任何一個像素群在某個 (i, ky) 需要的輸入，一定都在**同一個 word** 裡；任何一個像素群的 8 個輸出，一定是同一個 word 裡**連續、對齊 8 的 8 個 byte**。

### 6.10 運算的基本單位：批（batch）

**一批** = 16 個輸出通道（第 og 批）× 8 個輸出像素（第 g 個像素群）= 128 個輸出值。

**像素群**：輸出像素以 raster 順序編號 `n = oy × Wo + ox`，第 g 群 = `n = 8g … 8g+7`；第 p 行 PE 負責 `n = 8g + p`。

| 層 | 輸出 | 像素群 | 群數 | og 數 | 批數 | 每批 cycle（= Cin×K×K） |
|---|---|---|---|---|---|---|
| stem | 16×16 | 半列 | 32 | 1 | 32 | 27 |
| l1_a | 8×8 | 一列 | 8 | 1 | 8 | 144 |
| l1_sc + l1_b | 8×8 | 一列 | 8 | 1 | 8 | 16 + 144 |
| l2_a | 4×4 | 兩列 | 2 | 2 | 4 | 144 |
| l2_sc + l2_b | 4×4 | 兩列 | 2 | 2 | 4 | 16 + 288 |

**批的順序**：`for og: for g:`。**批內的順序**：`for i: for ky: for kx:`，每個 cycle 一組 (i, ky, kx)。

### 6.11 每個 cycle 的運算元

**權重**（16 個）：
```
WBUF 位址 = wbase[layer] + og × (Cin×K×K) + i × K×K + ky × K + kx
第 r 列的權重 = 讀出 word 的 byte r（int8）
```

**輸入**（8 個，第 p 行）：
```
n  = 8g + p,  oy = n / Wo,  ox = n mod Wo
iy = oy × stride + ky − pad
ix = ox × stride + kx − pad
valid = (0 ≤ iy < H) && (0 ≤ ix < W)
```
- **讀哪個 word**：以 `iy₀`（第 0 行的 iy）夾到 [0, H−1] 後代入 6.9 的公式。
- **何時讀**：kx = 0 時讀一次，存進 window 暫存器，同一個 (i, ky) 的 kx = 0..K−1 共用（1×1 的層每個 cycle 讀一次）。
- **in_sel** 對每一行：`byte = 6.9 的 byte 公式(i, iy, ix)`，從 window 暫存器挑出該 byte。
- **運算元（9-bit 有號）**：
  - `valid = 0` → 0（padding）
  - stem：`{1'b0, pixel} − 128`
  - 其他層：`{1'b0, activation}`

### 6.12 PE
```
product = weight(int8) × operand(9-bit 有號)        # 17-bit 有號
acc[bank] <= acc[bank] + product                    # 20-bit 有號
```
- 每個 PE 有 **bank0、bank1** 兩組累加器。
- **批的第一個 cycle**：累加器不清 0，而是載入 `bias`（第 r 列用 `b_q[og×16 + r]`），再加上第一個 product。
- 一般層：每批輪流用 bank0、bank1（ping-pong）。
- 殘差層：shortcut 用 bank1，主路徑用 bank0。

### 6.13 參數暫存器
- 每換一個 (layer, og)，從 PBUF 讀 3 個 word 到**參數暫存器組 0**（16 × bias 18-bit、16 × m 16-bit）。
- 殘差層再讀 3 個 word（shortcut 的區塊）到**參數暫存器組 1**。
- 讀取在該層第一批開始前完成（≤ 6 cycles），由 ctrl_fsm 安排。

### 6.14 requant（8 條通道，`requant_lane`）

一批結束後，把 128 個累加值依「列」排出，8 條通道分別處理 8 行。每條通道**只有 1 個乘法器**（20×16 → 37 bit）：

```
一般層（每列 1 cycle）：
    y = clamp8( (acc0[r][p] × m[r] + 2^(s−1)) >>> s )
殘差層（每列 2 cycle）：
    第 1 拍： part = acc1[r][p] × m_sc[r] + 2^(s−1)      # 存進通道內的 part 暫存器
    第 2 拍： y    = clamp8( (acc0[r][p] × m_b[r] + part) >>> s )
```
- s 只會是 23 或 24（loader 有模擬用的 assertion 檢查），所以右移是**二選一的接線**，不是可變移位器。
- 捨入常數 2^(s−1) 併進同一個加法器（一般層加 rnd、殘差第 1 拍加 rnd、第 2 拍加 part），stage 2 只有一個 38-bit 加法器。
- 通道延遲 2 cycle：stage 1 乘法、stage 2 加法 + clamp。
- 一般層 16 cycle 排完一批；殘差層 32 cycle。
- **寫回**：8 個 y 是通道 `c = og×16 + r`、像素 `n = 8g..8g+7`，寫到 6.9 公式算出的 word（`{8{y}}` + one-hot byte enable，逐 byte 寫）。
- **l2 區塊**（pool 模式）：不寫回，改成 `pool[c] += y[0] + … + y[7]`。
- 一般層的排出與下一批的運算重疊（另一個 bank）；殘差層排出時陣列暫停（兩個 bank 都在被讀）。

### 6.15 Buffer 輪替（每張圖）
| 步驟 | 讀 | 寫 | requant 模式 |
|---|---|---|---|
| stem | IMG | FM_A | 一般 |
| l1_a | FM_A | FM_B | 一般 |
| l1_sc → l1_b（每批先 sc 再 b） | FM_A（sc）、FM_B（b） | FM_C | 殘差 |
| l2_a | FM_C | FM_B | 一般 |
| l2_sc → l2_b | FM_C（sc）、FM_B（b） | — | 殘差 + pool |

每一步的讀和寫都在不同 SRAM，所以單埠 SRAM 不會衝突。

### 6.16 `fc_unit`
狀態：`S_IDLE → S_READ → S_LOAD → S_MAC → S_CMP →（10 類重複）→ S_MUL → S_SHIFT → S_OUT`
1. 直接讀 engine_dp 的 `pool[0..31]`（不複製；下一張圖的 l2 開始前 ctrl 會等 `fc_busy = 0` 再清 pool）。
2. 每一類 k：從 FCMEM 讀 1 個 word（32 個 int8 權重），`acc = bfc[k]`，32 cycle 序列乘加（13×8 乘法器），32-bit 累加器。
3. argmax：`k == 0` 時直接設定，之後只有 `acc > max` 才更新（平手保留較小的 k）。
4. `confidence = (max × conf_m + 2^(conf_s−1)) >>> conf_s`：16 cycle shift-add（逐 bit 掃 conf_m），再一次加捨入常數、每 cycle 右移 1 位直到 conf_s 次。取低 28 bit。
5. 寫入 `result[n] = {category[3:0], confidence[27:0]}`。

FC 約 10×(32+3) + 16 + 25 ≈ 400 cycle，與下一張圖的 stem / l1 同時進行，不在關鍵路徑上。

### 6.17 流程（`core` 的 FSM + `engine_ctrl` 的 FSM）
`core`：
```
reset 放開
  └─► S_LOAD_MODEL：請求 WSEC（5 個 burst）+ PSEC（1 個 burst），loader 寫入
S_IDLE：等 start（且模型已載入）
  └─► 取樣 N；請求第 0 張圖
for n in 0 .. N−1:
    S_WAIT_IMG：等第 n 張圖載入完成
    S_RUN：     engine 開始；engine 回報 stem_done 後就請求第 n+1 張圖（IMG 不再被讀）
S_WAIT_FC：等最後一張的 fc_unit 完成 → done = 1 → S_IDLE
```
`engine_ctrl`（每張圖）：
```
S_IDLE ─start─► S_PAR_WAIT ─quiet─► S_PAR（從 PBUF 讀 3 或 6 個 word 到參數暫存器）
     ─► S_BATCH ─► S_ISSUE（Cin·K·K 個 cycle，每 cycle 送出 1 個 token）─► 下一批 / S_STEP_END
S_STEP_END：quiet 後換下一步（stem → l1_a → l1 殘差 → l2_a → l2 殘差）
quiet = 運算管線與排出都已清空
```
- 換 (layer, og) 時才換參數，所以必須 quiet；殘差層每批開始前也要 quiet（避免 shortcut 覆寫 bank1，見 ARCH.md 的 race bug）。
- 第 3 → 4 步前等 `fc_busy = 0`，再送 `pool_clr`；第 4 步結束送 `fc_start` 與 `done`。

### 6.18 管線與 token（RTL 定案）
control path 每 cycle 產生一個 **token**（這一拍需要的所有控制資訊），token 跟著資料一起往後走，datapath 不需要自己知道「現在在第幾層、第幾批」。

| 階段 | 在哪 | 做什麼 |
|---|---|---|
| S0 | engine_ctrl | 迴圈計數器 (i, ky, kx)、算 WBUF / FM 位址、發出 SRAM 讀取、產生 token |
| S1 | engine_dp（`tk1_*`） | SRAM 資料回來；`win_ld` 時更新 window 暫存器；8 個 64 選 1 選 byte（含 padding 0 / stem −128） |
| S2 | engine_dp（`tk2_*`） | 權重、運算元進暫存器 |
| PE | pe_array | 乘法 + 累加（first 拍載入 bias） |

token 內容：`valid, first, last, final, bank, bset, win_ld, src, stem, sel[48], vld[8]`，加上排出描述 `dmode, ddst, dword, dbyte, dpacked, dwstep, dog, ds24`。

排出（token 的 `last && final` 到 PE 後啟動）：
| 階段 | 做什麼 |
|---|---|
| D0 | 選出第 r 列的 8 個累加值與 m，進通道輸入暫存器 |
| D1 | 通道 stage 1：乘法 |
| D2 | 通道 stage 2：加法 / 捨入 / clamp |
| D3 | 寫回 FM SRAM（或加進 pool） |

---

## 7. 位元寬度總表
| 訊號 | 寬度 | 有號 |
|---|---|---|
| 權重（conv、fc） | 8 | 是 |
| 輸入運算元 | 9 | 是 |
| activation | 8 | 否 |
| PE 乘積 | 17 | 是 |
| conv 累加器 | 20 | 是 |
| bias（conv） | 18 | 是 |
| m | 16 | 否 |
| s | 1（只有 23 或 24，用 s24 旗標） | 否 |
| requant 中間值 v | 38 | 是 |
| pool | 12 | 否 |
| fc 乘積 | 20 | 是 |
| fc 累加器 | 32 | 是 |
| bfc | 32（實際 ≤ 14） | 是 |
| confidence | 28 | 是 |

---

## 8. 效能與面積估算

### 8.1 時間

**RTL 實測（control/datapath 分離 + 面積最佳化，clk_dut 2.0 ns）**
| 項目 | 實測 |
|---|---|
| 64 張總時間（READ_LATENCY 12） | **739,541.5 ns（Goal 的 74.0%）** |
| READ_LATENCY 0 | 739,518.1 ns |
| READ_LATENCY 12 + DRAM 隨機空拍 30% | 739,541.5 ns（與無空拍相同：圖片載入被運算藏住） |
| 每批 16 張（第 2–4 批） | 184,228.2 ns → 每張約 5,757 cycles |
| 2 張總時間 | 26,527.9 ns < 739,542 / 16 = 46,221 ns ✓ |
| 正確性 | 64/64 bit-exact，與浮點模型類別一致 63/64 |

**原本的估計（`97_ARCH/schedule_C.py`）**
| 步驟 | 批數 | 每批 cycle | 本步 cycle |
|---|---|---|---|
| stem | 32 | 27 | 885 |
| l1_a | 8 | 144 | 1,173 |
| l1 殘差 | 8 | 16 + 144 + 16（排出等待） | 1,413 |
| l2_a | 4 | 144 | 597 |
| l2 殘差 | 4 | 16 + 288 + 16 | 1,285 |
| **每張圖** | | | **5,353** |

| 項目 | 估計 |
|---|---|
| 開機載入模型 | 614 beats × 2 + 43 ≈ 1,300 cycles（2.6 µs，只有一次） |
| 每批額外開銷 | 第一張圖載入 + 最後一張 FC + 同步，約 500 cycles |
| **64 張總時間** | **約 692,000 ns（Goal 的 69%）**，不含 host 輪詢間隔 |
| 2 張圖 | 約 25 µs < 692 / 16 = 43 µs ✓ |
| PE 利用率 | 運算期間約 95% |

### 8.2 記憶體與面積（數量級）
- SRAM 合計約 30 KB（約 24 萬 bit）。以 16nm 小型 macro 每 bit 0.15–0.25 µm² 估計，約 **37,000–62,000 µm²**（上限 500,000 µm² 的 7–12%）。
- 邏輯（Yosys generic gates，SRAM 當黑盒子，`02_SYN/area.ys`）：
| 模組 | cells |
|---|---|
| pe_array（含 128 個 pe） | 103,512 |
| requant（8 通道） | 20,624 |
| engine 其餘（ctrl + dp 其他） | 20,395 |
| fc_unit | 4,392 |
| **engine 合計** | **148,923** |
| **resacc 全部** | **159,690** |
| flip-flop | 14,201 |
- 邏輯深度（Yosys `ltp`，單位：gate 層數，加法器是 ripple，只能拿來相對比較）：pe 44、engine_ctrl 45、requant_lane 77（38-bit 加法 + clamp）、engine_dp 77、fc_unit 109。重算：`02_SYN/run_yosys.sh`。

### 8.3 能量（相對值，`97_ARCH/compare.py`）
以 MAC = 1、暫存器 = 1、buffer = 6（MIT tutorial）估計，每張圖約 **2.1 M**；權重常駐晶片，DRAM 只在開機讀一次模型、每張圖讀 3 KB 圖片。

---

## 9. 驗證計畫

### 9.1 環境
- 模擬器：**Icarus Verilog**（`iverilog -g2012`），可選 Verilator。
- 不使用助教的加密模型；自行撰寫：

| 檔案 | 內容 |
|---|---|
| `10_AXI_DRAM/axi_dram.sv` | AXI4 slave 行為模型：512 KB，`$readmemh` 載入 `model.dat`，參數化讀/寫延遲（預設 12 / 16 cycles），支援 INCR burst 與 `rready` 背壓，task 載入圖片 |
| `00_TESTBED/host_model.sv` | AXI4-Lite master：每批寫 0x00、輪詢 0x04、讀 16 個結果，與 golden 比對並印出錯誤率與時間 |
| `00_TESTBED/TESTBED.sv` | 產生三個時脈（clk_dut 週期 2.0 ns）、reset，批次控制與圖片載入，**外部模型在負緣驅動輸出** |

### 9.2 測試層級
| 層級 | 對象 | 比對 |
|---|---|---|
| 單元 | `async_fifo` | 兩個不相關時脈、隨機讀寫速率，資料順序與內容正確，不溢位 |
| 單元 | `sram_sp` | 讀取延遲、byte enable |
| 單元 | `dram_if` + `axi_dram` | burst、4KB 邊界、背壓 |
| 單元 | `pe_array` + `in_sel` + `agu` | 單一層對 GOLDEN 的 `*_acc.hex` |
| 單元 | `requant` | 對 GOLDEN 的 `*_out.hex` |
| 子系統 | `core`（直接載入 SRAM） | 每層結束後讀出 FM，依 6.9 換算後對 GOLDEN 逐 byte 比對 |
| 系統 | `resacc` + 全部模型，2 張 | category 與 confidence 對 `golden_results.txt` |
| 系統 | 64 張（4 批）、READ_LATENCY = 0 與 12 兩種 | 全部 bit-exact；記錄總時間 |

### 9.3 通過標準
1. 64 張的 category 與 confidence 與 `quant.py` **100% 相同**。
2. 與浮點模型的類別一致率 = 63/64（與 3.4 相同）。
3. 64 張總時間 < 10^6 ns；2 張的時間 < 64 張 / 16。

---

## 10. 檔案結構
```
ResAcc/
├── README.md  .gitignore
├── docs/         report.pdf, SPEC.md, ARCH.md
├── 00_TESTBED/   TESTBED.sv, host_model.sv, stub/, unit/, periph/, layer/,
│                 PATTERN/（IMAGES、csv、GOLDEN/、golden_results.txt、model.dat）
├── 01_RTL/       *.v, rtl.f（周邊）, engine.f（engine）, run_sys.sh, run_periph.sh
├── 02_SYN/       area.ys, depth.ys.in, sram_bb.v, run_yosys.sh（Yosys 面積 / 深度估計）
├── 10_AXI_DRAM/  axi_dram.sv
├── 97_ARCH/      arch_model.py, dataflow.py, compare.py, schedule_C.py
└── 98_ALGORITHM/ common.py, float_ref.py, quant.py, sweep.py, gen_model_dat.py, hw_model.py, quant_params.npz, environment.yml, *.ipynb, resnet6.pth
```

### 實作順序
1. `gen_model_dat.py`（依第 5 章）＋ `golden_results.txt`，並用 Python 模擬「依 6.11 的位址讀 model.dat 算出 GOLDEN」來驗證格式
2. `sram_sp`、`sync_2ff`、`reset_sync`、`async_fifo` ＋ 單元測試
3. `axi_dram.sv`、`dram_if`、`loader`
4. `pe_array`、`in_sel`、`agu`、`requant`（逐層對 GOLDEN）
5. `fc_unit`、`ctrl_fsm`、`core`
6. `host_if`、`host_model.sv`、`TESTBED.sv`，系統模擬

---

## 11. 待確認事項與風險
| 項目 | 說明 | 處理 |
|---|---|---|
| 管線深度 | 已定案：運算 3 級、排出 4 級 | 完成 |
| in_sel 面積 | 8 個 64 選 1 的 byte mux | 可利用 stride 規律簡化；Yosys 估計後再決定 |
| WBUF 載入 2 cycle/beat | 為了單純，開機多花約 1.2 µs | 必要時改成雙 bank |
| SRAM macro 規格 | 不確定 ADFP 提供的寬度與深度 | 只透過 `sram_sp` wrapper 使用，之後可替換 |
| host 輪詢間隔 | 原 HOST 加密 | 自訂；時間估算不含此項 |
| 真實標籤 | 沒有 CIFAR-10 標籤 | 錯誤率以浮點模型 0.328 為基準，量化版只差 1 張 |
| 原 TESTBED 的 define | 原檔寫死 `RESACC_CLK_PERIOD 1`、`READ_LATENCY 0` | 我們的 TESTBED 用 2.0 ns，並測試 0 與 12 兩種延遲 |

---

## 附錄 A：位址計算的實例

**情境**：l1_a，og = 0，g = 3（輸出第 3 列），i = 5，ky = 0。

| 項目 | 計算 | 結果 |
|---|---|---|
| 權重位址（kx = 0） | 27 + 0 + 5×9 + 0×3 + 0 | WBUF word 72 |
| 輸入列 | iy = 3×2 + 0 − 1 | 5 |
| 輸入 word | FM_A：rows_per_word = 4，words_per_ch = 4 → 5×4 + 5/4 | word 21 |
| 該列在 word 內的起點 | (5 mod 4) × 16 | byte 16 |

| kx | ix（p = 0..7） | 選出的 byte |
|---|---|---|
| 0 | −1, 1, 3, …, 13 | 0（padding）、17、19、21、23、25、27、29 |
| 1 | 0, 2, …, 14 | 16、18、20、22、24、26、28、30 |
| 2 | 1, 3, …, 15 | 17、19、21、23、25、27、29、31 |

**寫回**：這一批排出第 r 列時，輸出通道 c = r、像素 (3, 0..7)，寫到 FM_B word c、byte 24–31（8×8 平面一個 word，byte = 3×8 + x）。
