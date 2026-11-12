"""
schedule_C.py -- dataflow C 的實際排程與 buffer 大小估算（比 dataflow.py 多算了實作上的細節）

假設（都記在 ARCH.md）：
  * PE 陣列 16 輸出通道 × 8 輸出像素，每個 PE 有 2 組 20-bit 累加器（bank0 / bank1）
  * 一個「像素群」= 8 個輸出像素；一批 = (16 個輸出通道, 1 個像素群)；一批要 Cin×K×K 個 cycle
  * 累加器用 bias 當初始值
  * requantize 單元 R_LANES 條，每 cycle 處理 R_LANES 個輸出 → 一批 128 個要 128/R_LANES 個 cycle
  * 一般層：兩組累加器輪流用（ping-pong），排出結果和下一批運算重疊
  * 殘差層：shortcut 算進 bank1、主路徑算進 bank0，排出時兩組一起讀 → 這段時間陣列要等
  * 每層結束要等最後一批排完、寫回 buffer，下一層才能開始讀（PIPE 個 cycle 的管線延遲）
  * l2 的輸出不存，requantize 出來直接加進 pool 的 32 個加總
  * FC 用獨立的小單元，和下一張圖的 stem 重疊
"""
import math

R_LANES = 8
PIPE = 5                 # SRAM 讀 → 選擇 → 乘 → 累加 → 寫回 的管線深度（估計）
F_MHZ = 500
N_IMG = 64
BEAT_BYTES = 32          # 一個 256-bit AXI beat

DRAIN = 128 // R_LANES   # 一批 128 個輸出要排多久


def plain_layer(Ci, Co, K, n_pix):
    groups = math.ceil(Co / 16) * math.ceil(n_pix / 8)
    per = Ci * K * K
    assert DRAIN <= per, 'requant 太慢，會拖住陣列'
    return groups * per + DRAIN + PIPE, groups, per


def residual_layer(Ci_b, Ci_sc, Co, n_pix):
    groups = math.ceil(Co / 16) * math.ceil(n_pix / 8)
    per = Ci_sc * 1 + Ci_b * 9 + DRAIN          # shortcut + 主路徑 + 排出（陣列等待）
    return groups * per + PIPE, groups, per


if __name__ == '__main__':
    rows = [
        ('stem',           *plain_layer(3, 16, 3, 256)),
        ('l1_a',           *plain_layer(16, 16, 3, 64)),
        ('l1_b + l1_sc',   *residual_layer(16, 16, 16, 64)),
        ('l2_a',           *plain_layer(16, 32, 3, 16)),
        ('l2_b + l2_sc',   *residual_layer(32, 16, 32, 16)),
    ]
    total = 0
    print(f"{'層':14s} {'批數':>4s} {'每批cycle':>9s} {'本層cycle':>9s}")
    for n, cyc, g, per in rows:
        total += cyc
        print(f'{n:14s} {g:4d} {per:9d} {cyc:9,d}')
    print(f'每張圖 {total:,} cycle（理想值 5,088；多出的是排出等待與管線延遲）')

    w_beats = math.ceil(19952 / BEAT_BYTES)
    first = w_beats + 96 + 30        # 權重 + 第一張圖 + DRAM 延遲等
    t = (first + total * N_IMG) * 1000 / F_MHZ
    print(f'第一次啟動：權重 {w_beats} beat + 圖片 96 beat ≈ {first} cycle')
    print(f'64 張估計 ≈ {t:,.0f} ns（Goal 的 {t / 1e6:.0%}），還沒算 host 輪詢與批次切換')

    print('\n晶片內記憶體：')
    mem = [('權重 buffer', (27 + 144 + 144 + 16 + 288 + 576 + 32) * 16, '128-bit word：16 個濾鏡各 1 個權重'),
           ('參數 (bias, m)', 160 * (18 + 16) // 8, '每個輸出通道 bias 18b + m 16b，共 160 組'),
           ('FC 權重 + bias', 320 + 10 * 3, 'FC 單元自己的小記憶體'),
           ('IMG（輸入圖片）', 3 * 32 * 32, '512-bit word = 2 列；等 stem 用完就載下一張'),
           ('FM_A', 16 * 16 * 16, 'stem 輸出'),
           ('FM_B', 16 * 8 * 8, 'l1_a 輸出 → 之後放 l2_a 輸出'),
           ('FM_C', 16 * 8 * 8, 'l1 區塊輸出')]
    for n, b, note in mem:
        print(f'  {n:16s} {b:6,d} B   {note}')
    print(f"  {'合計':16s} {sum(b for _, b, _ in mem):6,d} B")
    print(f'  累加器 128 × 2 組 × 20 bit = {128 * 2 * 20:,} bit')
