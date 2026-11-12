"""
arch_model.py -- 硬體架構的數字模型（不是 RTL，只是用來算 cycle、面積、流量做比較）

第 2 輪：效能預算
  Part A : 每一層的運算量（MAC 數）是怎麼來的
  Part B : 時間預算 → 每個 cycle 至少要做幾個 MAC
  Part C : 利用率（utilization）的概念，用一個示範的 PE 排法說明為什麼 PE 要比理論值多
"""
import math

# ---------------------------------------------------------------- 網路形狀
# (名稱, 輸入通道 Cin, 輸入高 H, 輸入寬 W, 輸出通道 Cout, kernel K, stride s, padding p)
LAYERS = [
    ('stem',  3, 32, 32, 16, 3, 2, 1),
    ('l1_a', 16, 16, 16, 16, 3, 2, 1),
    ('l1_b', 16,  8,  8, 16, 3, 1, 1),
    ('l1_sc', 16, 16, 16, 16, 1, 2, 0),
    ('l2_a', 16,  8,  8, 32, 3, 2, 1),
    ('l2_b', 32,  4,  4, 32, 3, 1, 1),
    ('l2_sc', 16,  8,  8, 32, 1, 2, 0),
]
FC = (32, 10)          # 32 個輸入、10 個輸出

N_IMG = 64             # 評分用的圖片數
GOAL_NS = 1_000_000    # Goal：64 張 < 10^6 ns


def out_size(H, K, s, p):
    """conv 輸出的邊長：floor((H + 2p - K) / s) + 1"""
    return (H + 2 * p - K) // s + 1


def layer_info(Cin, H, W, Cout, K, s, p):
    Ho, Wo = out_size(H, K, s, p), out_size(W, K, s, p)
    n_out = Cout * Ho * Wo           # 輸出點的數量
    mac_per_out = Cin * K * K        # 每個輸出點要做的乘加
    return Ho, Wo, n_out, mac_per_out, n_out * mac_per_out, Cout * Cin * K * K


if __name__ == '__main__':
    # ============================================================ Part A
    print('Part A: 每一層的 MAC 數 = 輸出點數 × 每個輸出點的 MAC')
    print(f"{'層':6s} {'輸入':>11s} {'輸出':>11s} {'輸出點數':>8s} {'每點MAC':>8s} {'MAC':>9s} {'權重數':>7s}")
    total_mac = total_w = 0
    for name, Cin, H, W, Cout, K, s, p in LAYERS:
        Ho, Wo, n_out, mpo, mac, nw = layer_info(Cin, H, W, Cout, K, s, p)
        total_mac += mac; total_w += nw
        print(f'{name:6s} {Cin:3d}x{H:2d}x{W:2d}   {Cout:3d}x{Ho:2d}x{Wo:2d} {n_out:8d}'
              f' {Cin:2d}x{K}x{K}={mpo:3d} {mac:9,d} {nw:7,d}')
    fc_mac = FC[0] * FC[1]
    total_mac += fc_mac; total_w += fc_mac
    print(f"{'fc':6s} {FC[0]:>11d} {FC[1]:>11d} {FC[1]:8d} {FC[0]:8d} {fc_mac:9,d} {fc_mac:7,d}")
    print(f'每張圖合計 MAC = {total_mac:,}     權重數 = {total_w:,}')
    print(f'{N_IMG} 張合計 MAC = {total_mac * N_IMG:,}')

    # ============================================================ Part B
    print('\nPart B: 時間預算 → 每個 cycle 至少要做幾個 MAC（假設 100% 利用率、沒有其他額外時間）')
    print(f'每張圖可用時間 = {GOAL_NS:,} ns / {N_IMG} = {GOAL_NS / N_IMG:,.1f} ns')
    for f_mhz in (1000, 500, 400, 250):
        T = 1000 / f_mhz
        cycles_total = GOAL_NS / T
        print(f'  {f_mhz:4d} MHz (週期 {T:.1f} ns): 總 cycle {cycles_total:>9,.0f}  '
              f'每張 {cycles_total / N_IMG:6,.0f} cycle  → 至少 {total_mac * N_IMG / cycles_total:6.1f} MAC/cycle')

    # ============================================================ Part C
    print('\nPart C: 示範——128 個 PE 排成「16 個輸出通道 × 8 個輸入通道」，每個 cycle 做 16×8 個 MAC')
    print('        每層實際能用到的 PE = min(Cout,16)/16 × min(Cin,8)/8（只是示範排法，不是最終決定）')
    PO, PI = 16, 8
    total_cyc = 0
    for name, Cin, H, W, Cout, K, s, p in LAYERS:
        Ho, Wo, n_out, mpo, mac, _ = layer_info(Cin, H, W, Cout, K, s, p)
        # 迴圈次數：輸出通道要分幾組 × 輸入通道要分幾組 × kernel 位置 × 輸出位置
        cyc = math.ceil(Cout / PO) * math.ceil(Cin / PI) * K * K * Ho * Wo
        util = mac / (cyc * PO * PI)
        total_cyc += cyc
        print(f'  {name:6s} cycle {cyc:6,d}  利用率 {util:6.1%}')
    # fc 也用同一個陣列：10 個輸出通道佔 16 列中的 10 列，32 個輸入分 4 批
    fc_cyc = math.ceil(FC[1] / PO) * math.ceil(FC[0] / PI)
    total_cyc += fc_cyc
    print(f'  fc     cycle  {fc_cyc:5,d}  利用率 {fc_mac / (fc_cyc * PO * PI):6.1%}')
    avg_util = total_mac / (total_cyc * PO * PI)
    print(f'  每張圖 {total_cyc:,} cycle，平均利用率 {avg_util:.1%}')
    for f_mhz in (500, 1000):
        T = 1000 / f_mhz
        print(f'  @ {f_mhz} MHz: 64 張純運算時間 = {total_cyc * N_IMG * T:,.0f} ns'
              f'  ({total_cyc * N_IMG * T / GOAL_NS:.0%} of goal)')
