"""
dataflow.py -- 比較三種 dataflow 的讀寫量（全部都是 128 個乘法器，只有「排法」和「迴圈順序」不同）

  A. OS-通道   : 陣列 = 16 輸出通道 × 8 輸入通道。迴圈：輸出位置在外層，(kernel 格, 輸入通道批) 在內層
                 → 累加器固定在陣列裡（output stationary），權重每個 cycle 都換
  B. WS-通道   : 陣列 = 16 輸出通道 × 8 輸入通道。迴圈：(kernel 格, 輸入通道批) 在外層，輸出位置在內層
                 → 權重固定在陣列裡（weight stationary），未完成的累加值要存進 psum buffer
  C. OS-像素   : 陣列 = 16 輸出通道 × 8 個輸出像素。迴圈：輸出像素群在外層，(輸入通道, kernel 格) 在內層
                 → 累加器固定在陣列裡；同一個權重送給同一列的 8 個像素共用

統計單位：
  權重、輸入、輸出 = 8-bit 數值的個數（= bytes）
  psum            = 20-bit 未完成累加值的讀/寫次數
只統計陣列跟晶片內 buffer 之間的搬運（DRAM 流量是第 3 輪的事）。
"""
import math
from arch_model import LAYERS, FC, layer_info

ACC_BITS = 20


def flow_A(Ci, Co, K, N):
    """OS-通道：16 列 = 輸出通道，8 行 = 輸入通道"""
    cyc = math.ceil(Co / 16) * math.ceil(Ci / 8) * K * K * N
    used_i, used_o = min(Ci, 8), min(Co, 16)
    return dict(cyc=cyc,
                w=cyc * used_o * used_i,       # 每個 cycle 每個乘法器都要一個新權重
                x=cyc * used_i,                # 每 cycle 8 個輸入，往下 broadcast 給 16 列
                ps_r=0, ps_w=0,                # 累加器一直在陣列裡，做完才輸出
                acc_regs=16, psum_buf=0)


def flow_B(Ci, Co, K, N):
    """WS-通道：權重載入後固定，掃完所有輸出位置才換"""
    cyc = math.ceil(Co / 16) * math.ceil(Ci / 8) * K * K * N
    used_i, used_o = min(Ci, 8), min(Co, 16)
    n_wblock = math.ceil(Co / 16) * math.ceil(Ci / 8) * K * K   # 換權重的次數
    # 每個 cycle：讀出這個位置 16 個通道的 psum、加上去、寫回。
    # 第一批不用讀（從 0 開始），最後一批直接 requantize 不用寫回。
    first_last = math.ceil(Co / 16) * N * used_o
    return dict(cyc=cyc,
                w=n_wblock * used_o * used_i,   # 每個權重只讀一次
                x=cyc * used_i,
                ps_r=cyc * used_o - first_last, ps_w=cyc * used_o - first_last,
                acc_regs=16, psum_buf=16 * N)   # 一批輸出通道 × 所有輸出位置


def flow_C(Ci, Co, K, N):
    """OS-像素：16 列 = 輸出通道，8 行 = 8 個輸出像素"""
    px = min(N, 8)
    cyc = math.ceil(Co / 16) * math.ceil(N / 8) * Ci * K * K
    used_o = min(Co, 16)
    return dict(cyc=cyc,
                w=cyc * used_o,                 # 每列一個權重，橫向送給 8 個像素共用
                x=cyc * px,                     # 8 個不同像素（同一通道、同一 kernel 格）
                ps_r=0, ps_w=0,
                acc_regs=128, psum_buf=0)


FLOWS = {'A OS-通道': flow_A, 'B WS-通道': flow_B, 'C OS-像素': flow_C}

if __name__ == '__main__':
    layers = [(n, Cin, Cout, K, *layer_info(Cin, H, W, Cout, K, s, p)) for n, Cin, H, W, Cout, K, s, p in LAYERS]
    layers.append(('fc', FC[0], FC[1], 1, 1, 1, FC[1], FC[0], FC[0] * FC[1], FC[0] * FC[1]))

    totals = {}
    for fname, f in FLOWS.items():
        print(f'\n=== {fname} ===')
        print(f"{'層':6s} {'cycle':>6s} {'利用率':>6s} {'讀權重':>8s} {'讀輸入':>7s} {'psum讀':>7s} {'psum寫':>7s} {'寫輸出':>6s} {'psum buffer':>11s}")
        T = dict(cyc=0, w=0, x=0, ps_r=0, ps_w=0, out=0, mac=0, psum_buf=0, acc_regs=0)
        for n, Cin, Cout, K, Ho, Wo, n_out, mpo, mac, nw in layers:
            r = f(Cin, Cout, K, Ho * Wo)
            util = mac / (r['cyc'] * 128)
            print(f"{n:6s} {r['cyc']:6,d} {util:6.0%} {r['w']:8,d} {r['x']:7,d} {r['ps_r']:7,d} {r['ps_w']:7,d}"
                  f" {n_out:6,d} {r['psum_buf']:7,d} 個")
            for k in ('cyc', 'w', 'x', 'ps_r', 'ps_w'):
                T[k] += r[k]
            T['out'] += n_out; T['mac'] += mac
            T['psum_buf'] = max(T['psum_buf'], r['psum_buf']); T['acc_regs'] = r['acc_regs']
        T['util'] = T['mac'] / (T['cyc'] * 128)
        totals[fname] = T
        print(f"{'合計':6s} {T['cyc']:6,d} {T['util']:6.0%} {T['w']:8,d} {T['x']:7,d} {T['ps_r']:7,d} {T['ps_w']:7,d} {T['out']:6,d}")

    print('\n=== 每張圖總結 ===')
    print(f"{'dataflow':10s} {'cycle':>6s} {'64張@500MHz':>11s} {'利用率':>6s} {'每cycle平均讀取(bytes)':>22s} {'buffer總讀寫(KB)':>15s} {'累加器':>6s} {'psum buffer':>12s}")
    for fname, T in totals.items():
        ps_bytes = (T['ps_r'] + T['ps_w']) * ACC_BITS / 8
        per_cyc = (T['w'] + T['x'] + T['ps_r'] * ACC_BITS / 8) / T['cyc']
        total_kb = (T['w'] + T['x'] + T['out'] + ps_bytes) / 1024
        t64 = T['cyc'] * 64 * 2
        print(f"{fname:10s} {T['cyc']:6,d} {t64:9,d}ns {T['util']:6.0%} {per_cyc:18.1f}     {total_kb:12.0f}   {T['acc_regs']:6d}"
              f"   {T['psum_buf']:5,d}×{ACC_BITS}b")
