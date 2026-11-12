"""
compare.py -- 用「能量」和「面積」的代理指標比較 A / B / C 三種 dataflow

能量：每一次存取乘上它的相對耗能（資料夾裡 Tutorial-on-DNN-05 的「Normalized Energy Cost」表，以一次 MAC = 1×）
        MAC 1×   暫存器(RF, PE 內) 1×   晶片內 buffer 6×   DRAM 200×
      tutorial 的數字來自數百 KB 的大 buffer；我們的 buffer 只有幾 KB，倍率應該比較小，
      所以同時算 buffer = 3× 的情況，看結論會不會改變（敏感度分析）。
      20-bit psum 一次讀寫算 2.5 次 8-bit 存取（照位元數比例）。
      DRAM 流量三種 dataflow 一樣（第 3 輪再算），這裡不列入。

面積：只比「三者不同的部分」（權重、feature map 的儲存三者一樣）
      累加器 / psum 的儲存位元數、buffer 讀取埠寬度
"""
from dataflow import FLOWS, flow_A
from arch_model import LAYERS, FC, layer_info

ACC_BITS = 20


def per_image(f):
    layers = [(Cin, Cout, K, *layer_info(Cin, H, W, Cout, K, s, p)) for _, Cin, H, W, Cout, K, s, p in LAYERS]
    layers.append((FC[0], FC[1], 1, 1, 1, FC[1], FC[0], FC[0] * FC[1], FC[0] * FC[1]))
    T = dict(cyc=0, w=0, x=0, ps=0, out=0, mac=0, rf=0)
    for Cin, Cout, K, Ho, Wo, n_out, mpo, mac, nw in layers:
        r = f(Cin, Cout, K, Ho * Wo)
        T['cyc'] += r['cyc']; T['w'] += r['w']; T['x'] += r['x']
        T['ps'] += r['ps_r'] + r['ps_w']; T['out'] += n_out; T['mac'] += mac
        # PE 內暫存器的存取：
        #  A：每列每 cycle 更新 1 次累加器（8 個乘積先經過加法樹）
        #  B：每次 MAC 從 PE 內讀 1 次固定住的權重
        #  C：每次 MAC 更新 1 次自己的累加器
        if f is flow_A:
            T['rf'] += r['cyc'] * min(Cout, 16)
        else:
            T['rf'] += mac
    return T


AREA = {   # 三者不同的部分
    'A OS-通道': dict(acc_bits=16 * ACC_BITS, psum_bits=0, w_port=128 * 8, note='16 棵 8 輸入加法樹'),
    'B WS-通道': dict(acc_bits=16 * ACC_BITS, psum_bits=4096 * ACC_BITS, w_port=128 * 8, note='16 棵加法樹 + 128 個權重暫存器'),
    'C OS-像素': dict(acc_bits=128 * ACC_BITS, psum_bits=0, w_port=16 * 8, note='128 個累加器'),
}

if __name__ == '__main__':
    for buf_cost in (6, 3):
        print(f'\n=== 能量代理（每張圖，單位 = 一次 MAC 的能量），buffer = {buf_cost}× ===')
        print(f"{'':10s} {'MAC':>9s} {'RF':>9s} {'buffer':>10s} {'合計':>10s} {'相對 C':>7s}")
        res = {}
        for name, f in FLOWS.items():
            T = per_image(f)
            buf = (T['w'] + T['x'] + T['out'] + T['ps'] * ACC_BITS / 8) * buf_cost
            res[name] = (T['mac'], T['rf'], buf, T['mac'] + T['rf'] + buf)
        ref = res['C OS-像素'][3]
        for name, (m, rf, buf, tot) in res.items():
            print(f'{name:10s} {m:9,.0f} {rf:9,.0f} {buf:10,.0f} {tot:10,.0f} {tot / ref:6.2f}×')

    print('\n=== 面積：三者不同的部分 ===')
    for name, a in AREA.items():
        print(f"{name:10s} 累加器 {a['acc_bits']:5,d} bit   psum buffer {a['psum_bits']:6,d} bit"
              f"   權重讀取埠 {a['w_port']:5d} bit   ({a['note']})")

    print('\n=== 讀取能否被 ping-pong 藏起來：C 的每張圖 5,120 cycle 裡，各種搬運要幾個 cycle ===')
    BEAT = 32   # 一個 256-bit beat = 32 bytes；CDC 之後 dut 端每 cycle 收 1 個 beat
    items = [('一張圖片 3,072 B', 3072),
             ('全部權重 19,952 B', 19952),
             ('最大一層權重 l2_b 9,216 B', 9216)]
    for n, b in items:
        print(f'  {n:28s} → {-(-b // BEAT):4d} cycle')
    print('  對照：l2_b 這層的運算 1,152 cycle、一張圖的運算 5,120 cycle')
