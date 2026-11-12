"""
hw_model.py -- 照「硬體規格」運作的 Python 模型（SPEC.md 第 5、6 章的可執行版本）

目的：在寫 RTL 之前，先驗證 SPEC 本身是對的。
  * 只讀 model.dat 和 image_*.bin，不碰 quant.py 的任何東西
  * 用 SPEC 的資料格式、位址公式、批次順序、累加器 bank、requant 排出方式來計算
  * 結果必須跟 GOLDEN（quant.py 產生）每個 bit 都一樣
  * 順便檢查 SPEC 宣稱的性質（例如「一個 (i, ky) 需要的輸入一定在同一個 word」）

RTL 寫好之後，這個檔案也是 debug 的對照：每一個 SRAM word、每一批的累加值都可以在這裡印出來。

執行：python3 hw_model.py [圖片張數，預設 64]
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(HERE, '..', '00_TESTBED', 'PATTERN')

BEAT = 32            # DRAM word = 32 bytes
FMW = 64             # feature map word = 64 bytes (512 bit)
WSEC_ADDR, PSEC_ADDR = 0x10000, 0x15000
WSEC_BEATS, PSEC_BEATS = 614, 43


# =====================================================================
# 0. 小工具：有號數、位元寬度檢查
# =====================================================================
def s8(b):
    return b - 256 if b >= 128 else b


def le_int(bs, signed):
    v = int.from_bytes(bytes(bs), 'little')
    nb = 8 * len(bs)
    return v - (1 << nb) if signed and v >= (1 << (nb - 1)) else v


def check_signed(v, bits, what):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    assert lo <= np.min(v) and np.max(v) <= hi, f'{what} 超過 {bits}-bit 有號範圍'


# =====================================================================
# 1. DRAM 與 model.dat（SPEC 5.1）
# =====================================================================
class DRAM:
    """512 KB，word = 32 bytes；行為同 $readmemh"""
    def __init__(self):
        self.mem = np.zeros((16384, BEAT), np.uint8)

    def readmemh(self, path):
        addr = 0
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            if line.startswith('@'):
                addr = int(line[1:], 16)
                continue
            v = int(line, 16)
            self.mem[addr] = [(v >> (8 * k)) & 0xFF for k in range(BEAT)]   # byte k = bit[8k+7:8k]
            addr += 1

    def load_image(self, n, img_id):
        """testbench 把第 img_id 張圖放到第 n 個位置（0x1000·n）"""
        data = np.fromfile(os.path.join(PATTERN, 'IMAGES', f'image_{img_id}.bin'), np.uint8)
        base = 0x1000 * n // BEAT
        self.mem[base:base + 96] = data.reshape(96, BEAT)

    def read_burst(self, byte_addr, beats):
        """AXI INCR burst；檢查不跨 4 KB 邊界"""
        assert (byte_addr % 4096) + beats * BEAT <= 4096, f'burst 跨 4KB 邊界 @0x{byte_addr:x}'
        w = byte_addr // BEAT
        return [self.mem[w + k].copy() for k in range(beats)]


# =====================================================================
# 2. 晶片內記憶體（SPEC 6.8）
# =====================================================================
class Chip:
    def __init__(self):
        self.WBUF = np.zeros((1227, 16), np.uint8)
        self.PBUF = np.zeros((30, BEAT), np.uint8)
        self.FCMEM = np.zeros((10, BEAT), np.uint8)
        self.IMG = np.zeros((48, FMW), np.uint8)
        self.FM = {'A': np.zeros((64, FMW), np.uint8),
                   'B': np.zeros((16, FMW), np.uint8),
                   'C': np.zeros((16, FMW), np.uint8)}
        self.s = [0] * 7
        self.conf_m = self.conf_s = 0
        self.bfc = [0] * 10
        self.pool = np.zeros(32, np.int64)
        self.results = [0] * 16

    # ---------------------------------------------------------- loader（SPEC 6.6 / 6.7）
    def load_model(self, dram):
        bursts = [(0x10000, 128), (0x11000, 128), (0x12000, 128), (0x13000, 128), (0x14000, 102)]
        beats = [b for a, n in bursts for b in dram.read_burst(a, n)]
        assert len(beats) == WSEC_BEATS
        for j, b in enumerate(beats):                      # 一個 beat → 兩個 WBUF word
            for half in range(2):
                w = 2 * j + half
                if w < 1227:
                    self.WBUF[w] = b[16 * half:16 * half + 16]
        p = dram.read_burst(PSEC_ADDR, PSEC_BEATS)
        self.s = [int(x) for x in p[0][0:7]]
        self.conf_m = le_int(p[0][8:10], False)
        self.conf_s = int(p[0][10])
        for k in range(30):
            self.PBUF[k] = p[1 + k]
        for k in range(10):
            self.FCMEM[k] = p[31 + k]
        fb = np.concatenate([p[41], p[42]])
        self.bfc = [le_int(fb[4 * k:4 * k + 4], True) for k in range(10)]

    def load_image(self, dram, n):
        for b, beat in enumerate(dram.read_burst(0x1000 * n, 96)):
            c, y = b // 32, b % 32
            word, half = c * 16 + y // 2, y % 2             # SPEC 6.7
            self.IMG[word, 32 * half:32 * half + 32] = beat


# =====================================================================
# 3. 位址公式（SPEC 6.9）
# =====================================================================
def fm_loc(C, H, W, c, y, x):
    """tensor (C,H,W) 的元素 [c][y][x] 在 feature map buffer 的 (word, byte)"""
    plane = H * W
    if plane >= FMW:
        rpw, wpc = FMW // W, plane // FMW
        return c * wpc + y // rpw, (y % rpw) * W + x
    cpw = FMW // plane
    return c // cpw, (c % cpw) * plane + y * W + x


LAYERS = {   # name: (Cin, H, W, Cout, K, stride, pad, wbase, s_index, 輸入buffer, 輸出buffer)
    'stem':  (3, 32, 32, 16, 3, 2, 1, 0,   0, 'IMG', 'A'),
    'l1_a':  (16, 16, 16, 16, 3, 2, 1, 27,  1, 'A', 'B'),
    'l1_sc': (16, 16, 16, 16, 1, 2, 0, 171, 2, 'A', 'C'),
    'l1_b':  (16, 8, 8, 16, 3, 1, 1, 187,  3, 'B', 'C'),
    'l2_a':  (16, 8, 8, 32, 3, 2, 1, 331,  4, 'C', 'B'),
    'l2_sc': (16, 8, 8, 32, 1, 2, 0, 619,  5, 'C', None),
    'l2_b':  (32, 4, 4, 32, 3, 1, 1, 651,  6, 'B', None),
}
PBLOCK = {('stem', 0): 0, ('l1_a', 0): 1, ('l1_sc', 0): 2, ('l1_b', 0): 3, ('l2_a', 0): 4,
          ('l2_a', 1): 5, ('l2_sc', 0): 6, ('l2_sc', 1): 7, ('l2_b', 0): 8, ('l2_b', 1): 9}


def out_hw(name):
    Cin, H, W, Cout, K, S, Pd = LAYERS[name][:7]
    return (H + 2 * Pd - K) // S + 1, (W + 2 * Pd - K) // S + 1


# =====================================================================
# 4. 運算核心（SPEC 6.10 – 6.16）
# =====================================================================
class Core:
    def __init__(self, chip):
        self.c = chip
        self.acc = np.zeros((2, 16, 8), np.int64)      # bank, 列(輸出通道), 行(像素)
        self.cycles = 0

    def buf(self, name):
        return self.c.IMG if name == 'IMG' else self.c.FM[name]

    def params(self, layer, og):
        """SPEC 6.13：從 PBUF 讀一個參數區塊 → 16 個 bias、16 個 m"""
        k = PBLOCK[(layer, og)] * 3
        b = np.concatenate([self.c.PBUF[k], self.c.PBUF[k + 1]])
        bias = [le_int(b[4 * r:4 * r + 4], True) for r in range(16)]
        m = [le_int(self.c.PBUF[k + 2][2 * r:2 * r + 2], False) for r in range(16)]
        return np.array(bias, np.int64), np.array(m, np.int64)

    def run_batch(self, layer, og, g, bank):
        """一批：16 個輸出通道 × 8 個像素，Cin×K×K 個 cycle，結果累加在 acc[bank]"""
        Cin, H, W, Cout, K, S, Pd, wbase, _, src, _ = LAYERS[layer]
        Ho, Wo = out_hw(layer)
        bias, _ = self.params(layer, og)
        mem = self.buf(src)
        oys = [(8 * g + p) // Wo for p in range(8)]
        oxs = [(8 * g + p) % Wo for p in range(8)]
        first = True
        for i in range(Cin):
            for ky in range(K):
                # ---- 每個 (i, ky) 讀一次 FM word（SPEC 6.11）
                iy0 = min(max(oys[0] * S + ky - Pd, 0), H - 1)
                word, _ = fm_loc(Cin, H, W, i, iy0, 0)
                window = mem[word]
                for kx in range(K):
                    # ---- 權重：WBUF 位址公式
                    waddr = wbase + og * (Cin * K * K) + i * K * K + ky * K + kx
                    wts = np.array([s8(int(x)) for x in self.c.WBUF[waddr]], np.int64)
                    # ---- 輸入：in_sel
                    ops = np.zeros(8, np.int64)
                    for p in range(8):
                        iy, ix = oys[p] * S + ky - Pd, oxs[p] * S + kx - Pd
                        if 0 <= iy < H and 0 <= ix < W:
                            w2, byte = fm_loc(Cin, H, W, i, iy, ix)
                            assert w2 == word, f'{layer}: 需要的輸入不在同一個 word（SPEC 6.9 性質不成立）'
                            v = int(window[byte])
                            ops[p] = v - 128 if layer == 'stem' else v
                    assert ops.min() >= -256 and ops.max() <= 255          # 9-bit 有號
                    # ---- PE：第一個 cycle 載入 bias
                    prod = np.outer(wts, ops)
                    if first:
                        self.acc[bank] = bias[:, None] + prod
                        first = False
                    else:
                        self.acc[bank] += prod
                    check_signed(self.acc[bank], 20, f'{layer} 累加器')
                    self.cycles += 1

    def drain(self, layer, og, g, bank_main, residual_sc=None, pool=False):
        """SPEC 6.14：一列一個 cycle，8 條通道；寫回或加進 pool"""
        Cin, H, W, Cout, K, S, Pd, wbase, si, src, dst = LAYERS[layer]
        Ho, Wo = out_hw(layer)
        s = self.c.s[si]
        _, m = self.params(layer, og)
        if residual_sc:
            _, m_sc = self.params(residual_sc, og)
            assert self.c.s[LAYERS[residual_sc][8]] == s, '殘差兩條路徑的 s 不同'
        for r in range(16):
            v = self.acc[bank_main][r] * m[r]
            if residual_sc:
                v = v + self.acc[1][r] * m_sc[r]
            v = v + (1 << (s - 1))
            check_signed(v, 38, 'requant 中間值')
            y = np.clip(v >> s, 0, 255)
            c = og * 16 + r
            if pool:
                self.c.pool[c] += int(y.sum())
                continue
            oy0, ox0 = (8 * g) // Wo, (8 * g) % Wo
            word, byte0 = fm_loc(Cout, Ho, Wo, c, oy0, ox0)
            assert byte0 % 8 == 0, '寫回位置沒有對齊 8 byte'
            for p in range(8):
                n = 8 * g + p
                assert fm_loc(Cout, Ho, Wo, c, n // Wo, n % Wo) == (word, byte0 + p), '寫回的 8 個 byte 不連續'
            self.c.FM[dst][word, byte0:byte0 + 8] = y.astype(np.uint8)

    def plain_layer(self, layer):
        Cout = LAYERS[layer][3]
        Ho, Wo = out_hw(layer)
        bank = 0
        for og in range(Cout // 16):
            for g in range(Ho * Wo // 8):
                self.run_batch(layer, og, g, bank)
                self.drain(layer, og, g, bank)
                bank ^= 1                                  # ping-pong

    def residual_layer(self, sc, main, pool):
        Cout = LAYERS[main][3]
        Ho, Wo = out_hw(main)
        assert out_hw(sc) == (Ho, Wo)
        for og in range(Cout // 16):
            for g in range(Ho * Wo // 8):
                self.run_batch(sc, og, g, 1)               # shortcut → bank1
                self.run_batch(main, og, g, 0)             # 主路徑 → bank0
                self.drain(main, og, g, 0, residual_sc=sc, pool=pool)

    def fc(self, n):
        """SPEC 6.16"""
        pool = self.c.pool.copy()
        self.c.pool[:] = 0
        assert pool.max() < 4096, 'pool 超過 12-bit'
        best, best_k = None, 0
        logits = []
        for k in range(10):
            acc = self.c.bfc[k]
            for j in range(32):
                acc += int(pool[j]) * s8(int(self.c.FCMEM[k][j]))
            check_signed(acc, 32, 'fc 累加器')
            logits.append(acc)
            if best is None or acc > best:                 # 平手保留較小的 k
                best, best_k = acc, k
        conf = (best * self.c.conf_m + (1 << (self.c.conf_s - 1))) >> self.c.conf_s
        check_signed(conf, 28, 'confidence')
        self.c.results[n] = (best_k << 28) | (conf & 0x0FFFFFFF)
        return pool, logits

    def run_image(self, n):
        self.plain_layer('stem')
        snap = {'stem_out': self.c.FM['A'].copy()}
        self.plain_layer('l1_a')
        snap['l1_a_out'] = self.c.FM['B'].copy()
        self.residual_layer('l1_sc', 'l1_b', pool=False)
        snap['l1_out'] = self.c.FM['C'].copy()
        self.plain_layer('l2_a')
        snap['l2_a_out'] = self.c.FM['B'].copy()
        self.residual_layer('l2_sc', 'l2_b', pool=True)
        snap['pool'], snap['fc'] = self.fc(n)
        return snap


# =====================================================================
# 5. 跟 GOLDEN 比對
# =====================================================================
def read_hex(path, bits, signed):
    vals = [int(l, 16) for l in open(path) if l.strip()]
    if signed:
        vals = [v - (1 << bits) if v >= (1 << (bits - 1)) else v for v in vals]
    return np.array(vals, np.int64)


def fm_to_tensor(buf, C, H, W):
    t = np.zeros((C, H, W), np.int64)
    for c in range(C):
        for y in range(H):
            for x in range(W):
                word, byte = fm_loc(C, H, W, c, y, x)
                t[c, y, x] = buf[word, byte]
    return t


TENSORS = {'stem_out': (16, 16, 16), 'l1_a_out': (16, 8, 8), 'l1_out': (16, 8, 8), 'l2_a_out': (32, 4, 4)}

if __name__ == '__main__':
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    gold_res = [tuple(map(int, l.split())) for l in open(os.path.join(PATTERN, 'golden_results.txt'))]

    dram = DRAM()
    dram.readmemh(os.path.join(PATTERN, 'model.dat'))
    chip = Chip()
    chip.load_model(dram)
    core = Core(chip)

    n_bad = 0
    for start in range(0, n_img, 16):                      # 一批 16 張，跟 testbench 一樣
        N = min(16, n_img - start)
        for k in range(N):
            dram.load_image(k, start + k)
        for k in range(N):
            img = start + k
            chip.load_image(dram, k)
            snap = core.run_image(k)
            gdir = os.path.join(PATTERN, 'GOLDEN', f'image_{img}')
            ok = True
            if os.path.isdir(gdir):                        # 逐層比對（GOLDEN 只有前 64 張）
                for name, shape in TENSORS.items():
                    got = fm_to_tensor(snap[name], *shape)
                    exp = read_hex(os.path.join(gdir, name + '.hex'), 8, False).reshape(shape)
                    if not np.array_equal(got, exp):
                        print(f'image {img}: {name} 不一致，第一個錯在 {np.argwhere(got != exp)[0]}')
                        ok = False
                if not np.array_equal(snap['pool'], read_hex(os.path.join(gdir, 'pool_sum.hex'), 12, False)):
                    print(f'image {img}: pool 不一致'); ok = False
                if not np.array_equal(snap['fc'], read_hex(os.path.join(gdir, 'fc_logit.hex'), 32, True)):
                    print(f'image {img}: fc logit 不一致'); ok = False
            r = chip.results[k]
            cat, conf = r >> 28, r & 0x0FFFFFFF
            conf = conf - (1 << 28) if conf >= (1 << 27) else conf
            if (cat, conf) != gold_res[img]:
                print(f'image {img}: 結果 ({cat}, {conf}) ≠ golden {gold_res[img]}'); ok = False
            n_bad += not ok
    print(f'{n_img} 張圖：{n_img - n_bad} 張完全一致，{n_bad} 張有差異')
    print(f'運算 cycle（不含排出等待、管線）：每張 {core.cycles // n_img}')
