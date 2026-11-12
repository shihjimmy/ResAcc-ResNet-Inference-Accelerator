"""
gen_model_dat.py -- 把整數模型參數打包成 DRAM 的 model.dat（格式見 SPEC.md 第 5 章）

輸入：resnet6.pth（透過 quant.build_quant_model 重建整數模型，與 quant_params.npz 相同）
輸出：../00_TESTBED/PATTERN/model.dat         testbench 用 $readmemh 載入 DRAM
      ../00_TESTBED/PATTERN/golden_results.txt 每張圖一行：category confidence（confidence = 實數 logit × 2^20）

model.dat 的規則：
  * DRAM 一個 word = 256 bit = 32 bytes，word 編號 = byte 位址 / 32
  * byte k 放在 bit [8k+7:8k]；hex 字串最左邊是 byte 31
  * 用 @<word 編號> 指定起點
"""
import os
import numpy as np

from common import HERE, N_IMAGES, load_images
from quant import build_quant_model, int_forward
from common import load_pth

PATTERN = os.path.join(HERE, '..', '00_TESTBED', 'PATTERN')

# ---------------------------------------------------------------- 常數（SPEC 5.2 / 5.4 / 5.5）
WSEC_ADDR = 0x10000          # conv 權重的 byte 位址
PSEC_ADDR = 0x15000          # 參數 / FC 的 byte 位址
BEAT = 32                    # bytes per DRAM word

LAYER_ORDER = ['stem', 'l1_a', 'l1_sc', 'l1_b', 'l2_a', 'l2_sc', 'l2_b']   # 硬體執行順序
PARAM_BLOCKS = [('stem', 0), ('l1_a', 0), ('l1_sc', 0), ('l1_b', 0),
                ('l2_a', 0), ('l2_a', 1), ('l2_sc', 0), ('l2_sc', 1),
                ('l2_b', 0), ('l2_b', 1)]                                   # SPEC 5.5 的區塊編號


# ---------------------------------------------------------------- 打包工具
def u8(values):
    """任意整數陣列 → uint8 bytes（有號數取 2 補數的低 8 bit）"""
    return (np.asarray(values, dtype=np.int64) & 0xFF).astype(np.uint8)


def le_bytes(values, nbytes):
    """整數陣列 → little-endian bytes，每個數 nbytes 個 byte"""
    out = []
    for v in np.asarray(values, dtype=np.int64).ravel():
        v = int(v) & ((1 << (8 * nbytes)) - 1)
        out += [(v >> (8 * k)) & 0xFF for k in range(nbytes)]
    return np.array(out, dtype=np.uint8)


def to_beats(byte_array):
    """bytes → 一串 32-byte 的 beat（最後補 0）"""
    b = np.asarray(byte_array, dtype=np.uint8)
    pad = (-len(b)) % BEAT
    b = np.concatenate([b, np.zeros(pad, np.uint8)])
    return b.reshape(-1, BEAT)


def beat_hex(beat):
    """一個 beat（32 bytes）→ 64 個 hex 字元，byte 31 在最左邊"""
    return ''.join(f'{int(x):02x}' for x in beat[::-1])


# ---------------------------------------------------------------- 兩個區段
def build_wsec(P):
    """conv 權重：每個 WBUF word = 16 個 int8，byte r = 第 og*16+r 個輸出通道（SPEC 5.4）"""
    words, wbase = [], {}
    for layer in LAYER_ORDER:
        w = P[layer + '_w_q']                       # (Cout, Cin, K, K)
        Cout, Cin, K, _ = w.shape
        wbase[layer] = len(words)
        for og in range(Cout // 16):
            for i in range(Cin):
                for ky in range(K):
                    for kx in range(K):
                        words.append(u8(w[og * 16:og * 16 + 16, i, ky, kx]))
    flat = np.concatenate(words)                    # word 2j 在低位，2j+1 在高位 → 直接串接即可
    return to_beats(flat), wbase, len(words)


def build_psec(P):
    """參數區：header、10 個參數區塊、FC 權重、FC bias（SPEC 5.5）"""
    beats = []
    # beat 0：header
    hdr = np.zeros(BEAT, np.uint8)
    hdr[0:7] = [int(P[l + '_s']) for l in LAYER_ORDER]
    hdr[8:10] = le_bytes([int(P['conf_m'])], 2)
    hdr[10] = int(P['conf_s'])
    beats.append(hdr)
    # beat 1..30：參數區塊，每塊 3 beats
    for layer, og in PARAM_BLOCKS:
        b = P[layer + '_b_q'][og * 16:og * 16 + 16]
        m = P[layer + '_m'][og * 16:og * 16 + 16]
        assert np.all(np.abs(b) < 2 ** 17), 'bias 超過 18-bit'
        assert np.all((m >= 0) & (m < 2 ** 16)), 'm 超過 16-bit'
        beats += list(to_beats(le_bytes(b[:8], 4)))
        beats += list(to_beats(le_bytes(b[8:], 4)))
        beats += list(to_beats(le_bytes(m, 2)))
    # beat 31..40：FC 權重，一個類別一個 beat
    for k in range(10):
        beats.append(u8(P['fc_w_q'][k]))
    # beat 41..42：FC bias（int32 × 10）
    beats += list(to_beats(le_bytes(P['fc_b_q'], 4)))
    return np.stack(beats)


def write_model_dat(P, path):
    wsec, wbase, nwords = build_wsec(P)
    psec = build_psec(P)
    assert WSEC_ADDR + len(wsec) * BEAT <= PSEC_ADDR, 'WSEC 跟 PSEC 重疊'
    with open(path, 'w') as f:
        f.write(f'@{WSEC_ADDR // BEAT:x}\n')
        f.writelines(beat_hex(b) + '\n' for b in wsec)
        f.write(f'@{PSEC_ADDR // BEAT:x}\n')
        f.writelines(beat_hex(b) + '\n' for b in psec)
    return wbase, nwords, len(wsec), len(psec)


def save_params_npz(qm):
    """跟 quant.save_params 相同；這裡直接從 QModel 取，確保 model.dat 與 golden 出自同一個模型"""
    d = {'fc_w_q': qm.fc_w_q, 'fc_b_q': qm.fc_b_q, 'conf_m': qm.conf_m, 'conf_s': qm.conf_s}
    for name, c in qm.conv.items():
        d[name + '_w_q'], d[name + '_b_q'], d[name + '_m'], d[name + '_s'] = c.w_q, c.b_q, c.m, c.s
    return d


if __name__ == '__main__':
    qm = build_quant_model(load_pth())          # 重新建模型（保證參數與 golden 一致）
    P = save_params_npz(qm)

    wbase, nwords, nw_beats, np_beats = write_model_dat(P, os.path.join(PATTERN, 'model.dat'))
    print('model.dat 寫好了')
    print(f'  WSEC @0x{WSEC_ADDR:05x}: {nwords} 個 WBUF word = {nw_beats} beats')
    print('  各層 wbase:', wbase)
    print(f'  PSEC @0x{PSEC_ADDR:05x}: {np_beats} beats')

    cat, conf = int_forward(load_images(range(N_IMAGES)), qm)
    with open(os.path.join(PATTERN, 'golden_results.txt'), 'w') as f:
        for c, v in zip(cat, conf):
            f.write(f'{int(c)} {int(v)}\n')
    print(f'golden_results.txt：{N_IMAGES} 張（category confidence）')
