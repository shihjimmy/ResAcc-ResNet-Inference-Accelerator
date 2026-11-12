"""
quant.py -- ResNet6 的整數（量化）模型，也就是硬體的 golden model
=====================================================================

這個檔案做三件事：
  1. build_quant_model() : 把浮點模型 (resnet6.pth) 轉成「純整數」模型
  2. int_forward()       : 用純整數運算做推論（硬體要做到跟它每個 bit 都一樣）
  3. dump_golden()       : 把每一層的中間結果存成 hex 檔，給 RTL 模擬逐層比對

量化的基本關係（全部 zero-point = 0）：
      實數 r  =  scale S  ×  整數 q

每一層 conv 在硬體裡做的事：
      acc   = Σ (w_q × x_q) + b_q                  ← 純整數乘加（寬度約 20 bit）
      y_q   = clamp( (acc × m + 2^(s-1)) >> s, 0, 2^ACT_BITS - 1 )
              └─ 「× m >> s」近似「× (S_x·S_w / S_y)」這個小數倍率
              └─ clamp 下限 0 = ReLU

執行：
      python3 quant.py         → 印出量化參數與準確度、輸出 golden hex 與參數檔
      python3 sweep.py         → 掃描不同位元寬度（另一個檔案）
"""
import os
from dataclasses import dataclass, field

import numpy as np

from common import load_pth, load_images, load_golden, conv2d, N_IMAGES, N_TEST, HERE
from float_ref import fuse, forward_fused


# =====================================================================
# 0. 設定
# =====================================================================
WEIGHT_BITS = 8            # 權重位元數（有號）       → 範圍 [-127, 127]
ACT_BITS = 8               # activation 位元數（無號） → 範圍 [0, 255]
MULT_BITS = 16             # requantize 乘數 m 的位元數
CALIB_PERCENTILE = 99.99   # 用第幾百分位當 activation 的最大值（不用絕對最大，避免被極端值拖累）

CALIB_IMAGES = range(N_TEST, N_IMAGES)   # 64..191 → 只拿來決定 scale
TEST_IMAGES = range(0, N_TEST)           # 0..63   → 只拿來評估，絕不拿來調參數

# 模型輸入：硬體拿到 uint8 pixel p，先算 q_in = p - 128（有號 8 bit）
# 浮點模型的輸入是 x = (p - 127.5) / 127.5 = INPUT_SCALE × (q_in + 0.5)
INPUT_SCALE = 1 / 127.5

CONF_SHIFT = 20            # host 會把 confidence 除以 2^20 再印出來

# 需要量化的 activation（每個都會有自己的 scale）
#   stem   : 第一層 conv 的輸出               16×16×16
#   l1_a   : block1 第一個 conv 的輸出         16×8×8
#   l1_out : block1 的輸出（相加 + ReLU 後）  16×8×8
#   l2_a   : block2 第一個 conv 的輸出         32×4×4
#   l2_out : block2 的輸出                    32×4×4
ACT_TENSORS = ['stem', 'l1_a', 'l1_out', 'l2_a', 'l2_out']

# 每個 conv 的（輸入 tensor, 輸出 tensor）
# 注意：l1_b 和 l1_sc 的結果要相加，所以輸出都是 l1_out（l2 同理）
CONV_IO = {
    'stem':  ('input',  'stem'),
    'l1_a':  ('stem',   'l1_a'),
    'l1_b':  ('l1_a',   'l1_out'),
    'l1_sc': ('stem',   'l1_out'),     # shortcut：從 block 的輸入出發
    'l2_a':  ('l1_out', 'l2_a'),
    'l2_b':  ('l2_a',   'l2_out'),
    'l2_sc': ('l1_out', 'l2_out'),
}


# =====================================================================
# 1. 基本工具
# =====================================================================
def quantize_weight_per_channel(w, bits):
    """權重量化：每個輸出通道一個 scale，對稱（無 zero-point）。

    w     : 浮點權重 (O, C, K, K) 或 (O, C)
    回傳  : w_q (整數), scale (長度 O)
    做法  : 該通道 |w| 最大值 對應到 2^(bits-1)-1，例如 8 bit 對應到 127
    """
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(w).reshape(len(w), -1).max(axis=1) / qmax
    shape = (-1,) + (1,) * (w.ndim - 1)                 # 讓 scale 能對每個通道廣播
    w_q = np.round(w / scale.reshape(shape)).astype(np.int64)
    return w_q, scale


def float_to_multiplier(M, mult_bits):
    """把一組正的小數 M 近似成 m / 2^s（m 是整數，所有元素共用同一個 s）。

    例：M = 0.00345，mult_bits = 16
        選最大的 s 使 m 不超過 16 bit → s = 23
        m = round(0.00345 × 2^23) = 28940
        硬體算 (acc × 28940) >> 23 ≈ acc × 0.00345
    s 選越大，m 的有效位數越多，近似越準。
    """
    M = np.atleast_1d(M)
    s = int(np.floor(np.log2((2 ** mult_bits - 1) / M.max())))
    m = np.round(M * 2.0 ** s).astype(np.int64)
    return m, s


def round_shift(v, s):
    """四捨五入的右移：(v + 2^(s-1)) >> s。numpy 對 int64 的 >> 是算術右移（往 -∞），跟硬體一樣。"""
    return (v + (1 << (s - 1))) >> s


def clamp_act(v, act_bits):
    """夾到 activation 範圍 [0, 2^act_bits - 1]。下限 0 就是 ReLU。"""
    return np.clip(v, 0, 2 ** act_bits - 1)


def per_channel(v):
    """把長度 O 的向量變成 (1, O, 1, 1)，才能跟 (N, O, H, W) 的 tensor 逐通道運算。"""
    return np.asarray(v)[None, :, None, None]


# =====================================================================
# 2. 量化後的模型參數
# =====================================================================
@dataclass
class QConv:
    """一個量化後的 conv 層（BN 已融合）。"""
    name: str
    w_q: np.ndarray        # 整數權重 (O, C, K, K)
    b_q: np.ndarray        # 整數 bias (O,)，已換算到累加器的 scale
    stride: int
    pad: int
    m: np.ndarray = None   # requantize 乘數 (O,)，每個輸出通道一個
    s: int = None          # requantize 右移量，整層共用
    M_float: np.ndarray = None   # m/2^s 想要近似的真正倍率（只用來檢查）


@dataclass
class QModel:
    """整個量化模型。int_forward() 只需要這裡面的整數。"""
    weight_bits: int
    act_bits: int
    act_scale: dict                       # 每個 activation 的 scale（只在建模型時用到）
    conv: dict = field(default_factory=dict)   # name -> QConv
    fc_w_q: np.ndarray = None             # (10, 32)
    fc_b_q: np.ndarray = None             # (10,)
    conf_m: int = None                    # confidence = (logit × conf_m) >> conf_s
    conf_s: int = None


# =====================================================================
# 3. 建立量化模型
# =====================================================================
def calibrate(F, act_bits, percentile):
    """Calibration：用 CALIB_IMAGES 跑浮點模型，決定每個 activation 的 scale。

    S = (該層輸出的第 percentile 百分位數) / (2^act_bits - 1)
    activation 都在 ReLU 之後，一定 ≥ 0，所以用無號整數、zero-point = 0。
    """
    taps = {}
    forward_fused(load_images(CALIB_IMAGES), F, taps)        # taps 會收集每層浮點輸出
    scale = {'input': INPUT_SCALE}
    for name in ACT_TENSORS:
        scale[name] = np.percentile(taps[name], percentile) / (2 ** act_bits - 1)
    return scale


def build_quant_model(P, weight_bits=WEIGHT_BITS, act_bits=ACT_BITS,
                      mult_bits=MULT_BITS, percentile=CALIB_PERCENTILE):
    """浮點參數 P (resnet6.pth) → 整數模型 QModel。"""

    # --- 步驟 1：BN 融合（細節見 float_ref.fuse），之後每個 conv 都是 w', b'
    F = fuse(P)

    # --- 步驟 2：calibration，決定每個 activation 的 scale
    S = calibrate(F, act_bits, percentile)
    qm = QModel(weight_bits=weight_bits, act_bits=act_bits, act_scale=S)

    # --- 步驟 3：逐層量化權重與 bias
    for name, (src, dst) in CONV_IO.items():
        w, b, stride, pad = F[name]

        if name == 'stem':
            # 輸入 x = INPUT_SCALE × (q_in + 0.5)，那個 +0.5 對每個輸出通道貢獻一個常數
            # Σ w × INPUT_SCALE × 0.5，直接併進 bias，硬體就只要處理 q_in
            b = b + 0.5 * INPUT_SCALE * w.sum(axis=(1, 2, 3))

        w_q, s_w = quantize_weight_per_channel(w, weight_bits)

        # 累加器 acc = Σ w_q × x_q 的 scale 是 S_x × S_w（每通道不同）
        # bias 要先換算到同一個 scale 才能直接加進累加器
        acc_scale = S[src] * s_w
        b_q = np.round(b / acc_scale).astype(np.int64)

        # requantize 倍率：acc 的 scale → 輸出 activation 的 scale
        M = acc_scale / S[dst]
        qm.conv[name] = QConv(name, w_q, b_q, stride, pad, M_float=M)

    # --- 步驟 4：把每層的倍率 M 換成 m / 2^s
    for name in ('stem', 'l1_a', 'l2_a'):              # 一般的 conv
        c = qm.conv[name]
        c.m, c.s = float_to_multiplier(c.M_float, mult_bits)

    for blk in ('l1', 'l2'):                           # 殘差：conv_b 和 shortcut 要相加
        cb, csc = qm.conv[blk + '_b'], qm.conv[blk + '_sc']
        # 兩條路徑共用同一個 s，相加前各乘自己的 m：
        #   out = (acc_b × m_b + acc_sc × m_sc) >> s
        _, s = float_to_multiplier(np.concatenate([cb.M_float, csc.M_float]), mult_bits)
        for c in (cb, csc):
            c.s = s
            c.m = np.round(c.M_float * 2.0 ** s).astype(np.int64)

    # --- 步驟 5：FC 層
    W, b = F['fc']
    # FC 權重用「整層共用一個 scale」，這樣 10 個輸出在同一個 scale 下，
    # 可以直接比較整數大小做 argmax，不需要換算回實數
    s_fc = np.abs(W).max() / (2 ** (weight_bits - 1) - 1)
    qm.fc_w_q = np.round(W / s_fc).astype(np.int64)
    # avgpool 在硬體裡只做「16 個值相加」，除以 16 這一步併進 FC 的 scale
    fc_acc_scale = (S['l2_out'] / 16) * s_fc
    qm.fc_b_q = np.round(b / fc_acc_scale).astype(np.int64)
    # confidence = 實數 logit × 2^20 = fc_acc × (fc_acc_scale × 2^20)
    m, s = float_to_multiplier(fc_acc_scale * 2 ** CONF_SHIFT, mult_bits)
    qm.conf_m, qm.conf_s = int(m[0]), s

    return qm


# =====================================================================
# 4. 整數推論（硬體要做的事）
# =====================================================================
def conv_acc(x_q, c: QConv):
    """整數 conv：acc = Σ w_q × x_q + b_q。全部是整數運算。"""
    return conv2d(x_q, c.w_q, c.stride, c.pad) + per_channel(c.b_q)


def int_forward(pix, qm: QModel, record=None):
    """純整數推論。

    pix    : uint8 圖片 (N, 3, 32, 32)
    record : 給一個 dict 的話，會把每一層的中間結果存進去（給 dump_golden 用）
    回傳   : category (N,), confidence × 2^20 (N,)
    """
    r = record if record is not None else {}
    A = qm.act_bits

    # ---- 輸入：uint8 → 有號 8 bit
    x = pix.astype(np.int64) - 128

    # ---- stem：conv 3×3 s2 → requantize（含 ReLU）
    c = qm.conv['stem']
    acc = conv_acc(x, c)
    x = clamp_act(round_shift(acc * per_channel(c.m), c.s), A)
    r['stem_acc'], r['stem_out'] = acc, x

    # ---- 兩個 residual block
    for blk in ('l1', 'l2'):
        block_in = x

        # 主路徑第一個 conv：3×3 s2 → requantize（含 ReLU）
        ca = qm.conv[blk + '_a']
        acc_a = conv_acc(block_in, ca)
        y = clamp_act(round_shift(acc_a * per_channel(ca.m), ca.s), A)

        # 主路徑第二個 conv：3×3 s1，先不 requantize
        cb = qm.conv[blk + '_b']
        acc_b = conv_acc(y, cb)

        # shortcut：1×1 s2，從 block 的輸入出發，也先不 requantize
        csc = qm.conv[blk + '_sc']
        acc_sc = conv_acc(block_in, csc)

        # 相加：兩邊各乘自己的 m，加起來再一起右移、clamp（clamp 下限 0 = 最後的 ReLU）
        total = acc_b * per_channel(cb.m) + acc_sc * per_channel(csc.m)
        x = clamp_act(round_shift(total, cb.s), A)

        r[blk + '_a_acc'], r[blk + '_a_out'] = acc_a, y
        r[blk + '_b_acc'], r[blk + '_sc_acc'] = acc_b, acc_sc
        r[blk + '_out'] = x

    # ---- global average pool：只做加總（除以 16 已併進 FC 的 scale）
    pool_sum = x.sum(axis=(2, 3))                       # (N, 32)

    # ---- FC + argmax
    logit = pool_sum @ qm.fc_w_q.T + qm.fc_b_q          # (N, 10)
    category = logit.argmax(axis=1)
    confidence = round_shift(logit.max(axis=1) * qm.conf_m, qm.conf_s)

    r['pool_sum'], r['fc_logit'] = pool_sum, logit
    r['category'], r['confidence'] = category, confidence
    return category, confidence


# =====================================================================
# 5. 評估
# =====================================================================
def evaluate(qm, image_ids=TEST_IMAGES):
    """跟浮點模型（助教的 golden csv）比較。回傳 dict。"""
    ids = list(image_ids)
    gold_cat, gold_conf = load_golden()
    cat, conf = int_forward(load_images(ids), qm)
    return {
        'agree': float((cat == gold_cat[ids]).mean()),                     # 類別一致的比例
        'conf_err': float(np.abs(conf / 2 ** CONF_SHIFT - gold_conf[ids]).mean()),
        'mismatch': [i for i, a, g in zip(ids, cat, gold_cat[ids]) if a != g],
    }


# =====================================================================
# 6. 輸出 golden 檔案（給 RTL 比對）
# =====================================================================
# 每個中間結果的 (位元數, 是否有號)。hex 檔裡有號數用 2 補數表示。
def golden_formats(act_bits):
    fmt = {}
    for k in ('stem', 'l1_a', 'l1_b', 'l1_sc', 'l2_a', 'l2_b', 'l2_sc'):
        fmt[k + '_acc'] = (32, True)                    # 累加器（實際只用 ~20 bit，存 32 bit 方便看）
    for k in ('stem_out', 'l1_a_out', 'l1_out', 'l2_a_out', 'l2_out'):
        fmt[k] = (act_bits, False)                      # activation
    fmt['pool_sum'] = (act_bits + 4, False)            # 16 個值相加 → 多 4 bit
    fmt['fc_logit'] = (32, True)
    return fmt


def to_hex_lines(values, bits, signed):
    """整數陣列 → 每行一個 hex 字串。有號數取 2 補數。"""
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    v = np.asarray(values, dtype=np.int64).ravel()
    if not signed:
        assert v.min() >= 0 and v.max() <= mask, 'value out of range'
    return [f'{int(x) & mask:0{digits}x}' for x in v]


def dump_golden(qm, image_ids, out_dir):
    """把每張圖每一層的結果寫成 out_dir/image_<i>/<名稱>.hex。

    排列順序：tensor 是 (C, H, W) 的話，第 c*H*W + h*W + w 行就是 [c][h][w]
              （跟圖片在 DRAM 裡的排法一樣：一個通道接一個通道，每個通道一行一行排）
    """
    rec = {}
    ids = list(image_ids)
    int_forward(load_images(ids), qm, rec)
    fmt = golden_formats(qm.act_bits)
    for n, img in enumerate(ids):
        d = os.path.join(out_dir, f'image_{img}')
        os.makedirs(d, exist_ok=True)
        for name, (bits, signed) in fmt.items():
            with open(os.path.join(d, name + '.hex'), 'w') as f:
                f.write('\n'.join(to_hex_lines(rec[name][n], bits, signed)) + '\n')
        with open(os.path.join(d, 'result.txt'), 'w') as f:
            f.write(f"category   {int(rec['category'][n])}\n"
                    f"confidence {int(rec['confidence'][n])}  "
                    f"(= {rec['confidence'][n] / 2 ** CONF_SHIFT:.6f})\n")
    with open(os.path.join(out_dir, 'README.txt'), 'w') as f:
        f.write('Golden intermediate results from quant.py (one value per line, hex).\n'
                'Tensor order: index = c*H*W + h*W + w. Signed values are two\'s complement.\n\n')
        shapes = {k: rec[k].shape[1:] for k in fmt}
        for name, (bits, signed) in fmt.items():
            f.write(f"{name:12s} shape {str(shapes[name]):14s} {bits:2d} bit "
                    f"{'signed' if signed else 'unsigned'}\n")


def save_params(qm, path):
    """把所有整數參數存成 .npz，之後 gen_model_dat.py 會用它排出 model.dat。"""
    d = {'weight_bits': qm.weight_bits, 'act_bits': qm.act_bits,
         'fc_w_q': qm.fc_w_q, 'fc_b_q': qm.fc_b_q, 'conf_m': qm.conf_m, 'conf_s': qm.conf_s}
    for name, c in qm.conv.items():
        d[name + '_w_q'], d[name + '_b_q'] = c.w_q, c.b_q
        d[name + '_m'], d[name + '_s'] = c.m, c.s
    np.savez(path, **d)


# =====================================================================
# 7. 主程式
# =====================================================================
def print_summary(qm, rec):
    print(f'=== {qm.weight_bits}-bit weights / {qm.act_bits}-bit activations ===')
    print('activation scale:')
    for k, v in qm.act_scale.items():
        print(f'    {k:7s} {v:.5f}   (q=255 代表 {v * (2 ** qm.act_bits - 1):.3f})')
    print('per layer:      s    m (min..max)     |b_q| max   acc 需要的位元數(有號)')
    for name, c in qm.conv.items():
        acc = rec[name + '_acc']
        acc_bits = int(np.ceil(np.log2(np.abs(acc).max() + 1))) + 1
        rel_err = np.abs(c.m / 2.0 ** c.s - c.M_float).max() / c.M_float.min()
        print(f'    {name:6s} {c.s:3d}  {c.m.min():6d}..{c.m.max():6d}   {np.abs(c.b_q).max():8d}'
              f'   {acc_bits:3d}      (m/2^s 相對誤差 {rel_err:.1e})')
    fc_bits = int(np.ceil(np.log2(np.abs(rec['fc_logit']).max() + 1))) + 1
    print(f"    fc     confidence = (logit × {qm.conf_m}) >> {qm.conf_s}    fc acc {fc_bits} bit")


if __name__ == '__main__':
    qm = build_quant_model(load_pth())

    rec = {}
    int_forward(load_images(range(N_IMAGES)), qm, rec)
    print_summary(qm, rec)

    for label, ids in (('test  0..63 ', TEST_IMAGES), ('all   0..191', range(N_IMAGES))):
        e = evaluate(qm, ids)
        print(f"跟浮點模型一致 [{label}] {e['agree']:.4f}   "
              f"confidence 平均誤差 {e['conf_err']:.4f}   不一致的圖: {e['mismatch']}")

    out = os.path.join(HERE, '..', '00_TESTBED', 'PATTERN', 'GOLDEN')
    dump_golden(qm, TEST_IMAGES, out)
    save_params(qm, os.path.join(HERE, 'quant_params.npz'))
    print(f'golden hex → {os.path.normpath(out)}  (image_0 .. image_{N_TEST - 1})')
    print('整數參數   → quant_params.npz')
