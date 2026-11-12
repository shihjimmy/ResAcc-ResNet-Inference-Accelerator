"""
sweep.py -- 掃描不同的位元寬度，看準確度怎麼變（決定硬體要用幾 bit）

表格數字 = 跟浮點模型類別一致的比例：前 64 張 / 全部 192 張
只看 64 張雜訊很大（一張圖 = 1.6%），所以兩個一起看。
"""
from common import load_pth, N_IMAGES
from quant import build_quant_model, evaluate, TEST_IMAGES

WEIGHT_BITS = (8, 6, 5, 4, 3)
ACT_BITS = (8, 7, 6, 5, 4)

if __name__ == '__main__':
    P = load_pth()
    print('weight\\act ' + ''.join(f'{a:>16d}' for a in ACT_BITS))
    for wb in WEIGHT_BITS:
        row = ''
        for ab in ACT_BITS:
            qm = build_quant_model(P, weight_bits=wb, act_bits=ab)
            t = evaluate(qm, TEST_IMAGES)['agree']
            a = evaluate(qm, range(N_IMAGES))['agree']
            row += f'   {t:.3f} / {a:.3f}'
        print(f'{wb:9d}  ' + row)
