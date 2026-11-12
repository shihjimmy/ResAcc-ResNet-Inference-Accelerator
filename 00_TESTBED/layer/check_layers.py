"""
check_layers.py -- 逐層驗證：RTL 每一層算完寫回的 FM（TESTBED 的 DUMP_LAYERS 輸出）跟 GOLDEN 比對
用法：python3 check_layers.py <build 目錄>
"""
import os, sys, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '98_ALGORITHM'))
import hw_model as H

STEPS = {0: ('stem_out', (16, 16, 16)), 1: ('l1_a_out', (16, 8, 8)), 2: ('l1_out', (16, 8, 8)),
         3: ('l2_a_out', (32, 4, 4)), 4: ('pool_sum', (32,))}


def read_mem(path):
    rows = []
    for line in open(path):
        line = line.split('//')[0].strip()
        if not line or line.startswith('@'):
            continue
        if 'x' in line.lower():
            return None                                   # 有 X：還沒寫過
        v = int(line, 16)
        rows.append([(v >> (8 * k)) & 0xFF for k in range(64)])
    return np.array(rows, np.uint8)


build = sys.argv[1]
files = sorted(glob.glob(os.path.join(build, 'L*_img*.hex')))
n_ok = n_bad = 0
for f in files:
    step, img = map(int, re.findall(r'L(\d+)_img(\d+)', f)[0])
    name, shape = STEPS[step]
    gpath = os.path.join(H.PATTERN, 'GOLDEN', f'image_{img}', name + '.hex')
    if step == 4:
        got = np.array([int(l, 16) for l in open(f) if l.strip()])
        exp = H.read_hex(gpath, 12, False)
    else:
        mem = read_mem(f)
        if mem is None:
            print(f'image {img} {name}: 有 X（未寫入的位置）'); n_bad += 1; continue
        # 只比對這個 tensor 用到的 word（FM_B 在 l2_a 只用前 8 個 word）
        got = H.fm_to_tensor(np.vstack([mem, np.zeros((64, 64), np.uint8)]), *shape)
        exp = H.read_hex(gpath, 8, False).reshape(shape)
    if np.array_equal(got, exp):
        n_ok += 1
    else:
        n_bad += 1
        bad = np.argwhere(got != exp)
        print(f'image {img} {name}: {len(bad)} 個值不同，第一個在 {tuple(bad[0])}：RTL {got[tuple(bad[0])]} golden {exp[tuple(bad[0])]}')
print(f'逐層比對：{n_ok} 一致，{n_bad} 不一致（共 {len(files)} 個檔案）')
print('PASS' if n_bad == 0 and files else 'FAIL')
