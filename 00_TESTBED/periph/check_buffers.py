"""
check_buffers.py -- 周邊測試：把 RTL 載入的記憶體內容（TESTBED 用 DUMP_BUFFERS 輸出）
跟 hw_model.py（照 SPEC 寫的 Python 模型）載入的內容比對。

用法：python3 check_buffers.py <build 目錄>
"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '98_ALGORITHM'))
import hw_model as H


def read_mem(path, nbytes):
    """$writememh 的輸出 → (行數, nbytes) 的 uint8，byte 0 = 最低位"""
    rows = []
    for line in open(path):
        line = line.split('//')[0].strip()
        if not line or line.startswith('@'):
            continue
        v = int(line, 16)
        rows.append([(v >> (8 * k)) & 0xFF for k in range(nbytes)])
    return np.array(rows, np.uint8)


build = sys.argv[1]
dram = H.DRAM()
dram.readmemh(os.path.join(H.PATTERN, 'model.dat'))
chip = H.Chip()
chip.load_model(dram)
ok = True

def cmp(name, got, exp):
    global ok
    same = got.shape == exp.shape and np.array_equal(got, exp)
    print(f'  {name:8s} {"一致" if same else "不一致"}  {got.shape}')
    ok &= same

print('模型載入：')
cmp('WBUF', read_mem(os.path.join(build, 'wbuf.hex'), 16), chip.WBUF)
cmp('PBUF', read_mem(os.path.join(build, 'pbuf.hex'), 32), chip.PBUF)
cmp('FCMEM', read_mem(os.path.join(build, 'fcm.hex'), 32), chip.FCMEM)
hdr = [l.strip() for l in open(os.path.join(build, 'header.txt'))]
shift = int(hdr[0], 16)
s_rtl = [(shift >> (8 * k)) & 0xFF for k in range(7)]
bfc = int(hdr[3], 16)
bfc_rtl = [((bfc >> (32 * k)) & 0xFFFFFFFF) for k in range(10)]
bfc_rtl = [v - (1 << 32) if v >= (1 << 31) else v for v in bfc_rtl]
for name, got, exp in [('s', s_rtl, chip.s), ('conf_m', int(hdr[1], 16), chip.conf_m),
                       ('conf_s', int(hdr[2], 16), chip.conf_s), ('bfc', bfc_rtl, chip.bfc)]:
    same = got == exp
    print(f'  {name:8s} {"一致" if same else "不一致"}  RTL={got}')
    ok &= same

imgs = sorted(glob.glob(os.path.join(build, 'img_*.hex')), key=lambda p: int(p.split('_')[-1].split('.')[0]))
n_img_ok = 0
for p in imgs:
    img_id = int(p.split('_')[-1].split('.')[0])
    d = H.DRAM(); d.load_image(0, img_id)
    c = H.Chip(); c.load_image(d, 0)
    if np.array_equal(read_mem(p, 64), c.IMG):
        n_img_ok += 1
    else:
        print(f'  image {img_id} 的 IMG 內容不一致'); ok = False
print(f'圖片載入：{n_img_ok} / {len(imgs)} 張一致')
print('PASS' if ok and imgs else 'FAIL')
