"""Float reference model of ResNet6 in numpy.

Two versions that must give identical results:
  * forward_unfused : conv -> BatchNorm -> ReLU, exactly like the PyTorch model
  * forward_fused   : BN folded into conv weight/bias (what the hardware will use)

Run:  python3 float_ref.py
"""
import numpy as np
from common import load_pth, load_images, load_golden, conv2d, relu, N_IMAGES, N_TEST, BN_EPS

# (conv name, bn name, stride, pad) for every conv+BN pair in the network
CONV_BN = {
    'stem':    ('conv1.weight',              'bn1',                2, 1),
    'l1_a':    ('layer1.0.conv1.weight',     'layer1.0.bn1',       2, 1),
    'l1_b':    ('layer1.0.conv2.weight',     'layer1.0.bn2',       1, 1),
    'l1_sc':   ('layer1.0.shortcut.0.weight', 'layer1.0.shortcut.1', 2, 0),
    'l2_a':    ('layer2.0.conv1.weight',     'layer2.0.bn1',       2, 1),
    'l2_b':    ('layer2.0.conv2.weight',     'layer2.0.bn2',       1, 1),
    'l2_sc':   ('layer2.0.shortcut.0.weight', 'layer2.0.shortcut.1', 2, 0),
}


def preprocess(pix):
    """uint8 pixel -> model input, same as ToTensor() + Normalize(0.5, 0.5)."""
    return (pix.astype(np.float64) / 255.0 - 0.5) / 0.5


# ------------------------------------------------------------------ unfused
def bn(z, P, name):
    g, b = P[name + '.weight'], P[name + '.bias']
    mu, var = P[name + '.running_mean'], P[name + '.running_var']
    s = lambda v: v.astype(np.float64)[None, :, None, None]
    return s(g) * (z - s(mu)) / np.sqrt(s(var) + BN_EPS) + s(b)


def forward_unfused(pix, P):
    def cb(x, key):
        w, bn_name, st, pd = CONV_BN[key]
        return bn(conv2d(x, P[w].astype(np.float64), st, pd), P, bn_name)

    x = preprocess(pix)
    x = relu(cb(x, 'stem'))
    for L in ('l1', 'l2'):
        y = relu(cb(x, L + '_a'))
        y = cb(y, L + '_b')
        x = relu(y + cb(x, L + '_sc'))
    x = x.mean(axis=(2, 3))                                   # global avg pool
    return x @ P['fc.weight'].T.astype(np.float64) + P['fc.bias']


# ------------------------------------------------------------------ fused
def fuse(P):
    """Fold every BN into its conv.  Returns {key: (w', b', stride, pad)} + fc."""
    F = {}
    for key, (w, bn_name, st, pd) in CONV_BN.items():
        g = P[bn_name + '.weight'].astype(np.float64)
        b = P[bn_name + '.bias'].astype(np.float64)
        mu = P[bn_name + '.running_mean'].astype(np.float64)
        sigma = np.sqrt(P[bn_name + '.running_var'].astype(np.float64) + BN_EPS)
        k = g / sigma                                         # per output channel
        F[key] = (P[w].astype(np.float64) * k[:, None, None, None],   # w' = w*g/sigma
                  b - mu * k,                                         # b' = beta - mu*g/sigma
                  st, pd)
    F['fc'] = (P['fc.weight'].astype(np.float64), P['fc.bias'].astype(np.float64))
    return F


def forward_fused(pix, F, taps=None):
    """taps: optional dict, filled with every intermediate activation (for calibration)."""
    def cb(x, key):
        w, b, st, pd = F[key]
        return conv2d(x, w, st, pd) + b[None, :, None, None]

    t = taps if taps is not None else {}
    x = relu(cb(preprocess(pix), 'stem'));        t['stem'] = x
    for L in ('l1', 'l2'):
        y = relu(cb(x, L + '_a'));                t[L + '_a'] = y
        x = relu(cb(y, L + '_b') + cb(x, L + '_sc')); t[L + '_out'] = x
    x = x.mean(axis=(2, 3));                      t['pool'] = x
    W, b = F['fc']
    return x @ W.T + b


# ------------------------------------------------------------------ check
if __name__ == '__main__':
    P = load_pth()
    pix = load_images(range(N_IMAGES))
    gold_cat, gold_conf = load_golden()

    out_u = forward_unfused(pix, P)
    out_f = forward_fused(pix, fuse(P))

    print('unfused vs fused   max |diff| :', np.abs(out_u - out_f).max())
    pred = out_f.argmax(1)
    conf = out_f.max(1)
    n = len(gold_conf)
    print('numpy vs PyTorch   class match:', (pred[:n] == gold_cat[:n]).sum(), '/', n)
    print('numpy vs PyTorch   max |conf diff| :', np.abs(conf[:n] - gold_conf).max())
    print('first 8 images   numpy conf :', np.round(conf[:8], 5))
    print('                 golden csv :', np.round(gold_conf[:8], 5))
