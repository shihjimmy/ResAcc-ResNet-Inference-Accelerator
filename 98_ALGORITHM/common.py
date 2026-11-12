"""Shared helpers: load resnet6.pth without torch, load test images, conv2d.

Everything here is plain numpy so the flow runs on any machine.
"""
import os, zipfile, pickle, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN = os.path.join(HERE, '..', '00_TESTBED', 'PATTERN')
N_IMAGES = 192          # images provided
N_TEST = 64             # images used for scoring (image_0 .. image_63)
BN_EPS = 1e-5           # nn.BatchNorm2d default, not stored in the .pth


# ---------------------------------------------------------------- .pth loader
def load_pth(path=os.path.join(HERE, 'resnet6.pth')):
    """Read a PyTorch state_dict zip into {name: np.ndarray} without torch."""
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith('data.pkl')][0]
    root = pkl.rsplit('/', 1)[0]
    dtypes = {'FloatStorage': np.float32, 'LongStorage': np.int64}

    class _Storage:
        def __init__(self, dtype, key):
            self.dtype, self.key = dtype, key

    def _rebuild(storage, offset, size, stride, *args):
        raw = np.frombuffer(z.read(f'{root}/data/{storage.key}'), storage.dtype)
        n = int(np.prod(size)) if len(size) else 1
        return raw[offset:offset + n].reshape(size).copy()

    class _Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2':
                return _rebuild
            if mod == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if mod == 'torch' and name in dtypes:
                return dtypes[name]
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, storage_type, key, _loc, _n = pid
            return _Storage(storage_type, key)

    sd = _Unpickler(z.open(pkl)).load()
    return {k: v for k, v in sd.items() if not k.endswith('num_batches_tracked')}


# ---------------------------------------------------------------- data
def load_image(i):
    """uint8 array, shape (3, 32, 32), CHW as stored in image_i.bin."""
    p = os.path.join(PATTERN, 'IMAGES', f'image_{i}.bin')
    return np.fromfile(p, np.uint8).reshape(3, 32, 32)


def load_images(idx):
    return np.stack([load_image(i) for i in idx])          # (N, 3, 32, 32)


def load_golden():
    """PyTorch float-model results shipped by the TA (predicted class, max logit)."""
    cat = np.loadtxt(os.path.join(PATTERN, 'categories.csv'), dtype=int)
    conf = np.loadtxt(os.path.join(PATTERN, 'confidences.csv'))
    return cat, conf


# ---------------------------------------------------------------- math
def conv2d(x, w, stride, pad):
    """Batched conv. x: (N,C,H,W), w: (O,C,K,K) -> (N,O,Ho,Wo).

    Works for float and for integer arrays (integer stays exact if dtype is int64).
    """
    N, C, H, W = x.shape
    O, _, K, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - K) // stride + 1
    Wo = (W + 2 * pad - K) // stride + 1
    # im2col: gather every KxK window -> (N, C, K, K, Ho, Wo)
    cols = np.empty((N, C, K, K, Ho, Wo), dtype=xp.dtype)
    for ki in range(K):
        for kj in range(K):
            cols[:, :, ki, kj] = xp[:, :, ki:ki + stride * Ho:stride, kj:kj + stride * Wo:stride]
    return np.einsum('ockl,nckluv->nouv', w, cols)


def relu(x):
    return np.maximum(x, 0)
