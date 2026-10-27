"""omega-dnn: deep-learning primitives compiled from om/dnn.om (seeded Omega, OM-2).

This module is the Python wrapper over the omega-bridge ABI v2 binding (build/omega_native.py,
generated, not edited). It owns no arithmetic: every value is computed by the Omega unit. The
wrapper lays tensors out in one float32 arena, writes the node descriptor (docs/DESCRIPTOR.md)
and calls ``dnn_run``. Tensors enter through DLPack (``torch.from_dlpack``) and leave as torch
tensors that view the arena (zero copy) or as copies.
"""
import ctypes
import importlib.util
import os
import struct

import torch

MAGIC = 0x4F444E
NONE = 0xFFFFFFFFFFFFFFFF
MAX_NODES = 31
KEEP = 2

RMS, LN, SOFTMAX, ROPE = 1, 2, 3, 4
SILU, GELU, SWIGLU, ADD, MUL, BIAS, RELU = 5, 6, 7, 8, 9, 10, 11
MATMUL, ATTN, PAGED_DECODE, CONV, MAXPOOL, AVGPOOL, GAP = 12, 13, 14, 15, 16, 17, 18
ELEMENTWISE = (SILU, GELU, SWIGLU, ADD, MUL, BIAS, RELU)


def _load_binding():
    d = os.environ.get("OMEGA_DNN_BUILD") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "build")
    path = os.path.join(d, "omega_native.py")
    spec = importlib.util.spec_from_file_location("omega_dnn_native", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_nat = _load_binding()


class OmegaError(RuntimeError):
    pass


def f64bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def abi():
    r = ctypes.c_uint64()
    st = _nat.omega2_dnn_abi(ctypes.byref(r))
    if st:
        raise OmegaError("dnn_abi status %d" % st)
    return r.value


class Graph:
    """A node list over one float32 arena.

    ``input``/``ints``/``empty`` place tensors (returning their element offset); ``node`` appends
    one operator; ``run`` executes the list, fusing elementwise followers when ``fuse`` is set,
    and returns the number of passes the Omega unit ran.
    """

    def __init__(self):
        self._chunks = []
        self._size = 0
        self._nodes = []
        self.arena = None

    def _place(self, n, init):
        off = self._size
        self._size += max(int(n), 1)
        self._chunks.append((off, init))
        return off

    def input(self, t):
        t = torch.from_dlpack(t) if not isinstance(t, torch.Tensor) else t
        t = t.detach().to(torch.float32).contiguous().reshape(-1)
        return self._place(t.numel(), ("f", t))

    def ints(self, t):
        t = torch.as_tensor(t).to(torch.int64).reshape(-1)
        if t.numel() and (int(t.min()) < 0 or int(t.max()) > 0xFFFFFFFF):
            raise ValueError("uint32 side input out of range")
        return self._place(t.numel(), ("i", t))

    def empty(self, n):
        return self._place(n, None)

    def node(self, op, args, keep=False):
        if len(self._nodes) >= MAX_NODES:
            raise ValueError("at most %d nodes" % MAX_NODES)
        a = [0] * 16
        a[0] = op
        for k, v in args.items():
            if not 1 <= k <= 14:
                raise ValueError("argument index %d" % k)
            a[k] = int(v)
        a[15] = KEEP if keep else 0
        self._nodes.append(a)
        return a[3]

    def _desc(self, fuse):
        words = [0] * 512
        words[0] = MAGIC
        words[1] = len(self._nodes)
        words[2] = 1 if fuse else 0
        for i, a in enumerate(self._nodes):
            words[4 + 16 * i: 20 + 16 * i] = a
        return (ctypes.c_ubyte * 4096).from_buffer_copy(struct.pack("<512Q", *words))

    def materialize(self):
        self.arena = torch.zeros(self._size, dtype=torch.float32)
        iv = self.arena.view(torch.int32)
        for off, init in self._chunks:
            if init is None:
                continue
            kind, t = init
            if kind == "f":
                self.arena[off: off + t.numel()] = t
            else:
                iv[off: off + t.numel()] = t.to(torch.uint32).view(torch.int32) if hasattr(torch, "uint32") else t.to(torch.int32)
        return self.arena

    def run(self, fuse=True):
        if self.arena is None:
            self.materialize()
        desc = self._desc(fuse)
        res = ctypes.c_uint64()
        ws = ctypes.cast(self.arena.data_ptr(), ctypes.POINTER(ctypes.c_uint32))
        st = _nat.omega2_dnn_run(ctypes.byref(res), ws, self.arena.numel(),
                                 ctypes.cast(desc, ctypes.c_char_p))
        if st:
            raise OmegaError("dnn_run status %d" % st)
        return res.value

    def out(self, off, shape):
        n = 1
        for s in shape:
            n *= int(s)
        return self.arena[off: off + n].view(*shape)


def _one(op, ins, args, out_n, out_shape, ints=None, extra=None):
    g = Graph()
    offs = {k: g.input(v) for k, v in ins.items()}
    for k, v in (ints or {}).items():
        offs[k] = g.ints(v)
    for k, n in (extra or {}).items():
        offs[k] = g.empty(n)
    offs[3] = g.empty(out_n)
    a = dict(args)
    a.update(offs)
    g.node(op, a, keep=True)
    g.run()
    return g.out(offs[3], out_shape).clone()


def rms_norm(x, scale, eps=1e-6):
    c = x.shape[-1]
    return _one(RMS, {1: x, 2: scale}, {4: x.numel() // c, 5: c, 6: f64bits(eps)}, x.numel(), x.shape)


def layer_norm(x, scale, bias=None, eps=1e-5):
    c = x.shape[-1]
    ins = {1: x, 2: scale}
    args = {4: x.numel() // c, 5: c, 6: f64bits(eps)}
    if bias is None:
        args[7] = NONE
    else:
        ins[7] = bias
    return _one(LN, ins, args, x.numel(), x.shape)


def softmax(x):
    c = x.shape[-1]
    return _one(SOFTMAX, {1: x}, {4: x.numel() // c, 5: c}, x.numel(), x.shape)


def rope(x, cos, sin, pos, rot=0, interleaved=False):
    b, h, s, d = x.shape
    return _one(ROPE, {1: x, 2: cos, 4: sin},
                {6: b, 7: h, 8: s, 9: d, 10: rot, 11: 1 if interleaved else 0},
                x.numel(), x.shape, ints={5: pos})


def elementwise(op, a, b=None, gelu_tanh=False, cols=0):
    ins = {1: a}
    if b is not None:
        ins[2] = b
    args = {4: a.numel()}
    if op == GELU:
        args[6] = 1 if gelu_tanh else 0
    if op == BIAS:
        args[5] = cols or b.numel()
    return _one(op, ins, args, a.numel(), a.shape)


def matmul(a, b, bias=None, trans_b=False):
    m, k = a.shape
    n = b.shape[0] if trans_b else b.shape[1]
    ins = {1: a, 2: b}
    args = {4: m, 5: n, 6: k, 8: 1 if trans_b else 0}
    if bias is None:
        args[7] = NONE
    else:
        ins[7] = bias
    return _one(MATMUL, ins, args, m * n, (m, n))


def attention(q, k, v, causal=False, scale=None, block=64):
    b, hq, sq, d = q.shape
    hkv, skv = k.shape[1], k.shape[2]
    sc = scale if scale is not None else d ** -0.5
    return _one(ATTN, {1: q, 2: k, 4: v},
                {5: b, 6: hq, 7: hkv, 8: sq, 9: skv, 10: d, 11: 1 if causal else 0,
                 12: f64bits(sc), 13: block}, q.numel(), q.shape)


def paged_decode(q, k_cache, v_cache, block_table, seqlens, scale=None):
    b, hq, d = q.shape
    _, bs, hkv, _ = k_cache.shape
    sc = scale if scale is not None else d ** -0.5
    return _one(PAGED_DECODE, {1: q, 2: k_cache, 4: v_cache},
                {7: b, 8: hq, 9: hkv, 10: d, 11: bs, 12: block_table.shape[1], 13: f64bits(sc)},
                q.numel(), q.shape, ints={5: block_table, 6: seqlens})


def _odim(h, k, s, p, dl):
    return (h + 2 * p - dl * (k - 1) - 1) // s + 1


def _pack(vals):
    w = 0
    for i, v in enumerate(vals):
        if not 0 <= v <= 255:
            raise ValueError("packed conv/pool parameter out of 0..255")
        w |= int(v) << (8 * i)
    return w


def conv2d(x, w, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, algo=0):
    n, c, h, wd = x.shape
    o, cg, kh, kw = w.shape
    oh = _odim(h, kh, stride[0], padding[0], dilation[0])
    ow = _odim(wd, kw, stride[1], padding[1], dilation[1])
    ins = {1: x, 2: w}
    args = {6: n, 7: c, 8: h, 9: wd, 10: o, 12: groups, 13: algo,
            11: _pack([kh, kw, stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1]])}
    if bias is None:
        args[4] = NONE
    else:
        ins[4] = bias
    extra = {5: cg * kh * kw * oh * ow if algo == 1 else 1}
    return _one(CONV, ins, args, n * o * oh * ow, (n, o, oh, ow), extra=extra)


def pool2d(x, kernel, stride, padding=(0, 0), avg=False, count_include_pad=False):
    n, c, h, w = x.shape
    oh = _odim(h, kernel[0], stride[0], padding[0], 1)
    ow = _odim(w, kernel[1], stride[1], padding[1], 1)
    args = {4: n * c, 5: h, 6: w, 8: 1 if count_include_pad else 0,
            7: _pack([kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]])}
    return _one(AVGPOOL if avg else MAXPOOL, {1: x}, args, n * c * oh * ow, (n, c, oh, ow))


def global_avg_pool(x):
    n, c, h, w = x.shape
    return _one(GAP, {1: x}, {4: n * c, 5: h * w}, n * c, (n, c, 1, 1))
