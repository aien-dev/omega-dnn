"""GB10 tier: CUDApp native kernels against the omega-dnn host reference (docs/TOLERANCE.md, "GPU tier").

Usage:
  python tests/test_gpu.py --lib LIB --group ew|attn|gemm [--cases N] [--seed S] --out receipt.json [--hold]

LIB is a shared build of CUDApp's libomega_gpu.a (README, "GPU tier"). Every case is prepared on the host first
(the omega-dnn unit's output and a torch float64 reference); with --hold the chip calls then run under
flock /tmp/aien-gb10.lock with the BOARD and health-check discipline of the Spark. Without --hold nothing touches the
chip and the run stops after the preparation (a dry run).

Each chip output element y must satisfy
  |y - host| <= T_chip + T_host        (the gate: the omega-dnn unit is the reference)
  |y - ref64| <= T_chip + T_ref64      (diagnostic against torch float64)
where T_host is the host contract of tests/test_ops.py, T_ref64 = 2 gamma64(n + 64) M + 2^-149, and T_chip is the
op's bound for the CUDApp kernel against exact arithmetic, derived in docs/TOLERANCE.md and frozen before the run.
Every op also runs CUDApp's own chip mutants, and each mutant must fail the gate (negative control).
Exit status 1 on any gate violation, chip error, or surviving mutant.
"""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import omega_dnn as od  # noqa: E402

U = 2.0 ** -24            # f32 unit roundoff
U64 = 2.0 ** -53
TINY = 2.0 ** -149
FLOOR = 2.0 ** -124       # absolute allowance: a few f32 products that may flush below 2^-126 on the chip
E_EX2 = 5 * U             # PTX ISA 9.4 ex2.approx.f32: 2 ulp from the correctly rounded result -> 2.5 ulp <= 5u
E_RCP = 2 * U             # PTX ISA 9.4 rcp.approx.f32: 1 ulp
NEWTON = 3.0000007 * U    # Wolfram Maximize over the rsqrt Newton step (docs/TOLERANCE.md)
E_HEXP = 25 * U           # host_exp2 (Cephes polynomial in f32) relative error: 24.27u (Wolfram)
C_HMMA = (17 * 2.0 ** -25 + 2.0 ** -23) * (1 + 2.0 ** -20)   # one HMMA.16816 step, chip-exact model {16,2,RZ,nominal}
BOARD = "/home/drakestapleton/handoffs/BOARD.md"
LOCK = "/tmp/aien-gb10.lock"
LANE = "DnnGpu"


def g32(n):
    return n * U / (1 - n * U)


def g64(n):
    return n * U64 / (1 - n * U64)


def host_tol(ref, n, mag):
    return U * ref.abs() + 2 * g64(n + 64) * mag + TINY


def ref_tol(n, mag):
    return 2 * g64(n + 64) * mag + TINY


def ri(g, lo, hi):
    return int(torch.randint(lo, hi + 1, (1,), generator=g))


def rn(g, *shape, scale=1.0):
    return torch.randn(*shape, generator=g) * scale


def pick(g, xs):
    return xs[ri(g, 0, len(xs) - 1)]


def ptr(t):
    return C.c_void_p(t.data_ptr())


# ------------------------------------------------------------------ CUDApp ABI (src/*.h at the pinned sha)
class EwInfo(C.Structure):
    _fields_ = [("chip_calls", C.c_uint32), ("elapsed_ns", C.c_uint64), ("completion_marker", C.c_uint32),
                ("unwritten_words", C.c_uint32), ("kernel_cache_hit", C.c_bool), ("gpr_count", C.c_uint32),
                ("insn_count", C.c_uint32), ("threads_per_cta", C.c_uint32), ("ctas_last_launch", C.c_uint32),
                ("target_chip", C.c_char * 64), ("sm_architecture", C.c_uint32), ("call_ns", C.c_uint64)]


class AttnInfo(C.Structure):
    _fields_ = [("chip_calls", C.c_uint32), ("elapsed_ns", C.c_uint64), ("call_ns", C.c_uint64),
                ("completion_marker", C.c_uint32), ("unwritten_words", C.c_uint32), ("gpr_count", C.c_uint32),
                ("insn_count", C.c_uint32), ("kernel_cache_hit", C.c_bool), ("ctas_last_launch", C.c_uint32),
                ("threads_per_cta", C.c_uint32), ("target_chip", C.c_char * 64), ("sm_architecture", C.c_uint32),
                ("kv_bytes_staged", C.c_uint64), ("kv_bytes_naive", C.c_uint64), ("kv_blocks_logical", C.c_uint32),
                ("kv_blocks_unique", C.c_uint32), ("q_bytes", C.c_uint64), ("tab_bytes", C.c_uint64),
                ("out_bytes", C.c_uint64), ("kv_source", C.c_uint32)]


class KvLayout(C.Structure):
    _fields_ = [("block_stride_bytes", C.c_uint64), ("layer_stride_bytes", C.c_uint64),
                ("kv_plane_stride_bytes", C.c_uint64), ("token_stride_bytes", C.c_uint64),
                ("head_stride_bytes", C.c_uint64), ("pool_bytes", C.c_uint64), ("num_blocks", C.c_uint32),
                ("num_layers", C.c_uint32), ("block_size", C.c_uint32)]


class GemmExInfo(C.Structure):
    _fields_ = [("route", C.c_uint32), ("core_type", C.c_uint32), ("core_calls", C.c_uint64), ("host_fixed", C.c_uint64)]


class GemmExArgs(C.Structure):
    _fields_ = [("op_a", C.c_int), ("op_b", C.c_int), ("m", C.c_uint64), ("n", C.c_uint64), ("k", C.c_uint64),
                ("batch", C.c_uint64), ("a_type", C.c_int), ("b_type", C.c_int), ("c_type", C.c_int),
                ("a", C.c_void_p), ("lda", C.c_uint64), ("stride_a", C.c_uint64),
                ("b", C.c_void_p), ("ldb", C.c_uint64), ("stride_b", C.c_uint64),
                ("c", C.c_void_p), ("ldc", C.c_uint64), ("stride_c", C.c_uint64),
                ("alpha", C.c_float), ("beta", C.c_float), ("compute", C.c_int), ("scale", C.c_int),
                ("scale_a", C.c_void_p), ("scale_b", C.c_void_p), ("scale_stride_a", C.c_uint64),
                ("scale_stride_b", C.c_uint64), ("block_m", C.c_uint32), ("block_n", C.c_uint32),
                ("block_k", C.c_uint32), ("info", C.POINTER(GemmExInfo))]


EW_RMSNORM, EW_ROPE, EW_SWIGLU = 3, 4, 5
ATTN_NO_MAX, ATTN_KV_HEAD, ATTN_SLOT, ATTN_NO_RESCALE, ATTN_Q_ROW, ATTN_OUT_ROW = 1, 2, 3, 4, 5, 6  # attention_api.h:217-226
PRED_MIN = 4.0           # a mutant runs on a case only if the host predicts gate ratio >= this (NO_RESCALE)
GEMM_DT_F32, GEMM_DT_BF16 = 0, 1
LIB = None


def info_dict(info):
    d = {}
    for f, _ in info._fields_:
        v = getattr(info, f)
        d[f] = v.decode() if isinstance(v, bytes) else v
    return d


# ------------------------------------------------------------------ chip error bounds (docs/TOLERANCE.md, GPU tier)
def bound_rms(dim):
    gs = g32(dim // 128 + 11)
    return (1 + gs / (2 * (1 - gs))) * (1 + NEWTON) * (1 + U) ** 2 - 1


def bound_silu(gabs):
    e = torch.exp(gabs * g32(2)) * (1 + E_EX2) - 1
    return (1 + U) ** 2 * (1 + E_RCP) * (1 + e / (1 - e)) / (1 - U) - 1


def bound_attn(ctx, hd, splits, lam):
    """Relative bound (x A = p @ |v|) for one output row of the online-softmax kernels; ctx, lam tensors."""
    ctx = ctx.double()
    nch = torch.ceil(ctx / hd)
    ep = torch.exp(2 * lam * g32(hd + 3)) * (1 + E_EX2) - 1
    ea = torch.exp(2 * lam * U) * (1 + E_EX2) - 1
    ew = torch.exp(2 * lam * U) * (1 + E_HEXP) - 1 if splits > 1 else torch.zeros_like(lam)
    eta = (1 + ep) * (1 + ea) ** (nch - 1) * (1 + ew) - 1
    gs = g32(ctx + 2 * nch + 2 * splits + 8)
    alpha = (1 + eta) * (1 + gs) - 1
    phi = (1 + E_RCP) * (1 + U) - 1
    return (phi + alpha * (2 + phi)) / (1 - alpha)


def own_log2split(ctx, hd):
    x, l = (ctx + hd - 1) // hd, 0
    while l < 4 and (2 << l) <= x:
        l += 1
    return l


def gemm_steps(k):
    parts = -(-k // 1808) + -(-k // 16384)
    return 9 * (-(-k // 16) + parts) + parts


# ------------------------------------------------------------------ cases: each returns a dict with the chip call
def case_rmsnorm(g):
    rows, dim = ri(g, 1, 8), pick(g, [128, 256, 384, 640, 1024, 2048, 4096, 128 * ri(g, 1, 128), 16384])
    x, w, eps = rn(g, rows, dim, scale=float(torch.rand(1, generator=g)) * 8 + 0.1), rn(g, dim), 2.0 ** -20
    xd = x.double()
    ref = xd / torch.sqrt((xd * xd).mean(-1, keepdim=True) + eps) * w.double()

    def chip():
        out, info = torch.empty_like(x), EwInfo()
        rc = LIB.omega_gpu_rmsnorm_f32(rows, dim, ptr(x), ptr(w), C.c_float(eps), ptr(out), C.byref(info))
        return rc, out, info
    return dict(op="rmsnorm", shape=[rows, dim], host=od.rms_norm(x, w, eps), ref=ref, chip=chip,
                t_chip=bound_rms(dim) * ref.abs() + FLOOR, t_host=host_tol(ref, dim, ref.abs()),
                t_ref=ref_tol(dim, ref.abs()), mutants=[("ew", EW_RMSNORM, {})])


def case_rope(g):
    heads, hd = ri(g, 1, 8), pick(g, [2, 64, 128, 2 * ri(g, 1, 64), 2 * ri(g, 1, 1024), 2048])
    maxp = ri(g, 1, 64)
    x, cos, sin = rn(g, heads, hd), rn(g, maxp, hd // 2), rn(g, maxp, hd // 2)
    p = ri(g, 0, maxp - 1)
    xd, c, s = x.double(), cos.double()[p], sin.double()[p]
    x1, x2 = xd[:, : hd // 2], xd[:, hd // 2:]
    ref = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    mag = torch.cat([x1.abs() * c.abs() + x2.abs() * s.abs(), x2.abs() * c.abs() + x1.abs() * s.abs()], -1)
    ch, sh = cos[p].contiguous(), sin[p].contiguous()
    host = od.rope(x.view(1, heads, 1, hd), cos, sin, torch.tensor([[p]]), rot=hd, interleaved=False).view(heads, hd)

    def chip():
        out, info = torch.empty_like(x), EwInfo()
        rc = LIB.omega_gpu_rope_f32(heads, hd, ptr(x), ptr(ch), ptr(sh), ptr(out), C.byref(info))
        return rc, out, info
    return dict(op="rope", shape=[heads, hd, p], host=host, ref=ref, chip=chip, t_chip=g32(2) * mag + FLOOR,
                t_host=host_tol(ref, 2, mag), t_ref=ref_tol(2, mag), mutants=[("ew", EW_ROPE, {})])


def case_swiglu(g, silu=False):
    n = pick(g, [1, 127, 128, 129, ri(g, 1, 5000), ri(g, 1, 200000)])
    gate = rn(g, n, scale=pick(g, [1.0, 4.0, 16.0])).clamp(-80, 80)
    up = torch.ones(n) if silu else rn(g, n, scale=4)
    gd, ud = gate.double(), up.double()
    ref = F.silu(gd) * ud
    if silu:
        host, n_h, mag = od.elementwise(od.SILU, gate), 1, ref.abs() + gd.abs()
    else:
        host, n_h, mag = od.elementwise(od.SWIGLU, gate, up), 1, ref.abs() + gd.abs() * (1 + ud.abs())

    def chip():
        out, info = torch.empty_like(gate), EwInfo()
        rc = LIB.omega_gpu_swiglu_f32(n, ptr(gate), ptr(up), ptr(out), C.byref(info))
        return rc, out, info
    return dict(op="silu" if silu else "swiglu", shape=[n], host=host, ref=ref, chip=chip,
                t_chip=bound_silu(gd.abs()) * ref.abs() + FLOOR, t_host=host_tol(ref, n_h, mag), t_ref=ref_tol(n_h, mag),
                mutants=[("ew", EW_SWIGLU, {})])


def attn_stats(qd, kd, vd, scale, mask=None):
    """qd [H,Sq,D], kd/vd [H,Skv,D] float64 (kv heads already repeated). Returns ref, A = p@|v|, lam [H,Sq,1]."""
    lg = qd @ kd.transpose(-1, -2) * scale
    lam = qd.abs() @ kd.abs().transpose(-1, -2) * scale
    if mask is not None:
        lg = lg.masked_fill(mask, float("-inf"))
        lam = lam.masked_fill(mask, 0.0)
    p = torch.softmax(lg, -1)
    return p @ vd, p @ vd.abs(), lam.amax(-1, keepdim=True)


def nores_output(qd, kd, vd, scale, hd, mask=None):
    """Host model of CUDApp's NO_RESCALE mutant with one CTA per head (split 1): the chunk loop walks HD tokens at a
    time from token 0, rescales L by alpha = exp(M_old - M_new) but leaves the accumulator unscaled
    (src/omega_gpu_attention_api.c:571-572). Same shapes as attn_stats; returns the mutant's output."""
    lg = qd @ kd.transpose(-1, -2) * scale
    if mask is not None:
        lg = lg.masked_fill(mask, float("-inf"))
    m = torch.full(lg.shape[:-1] + (1,), float("-inf"), dtype=lg.dtype)
    L = torch.zeros_like(m)
    acc = torch.zeros(lg.shape[:-1] + (vd.shape[-1],), dtype=lg.dtype)
    for c0 in range(0, lg.shape[-1], hd):
        lc = lg[..., c0:c0 + hd]
        mn = torch.maximum(m, lc.amax(-1, keepdim=True))
        safe = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)   # a row with every key masked so far
        L = L * torch.exp(m - safe) + torch.exp(lc - safe).sum(-1, keepdim=True)
        acc = acc + torch.exp(lc - safe) @ vd[..., c0:c0 + hd, :]
        m = mn
    return acc / L


def gate_pred(mut, host, t_chip, t_host):
    """max |mutant - host| / (T_chip + T_host): the gate ratio the mutant is predicted to reach."""
    return float(((mut - host.double()).abs() / (t_chip + t_host)).max())


def row_mutants(nq, nkv):
    """Head-wiring mutants, on the shapes CUDApp's own tests run them (tests/gpu_attention_test.c:379,524)."""
    m = [("attn", ATTN_KV_HEAD, {})] if nkv >= 2 else []
    if nq >= 2:   # both swap rows h and h ^ 1: num_q_heads even (omega_gpu_attention_api.h:222-224)
        m += [("attn", ATTN_Q_ROW, {}), ("attn", ATTN_OUT_ROW, {})]
    return m


def visible(pred):
    """A mutant joins a case only if the host predicts it leaves the gate by PRED_MIN (no equivalent mutants)."""
    return pred >= PRED_MIN


def case_softmax(g):
    """softmax over c <= 64 logits through the attention kernel: D = 64, K = V = I[:c], q = 8 x (scale 1/8 exact)."""
    c = ri(g, 1, 64)
    x = rn(g, c, scale=float(torch.rand(1, generator=g)) * 8 + 0.1)
    if ri(g, 0, 1):
        x = x * 300   # far past exp overflow: only a max-subtracted softmax survives (the NO_MAX mutant overflows)
    q = torch.zeros(64)
    q[:c] = 8 * x
    eye = torch.zeros(c, 1, 64)
    eye[:, 0, :c] = torch.eye(c)
    xd = x.double()
    r = torch.softmax(xd, -1)
    ref = torch.zeros(64, dtype=torch.float64)
    ref[:c] = r
    span = (xd - xd.amax()).abs()
    host = torch.zeros(64)
    host[:c] = od.softmax(x.view(1, c)).view(c)
    mag = torch.zeros(64, dtype=torch.float64)
    mag[:c] = r * (1 + span) + r.amax() * span.amax()
    lam = xd.abs().amax().view(1)
    tc = bound_attn(torch.tensor([c]), 64, 1, lam) * ref.abs() + FLOOR   # A = p @ |e_d| = p_d = |ref|

    def chip():
        out, info = torch.empty(64), AttnInfo()
        rc = LIB.omega_gpu_gqa_attention_f32(ptr(q), ptr(eye), ptr(eye), c, 1, 1, 64, ptr(out), C.byref(info))
        return rc, out, info
    # NO_MAX is exact softmax unless 2^(x log2 e) leaves the f32 range (CUDApp tests/gpu_attention_test.c:523)
    lo2 = float(xd.amax()) * math.log2(math.e)
    muts = [("attn", ATTN_NO_MAX, dict(pred=math.inf))] if lo2 > 128 or lo2 < -126 else []
    return dict(op="softmax_via_attn", shape=[c, round(float(x.abs().max()), 1)], host=host, ref=ref, chip=chip,
                t_chip=tc, t_host=host_tol(ref, c, mag), t_ref=ref_tol(c, mag), mutants=muts)


def heads(g, max_q):
    while True:
        nkv, rep = pick(g, [1, 2, 4, 8]), pick(g, [1, 2, 4, 8])
        if nkv * rep <= max_q:
            return nkv, rep


def case_gqa(g):
    hd = pick(g, [64, 128])
    nkv, rep = heads(g, 64)
    nq = nkv * rep
    ctx = pick(g, [1, 2, hd - 1, hd, hd + 1, ri(g, 1, 512), ri(g, 1, 512), ri(g, 1, 4096), 4096])
    q = rn(g, nq, hd, scale=float(torch.rand(1, generator=g)) * 2 + 0.1)
    k, v = rn(g, ctx, nkv, hd), rn(g, ctx, nkv, hd)
    sc = hd ** -0.5
    kd, vd = (t.double().permute(1, 0, 2).repeat_interleave(rep, 0) for t in (k, v))
    ref, A, lam = attn_stats(q.double()[:, None], kd, vd, sc)
    ref, A, lam = ref[:, 0], A[:, 0], lam[:, 0]
    host = od.attention(q.view(1, nq, 1, hd), k.permute(1, 0, 2)[None].contiguous(), v.permute(1, 0, 2)[None].contiguous(),
                        causal=False).view(nq, hd)
    mag = A * (1 + 2 * lam) + ref.abs()
    tc = bound_attn(torch.full((nq, 1), ctx), hd, 1, lam) * A + FLOOR

    def chip():
        out, info = torch.empty(nq, hd), AttnInfo()
        rc = LIB.omega_gpu_gqa_attention_f32(ptr(q), ptr(k), ptr(v), ctx, nq, nkv, hd, ptr(out), C.byref(info))
        return rc, out, info
    t_host = host_tol(ref, hd + ctx, mag)
    muts = row_mutants(nq, nkv)
    pred = gate_pred(nores_output(q.double()[:, None], kd, vd, sc, hd)[:, 0], host, tc, t_host)
    if visible(pred):
        muts.append(("attn", ATTN_NO_RESCALE, dict(pred=pred)))
    return dict(op="attention_decode", shape=[nq, nkv, hd, ctx], host=host, ref=ref, chip=chip, t_chip=tc,
                t_host=t_host, t_ref=ref_tol(hd + ctx, mag), mutants=muts)


def case_rows(g):
    hd = pick(g, [64, 128])
    nkv, rep = heads(g, 16)
    nq, rows = nkv * rep, pick(g, [1, 2, ri(g, 1, 64), ri(g, 64, 200)])
    q = rn(g, rows, nq, hd, scale=float(torch.rand(1, generator=g)) * 2 + 0.1)
    k, v = rn(g, rows, nkv, hd), rn(g, rows, nkv, hd)
    kd, vd = (t.double().permute(1, 0, 2).repeat_interleave(rep, 0) for t in (k, v))
    mask = torch.ones(rows, rows, dtype=torch.bool).triu(1)
    ref, A, lam = attn_stats(q.double().permute(1, 0, 2), kd, vd, hd ** -0.5, mask)   # [nq, rows, hd]
    host = od.attention(q.permute(1, 0, 2)[None].contiguous(), k.permute(1, 0, 2)[None].contiguous(),
                        v.permute(1, 0, 2)[None].contiguous(), causal=True)[0]
    ctx = torch.arange(1, rows + 1).view(1, rows, 1).expand(nq, rows, 1)
    tc = bound_attn(ctx, hd, 1, lam) * A + FLOOR
    mag = A * (1 + 2 * lam) + ref.abs()
    perm = (1, 0, 2)

    def chip():
        out, info = torch.empty(rows, nq, hd), AttnInfo()
        rc = LIB.omega_gpu_gqa_attention_rows_f32(ptr(q), ptr(k), ptr(v), 0, rows, nq, nkv, hd, ptr(out), C.byref(info))
        return rc, out.permute(*perm), info
    t_host = host_tol(ref, hd + rows, mag)
    muts = row_mutants(nq, nkv)
    pred = gate_pred(nores_output(q.double().permute(1, 0, 2), kd, vd, hd ** -0.5, hd, mask), host, tc, t_host)
    if visible(pred):
        muts.append(("attn", ATTN_NO_RESCALE, dict(pred=pred)))
    return dict(op="attention_prefill_causal", shape=[nq, nkv, hd, rows], host=host, ref=ref, chip=chip, t_chip=tc,
                t_host=t_host, t_ref=ref_tol(hd + rows, mag), mutants=muts)


def case_paged(g, single=False):
    hd = pick(g, [64, 128])
    nkv, rep = heads(g, 32)
    nq, B, bs = nkv * rep, 1 if single else ri(g, 1, 4), pick(g, [8, 16, 32, 64])
    maxb = ri(g, 1, 2048 // bs)
    nblk = B * maxb + ri(g, 0, 4)
    lens = torch.randint(1, maxb * bs + 1, (B,), generator=g)
    table = torch.stack([torch.randperm(nblk, generator=g)[:maxb] for _ in range(B)])
    q = rn(g, B, nq, hd, scale=float(torch.rand(1, generator=g)) * 2 + 0.1)
    kc, vc = (rn(g, nblk, bs, nkv, hd).bfloat16().float() for _ in range(2))
    host = od.paged_decode(q, kc, vc, table, lens)
    refs, As, lams, nores = [], [], [], []
    for i in range(B):
        kb, vb = (t[table[i]].reshape(-1, nkv, hd)[: lens[i]].double().permute(1, 0, 2).repeat_interleave(rep, 0)
                  for t in (kc, vc))
        r, a, lm = attn_stats(q[i].double()[:, None], kb, vb, hd ** -0.5)
        refs.append(r[:, 0]), As.append(a[:, 0]), lams.append(lm[:, 0])
        nores.append(nores_output(q[i].double()[:, None], kb, vb, hd ** -0.5, hd)[:, 0])
    ref, A, lam = torch.stack(refs), torch.stack(As), torch.stack(lams)
    splits = 1 << max(own_log2split(int(n), hd) for n in lens)
    tc = bound_attn(lens.view(B, 1, 1).expand(B, nq, 1), hd, splits, lam) * A + FLOOR
    mag = A * (1 + 2 * lam) + ref.abs()
    pool = torch.stack([kc, vc], 1).bfloat16().contiguous()   # [block][plane][token][head][d], one layer
    lay = KvLayout()
    lay.head_stride_bytes, lay.token_stride_bytes = hd * 2, nkv * hd * 2
    lay.kv_plane_stride_bytes = bs * lay.token_stride_bytes
    lay.layer_stride_bytes = lay.block_stride_bytes = 2 * lay.kv_plane_stride_bytes
    lay.pool_bytes, lay.num_blocks, lay.num_layers, lay.block_size = nblk * lay.block_stride_bytes, nblk, 1, bs
    tab32, len32 = table.to(torch.int32).contiguous(), lens.to(torch.int32).contiguous()
    ids = table[0, : -(-int(lens[0]) // bs)].to(torch.int32).contiguous()   # single call: uint32 block ids

    def chip():
        out, info = torch.empty(B, nq, hd), AttnInfo()
        if single:
            rc = LIB.omega_gpu_paged_attention_bf16(ptr(q), ptr(pool), C.byref(lay), ptr(ids), ids.numel(), int(lens[0]),
                                                    0, nq, nkv, hd, ptr(out), C.byref(info))
        else:
            rc = LIB.omega_gpu_paged_attention_batch_bf16(ptr(q), ptr(pool), C.byref(lay), ptr(tab32), ptr(len32), maxb,
                                                          B, 0, nq, nkv, hd, ptr(out), C.byref(info))
        return rc, out, info
    n_h = hd + maxb * bs
    t_host = host_tol(ref, n_h, mag)
    muts = row_mutants(nq, nkv)
    if int(lens.min()) >= 2:   # one token: softmax weight 1 whatever K says, so a K/V slot mismatch is invisible
        muts.append(("attn", ATTN_SLOT, {}))
    # NO_RESCALE runs with one CTA per head as CUDApp's own test does (cut J gives each split one chunk otherwise)
    pred = gate_pred(torch.stack(nores), host, tc, t_host)
    if visible(pred):
        muts.append(("attn", ATTN_NO_RESCALE, dict(pred=pred, split0=True)))
    return dict(op="paged_decode_single" if single else "paged_decode_batch",
                shape=[B, nq, nkv, hd, bs, maxb, int(lens.max()), splits], host=host, ref=ref,
                chip=chip, t_chip=tc, t_host=t_host, t_ref=ref_tol(n_h, mag), mutants=muts)


def gemm_call(a, a_t, b, b_t, m, n, k, op_b, ldb, c, beta):
    info = GemmExInfo()
    args = GemmExArgs(op_a=0, op_b=op_b, m=m, n=n, k=k, batch=1, a_type=a_t, b_type=b_t, c_type=GEMM_DT_F32,
                      a=a.data_ptr(), lda=0, b=b.data_ptr(), ldb=ldb, c=c.data_ptr(), ldc=0, alpha=1.0, beta=beta,
                      compute=0, scale=0, info=C.pointer(info))
    return LIB.omega_gpu_gemm_ex(C.byref(args)), info


def case_matmul(g):
    M, N = pick(g, [1, 16, ri(g, 1, 70)]), pick(g, [1, 8, ri(g, 1, 300)])
    K = pick(g, [1, 15, 16, 17, ri(g, 1, 300), ri(g, 1, 2048), 1808, 1809])
    A, Bm = rn(g, M, K), rn(g, K, N)
    tb, has_bias = bool(ri(g, 0, 1)), ri(g, 0, 3) > 0
    bv = rn(g, N) if has_bias else torch.zeros(N)
    Bs = Bm.t().contiguous() if tb else Bm
    host = od.matmul(A, Bs, bv if has_bias else None, trans_b=tb)
    ref = A.double() @ Bm.double() + bv.double()
    sab = A.double().abs() @ Bm.double().abs()
    sapb = A.double().abs().sum(1, keepdim=True) + Bm.double().abs().sum(0, keepdim=True)
    acc = ((1 + C_HMMA) ** gemm_steps(K) - 1) * sab + 2.0 ** -124 * sapb + K * 2.0 ** -140
    tc = acc + 2.0 ** -52 * (sab + bv.double().abs()) + U * (ref.abs() + acc) + TINY
    mag = sab + bv.double().abs()

    def chip():
        c = bv.expand(M, N).contiguous()
        rc, info = gemm_call(A, GEMM_DT_F32, Bs, GEMM_DT_F32, M, N, K, 1 if tb else 0, K if tb else N, c,
                             1.0 if has_bias else 0.0)
        return rc, c, info
    mut = 2 if tb and N > 1 and K > 1 else (1 if K <= 256 else None)
    return dict(op="matmul", shape=[M, N, K, int(tb), int(has_bias)], host=host, ref=ref, chip=chip, t_chip=tc,
                t_host=host_tol(ref, K, mag), t_ref=ref_tol(K, mag), mutants=[("gemm", mut, {})] if mut else [])


GROUPS = {"ew": [case_rmsnorm, case_rope, case_swiglu, lambda g: case_swiglu(g, silu=True)],
          "attn": [case_softmax, case_gqa, case_rows, case_paged, lambda g: case_paged(g, single=True)],
          "gemm": [case_matmul]}


# ------------------------------------------------------------------ the CUDApp#139 per-step bound probe (host model: scratch)
def probe_hmma():
    """bf16 native gemm_ex, k = 1808 (one k part): a = [1, t x 1807], b = 1, t = 0x1.fcp-26. The chip-exact HMMA
    model {16,2,RZ,nominal} truncates every t at alignment to the running 1.0, so the result is 1.0 and the error is
    1807 t = 5.343e-5, above the old stated bound (steps+4) 2^-22 sab = 2.843e-5 and below ((1+c)^steps - 1) sab."""
    k, t = 1808, float.fromhex("0x1.fcp-26")
    a = torch.full((1, k), t)
    a[0, 0] = 1.0
    a16, b16 = a.bfloat16().contiguous(), torch.ones(k, 1).bfloat16().contiguous()
    assert torch.equal(a16.float(), a)
    c = torch.zeros(1, 1)
    rc, info = gemm_call(a16, GEMM_DT_BF16, b16, GEMM_DT_BF16, 1, 1, k, 0, 1, c, 0.0)
    ref = 1.0 + (k - 1) * t
    parts = 2
    steps = -(-k // 16) + parts
    err = abs(ref - float(c[0, 0]))
    return dict(rc=rc, got=float(c[0, 0]).hex(), ref=ref, err=err, old_bound=(steps + 4) * 2.0 ** -22 * ref + U * ref,
                new_bound=((1 + C_HMMA) ** steps - 1) * ref + U * ref, model_prediction="0x1.0000000000000p+0",
                info=info_dict(info))


class GemmInfo(C.Structure):   # OmegaGpuGemmInfo, src/omega_gpu_gemm.h:93-98
    _fields_ = [("pack_ns", C.c_uint64), ("main_ns", C.c_uint64), ("call_ns", C.c_uint64), ("gpr", C.c_uint32 * 3)]


def gemm_chip_witness():
    """OmegaGemmExInfo carries no device field, so the receipt also runs the core gemm_ex calls by default (chip_core ->
    omega_gpu_gemm_ld, src/omega_gpu_gemm_ex.c:20-22; this harness never calls omega_gpu_gemm_ex_set_backend) directly,
    with its kernel timings and register counts, and checks the product bit-for-bit against gemm_ex native bf16."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(32, 64, generator=g).bfloat16().contiguous()
    b = torch.randn(64, 16, generator=g).bfloat16().contiguous()
    c1, c2, info = torch.zeros(32, 16), torch.zeros(32, 16), GemmInfo()
    rc1 = LIB.omega_gpu_gemm_ld(0, 0, 32, 16, 64, ptr(a), C.c_uint64(0), ptr(b), C.c_uint64(0), ptr(c1), C.byref(info))
    rc2, xinfo = gemm_call(a, GEMM_DT_BF16, b, GEMM_DT_BF16, 32, 16, 64, 0, 16, c2, 0.0)
    ok = rc1 == 0 and rc2 == 0 and info.main_ns > 0 and torch.equal(c1, c2)
    return dict(ok=ok, rc_core=rc1, rc_ex=rc2, pack_ns=info.pack_ns, main_ns=info.main_ns, call_ns=info.call_ns,
                gpr=list(info.gpr), bit_identical=bool(torch.equal(c1, c2)), ex_info=info_dict(xinfo))

# ------------------------------------------------------------------ chip session discipline
def window_open():
    try:
        lines = [ln for ln in open(BOARD) if re.match(r"^[0-9TZ:.-]+ Main: WINDOW-(START|END|ABORTED)", ln)]
    except OSError:
        return True
    return bool(lines) and "WINDOW-START" in lines[-1]


def board(msg):
    with open(BOARD, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {LANE}: {msg}\n")


def health():
    r = subprocess.run(["curl", "-s", "-m", "5", "127.0.0.1:18091/health"], capture_output=True, text=True)
    return r.stdout.strip() or f"rc={r.returncode}"


def judge(case, rc, y):
    if rc != 0:
        return dict(rc=rc, gate_viol=-1, exact_viol=-1, gate_ratio=float("inf"), exact_ratio=float("inf"))
    y = y.double().reshape(case["ref"].shape)
    out = {"rc": 0}
    for key, base, tol in (("gate", case["host"].double().reshape(case["ref"].shape), case["t_chip"] + case["t_host"]),
                           ("exact", case["ref"], case["t_chip"] + case["t_ref"])):
        err = (y - base).abs()
        out[f"{key}_viol"] = int((~(err <= tol)).sum())
        out[f"{key}_ratio"] = float((err / tol).nan_to_num(float("inf")).max())
    return out


MUTANT_SETTER = {"ew": "omega_gpu_elementwise_test_set_mutant", "attn": "omega_gpu_attention_test_set_mutant",
                 "gemm": "omega_gpu_gemm_ex_test_set_mutant"}


def set_mutant(kind, m):
    getattr(LIB, MUTANT_SETTER[kind])(m)


def main():
    global LIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--mutant-cases", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cudapp-sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", action="store_true")
    a = ap.parse_args()
    LIB = C.CDLL(os.path.abspath(a.lib))
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    cases = [mk(g) for _ in range(a.cases) for mk in GROUPS[a.group]]
    print(f"prepared {len(cases)} cases on the host in {time.time() - t0:.1f}s", flush=True)
    for kind in {m[0] for cs in cases for m in cs["mutants"]}:
        getattr(LIB, MUTANT_SETTER[kind])   # an absent hook fails here, before any hold
    rec = dict(lane=LANE, group=a.group, seed=a.seed, cases_per_op=a.cases, cudapp_sha=a.cudapp_sha,
               lib=os.path.abspath(a.lib), lib_sha256=hashlib.sha256(open(a.lib, "rb").read()).hexdigest(),
               harness_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
               torch=torch.__version__, host=os.uname().nodename, kernel=os.uname().release,
               driver=open("/proc/driver/nvidia/version").readline().strip(), ops={}, mutants={}, cases=[])
    if not a.hold:
        print("dry run: no --hold, chip not touched")
        return 0
    if window_open():
        print("REFUSED: Main WINDOW-START without END/ABORTED on BOARD; no hold taken")
        return 2
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666)
    deadline = time.time() + 900
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() > deadline:
                print("REFUSED: lock not free within 900 s")
                return 2
            time.sleep(2)
    if window_open():
        os.close(fd)
        print("REFUSED: window opened while waiting")
        return 2
    rec["health_before"] = health()
    rec["utc_start"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    board(f"GPU-START omega-dnn GPU tier group={a.group} (CUDApp kernels, no model) health={rec['health_before'][:60]}")
    t_hold = time.time()
    bad = 0
    try:
        for cs in cases:
            rc, y, info = cs["chip"]()
            j = judge(cs, rc, y)
            row = rec["ops"].setdefault(cs["op"], dict(cases=0, gate_viol=0, exact_viol=0, rc_fail=0, gate_worst=0.0,
                                                        exact_worst=0.0, chip_calls=0, elapsed_ns=0))
            row["cases"] += 1
            row["rc_fail"] += rc != 0
            row["gate_viol"] += max(j["gate_viol"], 0)
            row["exact_viol"] += max(j["exact_viol"], 0)
            row["gate_worst"] = max(row["gate_worst"], j["gate_ratio"])
            row["exact_worst"] = max(row["exact_worst"], j["exact_ratio"])
            inf = info_dict(info)
            row["chip_calls"] += inf.get("chip_calls", inf.get("core_calls", 0))
            row["elapsed_ns"] += inf.get("elapsed_ns", 0)
            for f in ("target_chip", "sm_architecture", "route", "core_type"):
                if f in inf:
                    row[f] = inf[f]
            rec["cases"].append(dict(op=cs["op"], shape=cs["shape"], **j))
            bad += (rc != 0) + (j["gate_viol"] != 0)
        # negative controls: CUDApp's own mutants must fail the gate. A case only carries a mutant the host
        # predicts to be visible on it (attention: nores_output; docs/TOLERANCE.md, GPU tier).
        done = {}
        for cs in cases:
            for kind, m, opt in cs["mutants"]:
                key = f"{cs['op']}:{kind}:{m}"
                if done.get(key, 0) >= a.mutant_cases:
                    continue
                done[key] = done.get(key, 0) + 1
                if opt.get("split0"):
                    LIB.omega_gpu_attention_test_set_split(0)
                set_mutant(kind, m)
                try:
                    rc, y, _ = cs["chip"]()
                finally:
                    set_mutant(kind, 0)
                    if opt.get("split0"):
                        LIB.omega_gpu_attention_test_set_split(-1)
                j = judge(cs, rc, y)
                killed = rc == 0 and j["gate_viol"] > 0
                row = rec["mutants"].setdefault(key, dict(runs=0, killed=0, refused=0, survived=0, runs_detail=[]))
                row["runs"] += 1
                row["killed"] += killed
                row["refused"] += rc != 0
                row["survived"] += rc == 0 and not killed
                row["runs_detail"].append(dict(shape=cs["shape"], rc=rc, gate_viol=j["gate_viol"],
                                               gate_ratio=j["gate_ratio"], predicted_ratio=opt.get("pred"),
                                               split0=bool(opt.get("split0"))))
                bad += rc == 0 and not killed
        if a.group == "gemm":
            rec["probe_cudapp139_hmma_bound"] = probe_hmma()
            rec["gemm_chip_witness"] = gemm_chip_witness()
            bad += not rec["gemm_chip_witness"]["ok"]
    except Exception as e:   # recorded in the receipt as a failure; the hold is still released below
        rec["error"] = repr(e)
        bad += 1
    finally:
        rec["hold_s"] = round(time.time() - t_hold, 1)
        rec["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["health_after"] = health()
        board(f"GPU-END group={a.group} hold={rec['hold_s']}s health={rec['health_after'][:60]}")
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    rec["result"] = "PASS" if bad == 0 else "FAIL"
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    for name, r in rec["ops"].items():
        print(f"{name:26s} cases={r['cases']:3d} gate_worst={r['gate_worst']:.3g} gate_viol={r['gate_viol']} "
              f"exact_worst={r['exact_worst']:.3g} exact_viol={r['exact_viol']} rc_fail={r['rc_fail']} "
              f"chip={r.get('target_chip', '')} sm={r.get('sm_architecture', '')}")
    for key, r in rec["mutants"].items():
        print(f"MUTANT {key:34s} runs={r['runs']} killed={r['killed']} refused={r['refused']} survived={r['survived']}")
    if "probe_cudapp139_hmma_bound" in rec:
        print("PROBE", json.dumps(rec["probe_cudapp139_hmma_bound"]))
    if "gemm_chip_witness" in rec:
        print("WITNESS", json.dumps(rec["gemm_chip_witness"]))
    print(f"RESULT {rec['result']} hold={rec['hold_s']}s")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
