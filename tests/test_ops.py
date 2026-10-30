"""Randomized check of every omega-dnn op against torch float64 (docs/TOLERANCE.md).

Usage: python tests/test_ops.py [cases_per_op] [seed]
Prints one line per op (cases, worst err/tol, violations) and exits 1 on any violation.
Each element must satisfy |y - ref| <= U32*|ref| + 2*gamma(n + 64)*M + 2^-149, with n and M per op.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import omega_dnn as od  # noqa: E402

U32 = 2.0 ** -24
U64 = 2.0 ** -53
TINY = 2.0 ** -149
# Negative control only: OMEGA_DNN_TOL_SCALE=0.5 halves every bound and must produce violations.
TOL_SCALE = float(os.environ.get("OMEGA_DNN_TOL_SCALE", "1"))


def gamma(n):
    return n * U64 / (1 - n * U64)


class Checker:
    def __init__(self):
        self.rows = {}

    def check(self, name, y, ref, n, mag):
        y = y.double().reshape(ref.shape)
        tol = (U32 * ref.abs() + 2 * gamma(n + 64) * mag + TINY) * TOL_SCALE
        err = (y - ref).abs()
        bad = int((~(err <= tol)).sum())
        ratio = float((err / tol).max()) if err.numel() else 0.0
        c, w, b = self.rows.get(name, (0, 0.0, 0))
        self.rows[name] = (c + 1, max(w, ratio), b + bad)

    def report(self):
        total = 0
        for name, (c, w, b) in self.rows.items():
            print(f"{name:14s} cases={c:4d} worst_err/tol={w:.3f} violations={b}")
            total += b
        print(f"TOTAL ops={len(self.rows)} violations={total}")
        return total


def ri(g, lo, hi):
    return int(torch.randint(lo, hi + 1, (1,), generator=g))


def rn(g, *shape, scale=1.0):
    return torch.randn(*shape, generator=g) * scale


def run(cases, seed):
    g = torch.Generator().manual_seed(seed)
    ck = Checker()
    for _ in range(cases):
        # norms and softmax
        R, C = ri(g, 1, 6), ri(g, 1, 300)
        x = rn(g, R, C, scale=float(torch.rand(1, generator=g)) * 8 + 0.1)
        w, b = rn(g, C), rn(g, C)
        xd = x.double()
        r = xd / torch.sqrt((xd * xd).mean(-1, keepdim=True) + 1e-6) * w.double()
        ck.check("rms_norm", od.rms_norm(x, w, 1e-6), r, C, r.abs())
        r = F.layer_norm(xd, (C,), w.double(), b.double(), 1e-5)
        sd = torch.sqrt(xd.var(-1, unbiased=False, keepdim=True) + 1e-5)
        mag = r.abs() + (xd.abs().amax(-1, keepdim=True) / sd) * w.double().abs() + b.double().abs()
        ck.check("layer_norm", od.layer_norm(x, w, b), r, C, mag)
        r = torch.softmax(xd, -1)
        span = (xd - xd.amax(-1, keepdim=True)).abs()
        ck.check("softmax", od.softmax(x), r, C, r * (1 + span) + r.amax(-1, keepdim=True) * span.amax(-1, keepdim=True))
        xl = x * 300  # logits far past exp overflow: only a max-subtracted softmax survives
        r = torch.softmax(xl.double(), -1)
        span = (xl.double() - xl.double().amax(-1, keepdim=True)).abs()
        ck.check("softmax_large", od.softmax(xl), r, C, r * (1 + span) + r.amax(-1, keepdim=True) * span.amax(-1, keepdim=True))
        # elementwise
        n = ri(g, 1, 5000)
        a, bb = rn(g, n, scale=4), rn(g, n, scale=4)
        ad, bd = a.double(), bb.double()
        for nm, op, kw, ref in (
            ("silu", od.SILU, {}, F.silu(ad)),
            ("gelu_erf", od.GELU, {}, F.gelu(ad)),
            ("gelu_tanh", od.GELU, {"gelu_tanh": True}, F.gelu(ad, approximate="tanh")),
            ("relu", od.RELU, {}, torch.relu(ad)),
        ):
            ck.check(nm, od.elementwise(op, a, **kw), ref, 1, ref.abs() + ad.abs())
        for nm, op, ref in (("swiglu", od.SWIGLU, F.silu(ad) * bd), ("add", od.ADD, ad + bd), ("mul", od.MUL, ad * bd)):
            ck.check(nm, od.elementwise(op, a, bb), ref, 1, ref.abs() + ad.abs() * (1 + bd.abs()))
        cols = ri(g, 1, 64)
        a2, bias = rn(g, ri(g, 1, 20) * cols), rn(g, cols)
        ref = (a2.double().view(-1, cols) + bias.double()).reshape(-1)
        ck.check("bias", od.elementwise(od.BIAS, a2, bias, cols=cols), ref, 1, ref.abs())
        # rope
        B, H, S, D = ri(g, 1, 3), ri(g, 1, 4), ri(g, 1, 9), 2 * ri(g, 1, 32)
        rot = 2 * ri(g, 1, D // 2)
        maxp = ri(g, S, 64)
        xr, cos, sin = rn(g, B, H, S, D), rn(g, maxp, rot // 2), rn(g, maxp, rot // 2)
        pos = torch.randint(0, maxp, (B, S), generator=g)
        for inter in (False, True):
            xrd = xr.double()
            c, s = cos.double()[pos][:, None], sin.double()[pos][:, None]
            xp, rest = xrd[..., :rot], xrd[..., rot:]
            x1, x2 = (xp[..., 0::2], xp[..., 1::2]) if inter else (xp[..., : rot // 2], xp[..., rot // 2:])
            y1, y2 = x1 * c - x2 * s, x2 * c + x1 * s
            yp = torch.stack([y1, y2], -1).reshape(xp.shape) if inter else torch.cat([y1, y2], -1)
            ref = torch.cat([yp, rest], -1)
            m1, m2 = x1.abs() * c.abs() + x2.abs() * s.abs(), x2.abs() * c.abs() + x1.abs() * s.abs()
            mp = torch.stack([m1, m2], -1).reshape(xp.shape) if inter else torch.cat([m1, m2], -1)
            mag = torch.cat([mp, rest.abs()], -1)
            ck.check("rope_interleaved" if inter else "rope", od.rope(xr, cos, sin, pos, rot=rot, interleaved=inter), ref, 2, mag)
        # matmul
        M, N, K = ri(g, 1, 17), ri(g, 1, 70), ri(g, 1, 300)
        A, Bm, bv = rn(g, M, K), rn(g, K, N), rn(g, N)
        tb = bool(ri(g, 0, 1))
        ref = A.double() @ Bm.double() + bv.double()
        mag = A.double().abs() @ Bm.double().abs() + bv.double().abs()
        ck.check("matmul", od.matmul(A, Bm.t().contiguous() if tb else Bm, bv, trans_b=tb), ref, K, mag)
        # attention prefill (GQA, causal top-left)
        B, Hkv = ri(g, 1, 2), ri(g, 1, 3)
        Hq = Hkv * ri(g, 1, 3)
        Sq, Skv, D = ri(g, 1, 20), ri(g, 1, 40), ri(g, 1, 96)
        causal = bool(ri(g, 0, 1))
        if causal:
            Skv = max(Skv, Sq)
        q, k, v = rn(g, B, Hq, Sq, D), rn(g, B, Hkv, Skv, D), rn(g, B, Hkv, Skv, D)
        rep = Hq // Hkv
        kd, vd = k.double().repeat_interleave(rep, 1), v.double().repeat_interleave(rep, 1)
        scale = D ** -0.5
        logits = q.double() @ kd.transpose(-1, -2) * scale
        if causal:
            logits = logits.masked_fill(torch.ones(Sq, Skv, dtype=torch.bool).triu(1), float("-inf"))
        p = torch.softmax(logits, -1)
        ref = p @ vd
        lmag = (q.double().abs() @ kd.abs().transpose(-1, -2) * scale).amax(-1, keepdim=True)
        mag = (p @ vd.abs()) * (1 + 2 * lmag) + ref.abs()
        ck.check("attention", od.attention(q, k, v, causal=causal, block=ri(g, 1, 64)), ref, D + Skv, mag)
        # paged decode
        B, Hkv, D, bs = ri(g, 1, 3), ri(g, 1, 2), ri(g, 1, 64), ri(g, 1, 16)
        Hq = Hkv * ri(g, 1, 4)
        maxb, nblk = ri(g, 1, 5), 12
        lens = torch.randint(1, maxb * bs + 1, (B,), generator=g)
        table = torch.stack([torch.randperm(nblk, generator=g)[:maxb] for _ in range(B)])
        qd, kc, vc = rn(g, B, Hq, D), rn(g, nblk, bs, Hkv, D), rn(g, nblk, bs, Hkv, D)
        out = od.paged_decode(qd, kc, vc, table, lens)
        rr, mm = [], []
        for i in range(B):
            Kb = kc[table[i]].reshape(-1, Hkv, D)[: lens[i]].double().repeat_interleave(Hq // Hkv, 1).transpose(0, 1)
            Vb = vc[table[i]].reshape(-1, Hkv, D)[: lens[i]].double().repeat_interleave(Hq // Hkv, 1).transpose(0, 1)
            lg = (qd[i].double()[:, None] @ Kb.transpose(-1, -2)) * D ** -0.5
            pp = torch.softmax(lg, -1)
            rr.append((pp @ Vb)[:, 0])
            lm = (qd[i].double().abs()[:, None] @ Kb.abs().transpose(-1, -2) * D ** -0.5).amax(-1, keepdim=True)
            mm.append(((pp @ Vb.abs()) * (1 + 2 * lm))[:, 0])
        ref = torch.stack(rr)
        ck.check("paged_decode", out, ref, D + maxb * bs, torch.stack(mm) + ref.abs())
        # conv2d, both algorithms
        gr = ri(g, 1, 3)
        Cg, Og = ri(g, 1, 4), ri(g, 1, 4)
        kh, kw, sh, sw = ri(g, 1, 4), ri(g, 1, 4), ri(g, 1, 3), ri(g, 1, 3)
        ph, pw, dh, dw = ri(g, 0, kh // 2), ri(g, 0, kw // 2), ri(g, 1, 2), ri(g, 1, 2)
        Hh, Ww = ri(g, dh * (kh - 1) + 1, 20), ri(g, dw * (kw - 1) + 1, 20)
        xi, wi, bi = rn(g, ri(g, 1, 2), Cg * gr, Hh, Ww), rn(g, Og * gr, Cg, kh, kw), rn(g, Og * gr)
        cargs = dict(stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw), groups=gr)
        ref = F.conv2d(xi.double(), wi.double(), bi.double(), **cargs)
        mag = F.conv2d(xi.double().abs(), wi.double().abs(), bi.double().abs(), **cargs)
        for al in (0, 1):
            ck.check(f"conv2d_algo{al}", od.conv2d(xi, wi, bi, (sh, sw), (ph, pw), (dh, dw), gr, al), ref, Cg * kh * kw, mag)
        # pooling
        kh, kw = ri(g, 1, 4), ri(g, 1, 4)
        sh, sw, ph, pw = ri(g, 1, 3), ri(g, 1, 3), ri(g, 0, kh // 2), ri(g, 0, kw // 2)
        xp = rn(g, ri(g, 1, 2), ri(g, 1, 5), ri(g, kh, 18), ri(g, kw, 18))
        pa = ((kh, kw), (sh, sw), (ph, pw))
        ref = F.max_pool2d(xp.double(), *pa)
        ck.check("maxpool", od.pool2d(xp, *pa), ref, 1, ref.abs())
        cip = bool(ri(g, 0, 1))
        ref = F.avg_pool2d(xp.double(), *pa, count_include_pad=cip)
        ck.check("avgpool", od.pool2d(xp, *pa, avg=True, count_include_pad=cip), ref, kh * kw,
                 F.avg_pool2d(xp.double().abs(), *pa, count_include_pad=cip))
        ref = xp.double().mean((2, 3), keepdim=True)
        ck.check("global_avgpool", od.global_avg_pool(xp), ref, xp.shape[2] * xp.shape[3],
                 xp.double().abs().mean((2, 3), keepdim=True))
        # fusion: MatMul -> bias -> SwiGLU in one pass, against f64 and against the unfused list
        M, N, K = ri(g, 1, 8), ri(g, 1, 40), ri(g, 1, 80)
        A, W, bv, Gt = rn(g, M, K), rn(g, K, N), rn(g, N), rn(g, M, N)
        gph = od.Graph()
        ia, iw, ib, ig = gph.input(A), gph.input(W), gph.input(bv), gph.input(Gt)
        y1, y2, y3 = gph.empty(M * N), gph.empty(M * N), gph.empty(M * N)
        gph.node(od.MATMUL, {1: ia, 2: iw, 3: y1, 4: M, 5: N, 6: K, 7: od.NONE, 8: 0})
        gph.node(od.BIAS, {1: y1, 2: ib, 3: y2, 4: M * N, 5: N})
        gph.node(od.SWIGLU, {1: y2, 2: ig, 3: y3, 4: M * N}, keep=True)
        gph.materialize()
        start = gph.arena.clone()
        passes = gph.run(True)
        fused = gph.out(y3, (M, N)).clone()
        gph.arena.copy_(start)
        passes0 = gph.run(False)
        h = A.double() @ W.double() + bv.double()
        ref = F.silu(h) * Gt.double()
        mag = ((A.double().abs() @ W.double().abs() + bv.double().abs()) * (1 + h.abs()) + ref.abs()) * (1 + Gt.double().abs())
        ck.check("fused_mm_bias_swiglu", fused, ref, K, mag)
        if (passes, passes0) != (1, 3):
            ck.rows["fusion_passes"] = (1, float("inf"), 1)
        else:
            c, w_, b_ = ck.rows.get("fusion_passes", (0, 0.0, 0))
            ck.rows["fusion_passes"] = (c + 1, 0.0, b_)
    return ck.report()


if __name__ == "__main__":
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261027
    sys.exit(1 if run(cases, seed) else 0)
