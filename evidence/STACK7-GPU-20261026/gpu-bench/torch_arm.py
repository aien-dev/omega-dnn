# THROWAWAY measurement driver (lane OmegaDnn.GpuDnn, 2026-10-26), not product code.
# Baseline arm: PyTorch CUDA (+cuDNN where the op has a cuDNN path) on the same inputs the Omega driver wrote.
# Usage: python torch_arm.py INDIR OUTDIR [--warm 10] [--reps 50]
# Per case and backend: `warm` untimed calls, then `reps` timed calls; each timed call records
#   dev_ns  = torch.cuda.Event elapsed between records placed right before and after the op (device-resident inputs)
#   wall_ns = perf_counter_ns around host->device copy of the per-call inputs, the op, device->host copy (synchronous)
# First output of each backend is saved as OUTDIR/<case>.<backend>.out.npy. One JSON line per (case, backend).
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

indir, outdir = sys.argv[1], sys.argv[2]
warm = int(sys.argv[sys.argv.index("--warm") + 1]) if "--warm" in sys.argv else 10
reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 50
dev = torch.device("cuda")

def f32(case, what, shape):
    return torch.from_numpy(np.fromfile(f"{indir}/{case}.{what}.bin", dtype=np.float32).reshape(shape))

def bf16_pool(case):
    raw = np.fromfile(f"{indir}/{case}.pool.bin", dtype=np.uint16)
    return torch.from_numpy(raw.astype(np.int32)).to(torch.int32).mul(65536).view(torch.float32)

def bench(case, backend, host_inputs, fn):
    """host_inputs: CPU tensors copied per call for wall_ns; fn(dev_inputs) -> device output."""
    dins = [t.to(dev) for t in host_inputs]
    try:
        out = fn(dins); torch.cuda.synchronize()
    except Exception as e:  # backend refused this dtype/shape: record, do not substitute
        print(json.dumps({"case": case, "backend": backend, "error": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)
        return
    np.save(f"{outdir}/{case}.{backend}.out.npy", out.float().cpu().numpy())
    for _ in range(warm):
        fn(dins)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dev_ns, wall_ns = [], []
    for _ in range(reps):
        s.record(); fn(dins); e.record(); e.synchronize()
        dev_ns.append(int(round(s.elapsed_time(e) * 1e6)))
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        d2 = [t.to(dev) for t in host_inputs]
        o = fn(d2).cpu()
        wall_ns.append(time.perf_counter_ns() - t0)
    print(json.dumps({"case": case, "backend": backend, "warm": warm, "reps": reps, "dev_ns": dev_ns, "wall_ns": wall_ns}), flush=True)

def sdpa(backend, q, k, v, causal=False):
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)

BACKENDS = [("cudnn", SDPBackend.CUDNN_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)]

for rows, dim in ((1, 2048), (64, 2048)):
    c = f"rmsnorm_{rows}x{dim}"
    x, w = f32(c, "x", (rows, dim)), f32(c, "w", (dim,))
    wd = w.to(dev)
    bench(c, "torch_rms_norm", [x], lambda d: F.rms_norm(d[0], (dim,), wd, 1e-5))

c = "rope_32x128"
x, cs, sn = f32(c, "x", (32, 128)), f32(c, "cos", (64,)), f32(c, "sin", (64,))
csd, snd = cs.to(dev), sn.to(dev)
def rope(d):
    x0, x1 = d[0][:, :64], d[0][:, 64:]
    return torch.cat((x0 * csd - x1 * snd, x0 * snd + x1 * csd), dim=1)
bench(c, "torch_ops", [x], rope)

for n in (5632, 131072):
    c = f"swiglu_{n}"
    g, u = f32(c, "gate", (n,)), f32(c, "up", (n,))
    bench(c, "torch_silu_mul", [g, u], lambda d: F.silu(d[0]) * d[1])

for c, nq, nkv, hd, ctx in (("gqa_f32_32q8kv_hd128_ctx2048", 32, 8, 128, 2048), ("gqa_f32_32q4kv_hd64_ctx2048", 32, 4, 64, 2048)):
    q = f32(c, "q", (1, nq, 1, hd))
    k = f32(c, "k", (ctx, nkv, hd)).permute(1, 0, 2).unsqueeze(0).contiguous()
    v = f32(c, "v", (ctx, nkv, hd)).permute(1, 0, 2).unsqueeze(0).contiguous()
    for name, be in BACKENDS:
        bench(c, f"sdpa_{name}_f32", [q, k, v], lambda d, be=be: sdpa(be, d[0], d[1], d[2]).reshape(nq, hd))
    for name, be in BACKENDS[:2]:  # reference-only: the cuDNN/flash paths need half precision
        bench(c, f"sdpa_{name}_bf16_REFONLY", [q, k, v], lambda d, be=be: sdpa(be, d[0].bfloat16(), d[1].bfloat16(), d[2].bfloat16()).reshape(nq, hd))

c, nq, nkv, hd, m = "gqa_rows_f32_32q8kv_hd128_m256", 32, 8, 128, 256
q = f32(c, "q", (m, nq, hd)).permute(1, 0, 2).unsqueeze(0).contiguous()
k = f32(c, "k", (m, nkv, hd)).permute(1, 0, 2).unsqueeze(0).contiguous()
v = f32(c, "v", (m, nkv, hd)).permute(1, 0, 2).unsqueeze(0).contiguous()
for name, be in BACKENDS:
    bench(c, f"sdpa_{name}_f32", [q, k, v], lambda d, be=be: sdpa(be, d[0], d[1], d[2], True)[0].permute(1, 0, 2).reshape(m, nq * hd))
for name, be in BACKENDS[:2]:
    bench(c, f"sdpa_{name}_bf16_REFONLY", [q, k, v], lambda d, be=be: sdpa(be, d[0].bfloat16(), d[1].bfloat16(), d[2].bfloat16(), True)[0].permute(1, 0, 2).reshape(m, nq * hd))

# paged bf16 KV: pool [blk][plane k/v][slot 16][kv 8][hd 128] (one layer), ids[i] = physical block of logical block i
c, nq, nkv, hd, ctx, bs = "paged_bf16_32q8kv_hd128_ctx2048_bs16", 32, 8, 128, 2048, 16
nblk = ctx // bs
pool = bf16_pool(c).reshape(nblk, 2, bs, nkv, hd).to(torch.bfloat16)  # exact: values are bf16 bit patterns
ids = torch.from_numpy(np.fromfile(f"{indir}/{c}.ids.bin", dtype=np.uint32).astype(np.int64))
q = f32(c, "q", (1, nq, 1, hd))
poold, idsd = pool.to(dev), ids.to(dev)
def gather():
    g = poold.index_select(0, idsd)  # [nblk][2][bs][nkv][hd] in logical order
    kk = g[:, 0].reshape(ctx, nkv, hd).permute(1, 0, 2).unsqueeze(0)
    vv = g[:, 1].reshape(ctx, nkv, hd).permute(1, 0, 2).unsqueeze(0)
    return kk.contiguous(), vv.contiguous()
kc, vc = gather()
kcf, vcf = kc.float(), vc.float()
for name, be in BACKENDS:
    # KV pre-gathered contiguous (gather not timed: favours the baseline); q rounded to bf16 for the bf16 kernels
    bench(c, f"sdpa_{name}_bf16_pregathered", [q], lambda d, be=be: sdpa(be, d[0].bfloat16(), kc, vc).reshape(nq, hd))
    bench(c, f"gather+sdpa_{name}_bf16", [q], lambda d, be=be: sdpa(be, d[0].bfloat16(), *gather()).reshape(nq, hd))
bench(c, "sdpa_efficient_f32q_f32kv_pregathered", [q], lambda d: sdpa(SDPBackend.EFFICIENT_ATTENTION, d[0], kcf, vcf).reshape(nq, hd))
