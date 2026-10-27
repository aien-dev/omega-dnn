# THROWAWAY measurement checker (lane OmegaDnn.GpuDnn, 2026-10-26), not product code.
# float64 numpy reference from the input files the Omega driver wrote; compares every saved output.
# Usage: python check.py INDIR OMEGA_OUTDIR TORCH_OUTDIR > errors.jsonl
# Per (case, arm): max_abs = max|got-ref|; max_rel = max_abs / max|ref| (normwise);
# violations = count(|got-ref| > rtol*|ref| + atol), tolerances from omega-mv tests (gpu_elementwise_test.c,
# gpu_attention_test.c header comments). The gate applies to the Omega arm; baseline errors are reported.
import glob, json, os, sys
import numpy as np

indir, odir, tdir = sys.argv[1], sys.argv[2], sys.argv[3]
def f32(c, w): return np.fromfile(f"{indir}/{c}.{w}.bin", dtype=np.float32).astype(np.float64)

def attn(q, k, v, nkv, causal_base=None):
    # q [rows][nq][hd], k/v [ctx][nkv][hd]
    rows, nq, hd = q.shape; ratio = nq // nkv; out = np.empty_like(q)
    for h in range(nq):
        kh, vh = k[:, h // ratio, :], v[:, h // ratio, :]
        s = q[:, h, :] @ kh.T / np.sqrt(hd)
        if causal_base is not None:
            mask = np.arange(k.shape[0])[None, :] > (causal_base + np.arange(rows))[:, None]
            s = np.where(mask, -np.inf, s)
        s = np.exp(s - s.max(axis=1, keepdims=True)); s /= s.sum(axis=1, keepdims=True)
        out[:, h, :] = s @ vh
    return out

REF, TOL = {}, {}
for rows in (1, 64):
    c = f"rmsnorm_{rows}x2048"; x = f32(c, "x").reshape(rows, 2048); w = f32(c, "w")
    REF[c] = (x / np.sqrt((x * x).mean(axis=1, keepdims=True) + np.float64(np.float32(1e-5))) * w).ravel(); TOL[c] = (1e-5, 1e-6)
c = "rope_32x128"; x = f32(c, "x").reshape(32, 128); cs, sn = f32(c, "cos"), f32(c, "sin")
REF[c] = np.concatenate((x[:, :64] * cs - x[:, 64:] * sn, x[:, :64] * sn + x[:, 64:] * cs), axis=1).ravel(); TOL[c] = (1e-6, 1e-6)
for n in (5632, 131072):
    c = f"swiglu_{n}"; g, u = f32(c, "gate"), f32(c, "up")
    REF[c] = g / (1.0 + np.exp(-g)) * u; TOL[c] = (1e-5, 1e-6)
for c, nq, nkv, hd, ctx in (("gqa_f32_32q8kv_hd128_ctx2048", 32, 8, 128, 2048), ("gqa_f32_32q4kv_hd64_ctx2048", 32, 4, 64, 2048)):
    REF[c] = attn(f32(c, "q").reshape(1, nq, hd), f32(c, "k").reshape(ctx, nkv, hd), f32(c, "v").reshape(ctx, nkv, hd), nkv).ravel(); TOL[c] = (2e-4, 2e-5)
c = "gqa_rows_f32_32q8kv_hd128_m256"
REF[c] = attn(f32(c, "q").reshape(256, 32, 128), f32(c, "k").reshape(256, 8, 128), f32(c, "v").reshape(256, 8, 128), 8, causal_base=0).ravel(); TOL[c] = (2e-4, 2e-5)
c = "paged_bf16_32q8kv_hd128_ctx2048_bs16"
raw = np.fromfile(f"{indir}/{c}.pool.bin", dtype=np.uint16).astype(np.uint32) << 16
pool = raw.view(np.float32).astype(np.float64).reshape(128, 2, 16, 8, 128)
ids = np.fromfile(f"{indir}/{c}.ids.bin", dtype=np.uint32)
g = pool[ids]; k = g[:, 0].reshape(2048, 8, 128); v = g[:, 1].reshape(2048, 8, 128)
qp = f32(c, "q")
if "--perturb-q" in sys.argv: qp[0] += 0.5  # input-side negative control: reference from a wrong q; omega output must be flagged
REF[c] = attn(qp.reshape(1, 32, 128), k, v, 8).ravel(); TOL[c] = (2e-4, 2e-5)

def report(case, arm, got):
    ref = REF[case]; rt, at = TOL[case]
    if got.size != ref.size:
        print(json.dumps({"case": case, "arm": arm, "error": f"size {got.size} != {ref.size}"})); return
    d = np.abs(got.astype(np.float64) - ref)
    print(json.dumps({"case": case, "arm": arm, "n": int(ref.size), "max_abs": float(d.max()), "max_rel": float(d.max() / np.abs(ref).max()),
                      "rtol": rt, "atol": at, "violations": int((d > rt * np.abs(ref) + at).sum()), "nonfinite": int((~np.isfinite(got)).sum())}))

for case in REF:
    p = f"{odir}/{case}.out.bin"
    if os.path.exists(p): report(case, "omega", np.fromfile(p, dtype=np.float32))
    for t in sorted(glob.glob(f"{tdir}/{case}.*.out.npy")):
        report(case, os.path.basename(t)[len(case) + 1:-len(".out.npy")], np.load(t).ravel())
