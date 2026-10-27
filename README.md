# omega-dnn

Open deep-learning primitives written in seeded Omega: one OM-2 unit, [`om/dnn.om`](om/dnn.om),
with ONNX operator semantics and a small fusion-graph API. It is the Stack7 answer to NVIDIA
layer #12 (cuDNN): no NVIDIA headers, no cuDNN API names, only ONNX operator definitions.

**Parent:** [aien-dev/omega-mv](https://github.com/aien-dev/omega-mv) at
`e5b32567a7aff3ce594e3254a622a846a6cd190b` (the OM-2 compiler `seed/out/omc`, sha256 `2c824805…029f`,
and the seeded OM-1 compiler `evidence/SEED-0/omc/omc.elf`, sha256 `734f0a39…6b61`).
**Binding:** [aien-dev/omega-bridge](https://github.com/aien-dev/omega-bridge) native ABI v2
(PR [#15](https://github.com/aien-dev/omega-bridge/pull/15), `2b59387`). Both pins are in [`PINS`](PINS).

## Status (evidence labels kept separate)

| item | label |
|---|---|
| RMSNorm, LayerNorm, Softmax, RoPE (split and interleaved), SiLU, GELU (erf, tanh), SwiGLU, Add, Mul, bias, ReLU | IMPLEMENTED in .om, EXERCISED on AArch64 (host): [tests/test_ops.py](tests/test_ops.py), receipt [evidence/HOST-01dc3de2](evidence/HOST-01dc3de2) |
| MatMul (+bias, transposed B), flash-tiled attention prefill (GQA, causal), paged-KV decode | IMPLEMENTED in .om, EXERCISED on AArch64 (host), same receipt |
| Conv2d (direct and im2col + GEMM, stride, pad, dilation, groups), MaxPool, AveragePool, GlobalAveragePool | IMPLEMENTED in .om, EXERCISED on AArch64 (host), same receipt |
| Fusion graph (producer + elementwise chain, one store) | IMPLEMENTED, EXERCISED on AArch64 (host): pass counts and values in [tests/test_ops.py](tests/test_ops.py) |
| Mutants of om/dnn.om | 14 of 14 killed by the host suite: [tests/mutants.py](tests/mutants.py), same receipt |
| GB10 GPU tier: RMSNorm, RoPE (split), SwiGLU, SiLU, Softmax (c ≤ 64, through the attention kernel), GQA decode attention, causal prefill attention, paged-KV decode (single and batch), MatMul (f32, +bias, transposed B) | EXERCISED on the GB10 chip through CUDApp's native kernels, gated against this repo's .om host result: [tests/test_gpu.py](tests/test_gpu.py), receipts [evidence/GPU-9841bb3a](evidence/GPU-9841bb3a) (elementwise, attention; CUDApp main `9841bb3a`) and [evidence/GPU-b28372d1](evidence/GPU-b28372d1) (MatMul; CUDApp PR #139 `b28372d1`, not merged). Not QUALIFIED: one seed, 32 cases per op. Failed earlier runs: [evidence/GPU-9841bb3a/LOSSES.md](evidence/GPU-9841bb3a/LOSSES.md) |
| No GPU path | see "Not on the GPU" below |
| Earlier GB10 measurement (2026-10-26) | omega-mv's interim C SASS codegen measured against torch/cuDNN: [evidence/STACK7-GPU-20261026/GPU.md](evidence/STACK7-GPU-20261026/GPU.md). Those kernels are not the CUDApp kernels above |
| SPIR-V | PROPOSED, blocked: omega-mv IR mode has no floats or memory |
| GEMM from aien-dev/omega-blas | PROPOSED (the MatMul here is an interim f64-accumulating GEMM in .om) |

### Not on the GPU

NOT_RUN: CUDApp has no kernel for these (header search of CUDApp `src/*.h` at `9841bb3a`):
LayerNorm, GELU (erf and tanh), ReLU, Add, Mul, bias, RoPE interleaved, Conv2d (direct and
im2col), MaxPool, AveragePool, GlobalAveragePool, the fusion graph, and standalone Softmax.

Unsupported shapes, refused by the CUDApp kernels and not tested:
- RMSNorm: dim % 128 ≠ 0, or dim > 16384.
- Attention: head_dim ∉ {64, 128}; more than 64 query heads; a GQA ratio that is not a power of two; context above 4096.
- RoPE: rotary width below head_dim.
- Softmax: more than 64 logits.

## Build and test

```sh
sh tools/build.sh                  # fetches the pins into .deps/, checks both compiler hashes, binds
python tests/test_ops.py 200       # needs torch (CPU); randomized, compares against torch float64
python tests/mutants.py            # each mutant of om/dnn.om must fail the suite
python tests/test_gpu.py --lib <CUDApp>/build-pic/libcudapp_dnn.so --group ew|attn|gemm \
    --cudapp-sha <sha> --out <receipt.json> [--hold]   # GB10; without --hold only the host side runs
```

`build/` gets `libomega_native.so`, the generated `omega_native.{py,h,c,rs,zig,go}` and
`omega_api.h`. The Python package [`python/omega_dnn`](python/omega_dnn/__init__.py) only lays out
the arena and the descriptor ([docs/DESCRIPTOR.md](docs/DESCRIPTOR.md)); every value is computed by
the Omega unit. Tensors enter through DLPack (`torch.from_dlpack`).

## Numerics

f32 in and out, f64 inside, one rounding per stored element. exp is Cody–Waite with a degree-13
Taylor polynomial, erf a positive series (|x| ≥ 6 gives ±1). Tolerances are derived in
[docs/TOLERANCE.md](docs/TOLERANCE.md), the GPU-tier bounds in its "GPU tier" section, and checked
element by element in `tests/`.

## Limits of the seeded compiler (and the proposals)

The OM-2 compiler gives each function 16 registers with no spilling, has no vector or FMA words,
and bounds every loop at 1,000,000 passes. The kernels are scalar and bounds-checked. The speed
gap to oneDNN is NOT_RUN: no comparison against oneDNN has been measured. The Omega extensions that close it: f32x4
vector words and FMA, a register allocator with spilling, and IR-mode floats plus a buffer ABI for
the SPIR-V backend.

Licence: Apache-2.0 WITH LLVM-exception.
