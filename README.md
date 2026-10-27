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
| RMSNorm, LayerNorm, Softmax, RoPE (split and interleaved), SiLU, GELU (erf, tanh), SwiGLU, Add, Mul, bias, ReLU | IMPLEMENTED in .om, EXERCISED on AArch64 (tests/) |
| MatMul (+bias, transposed B), flash-tiled attention prefill (GQA, causal), paged-KV decode | IMPLEMENTED in .om, EXERCISED on AArch64 |
| Conv2d (direct and im2col + GEMM, stride, pad, dilation, groups), MaxPool, AveragePool, GlobalAveragePool | IMPLEMENTED in .om, EXERCISED on AArch64 |
| Fusion graph (producer + elementwise chain, one store) | IMPLEMENTED, EXERCISED (pass counts and values in tests/) |
| GB10 GPU rows | interim C SASS codegen in omega-mv, measured against cuDNN: see the Stack7 OmegaDnn report |
| SPIR-V | PROPOSED, blocked: omega-mv IR mode has no floats or memory |
| GEMM from aien-dev/omega-blas | PROPOSED (the MatMul here is an interim f64-accumulating GEMM in .om) |

## Build and test

```sh
sh tools/build.sh                  # fetches the pins into .deps/, checks both compiler hashes, binds
python -m pytest -q tests          # needs torch (CPU); compares against torch float64
python tests/mutants.py            # each mutant of om/dnn.om must fail the suite
```

`build/` gets `libomega_native.so`, the generated `omega_native.{py,h,c,rs,zig,go}` and
`omega_api.h`. The Python package [`python/omega_dnn`](python/omega_dnn/__init__.py) only lays out
the arena and the descriptor ([docs/DESCRIPTOR.md](docs/DESCRIPTOR.md)); every value is computed by
the Omega unit. Tensors enter through DLPack (`torch.from_dlpack`).

## Numerics

f32 in and out, f64 inside, one rounding per stored element. exp is Cody–Waite with a degree-13
Taylor polynomial, erf a positive series (|x| ≥ 6 gives ±1). Tolerances are derived in
[docs/TOLERANCE.md](docs/TOLERANCE.md) and checked element by element in `tests/`.

## Limits of the seeded compiler (and the proposals)

The OM-2 compiler gives each function 16 registers with no spilling, has no vector or FMA words,
and bounds every loop at 1,000,000 passes. The kernels are scalar and bounds-checked, so they trail
oneDNN on speed; the measured gap is in the report. The Omega extensions that close it: f32x4
vector words and FMA, a register allocator with spilling, and IR-mode floats plus a buffer ABI for
the SPIR-V backend.

Licence: Apache-2.0 WITH LLVM-exception.
