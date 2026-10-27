# Node descriptor (`layout dsc 4096`)

`dnn_run(ws, d)` runs a node list over one arena `ws` (uint32 words: f32 bit patterns, and uint32
integers for position ids, block tables and sequence lengths). Every offset is an element index
into `ws`. The descriptor is 512 little-endian u64 words:

| word | meaning |
|---|---|
| 0 | magic `0x4f444e` (`OMEGA_DNN_MAGIC`, also returned by `dnn_abi`) |
| 1 | node count, at most 31 |
| 2 | bit 0: fuse elementwise followers |
| 4 + 16·i + k | node i, argument k (k = 0 the op, 1..14 operands, 15 flags; flag bit 1 = keep, a graph output) |

Argument 3 is always the output offset. `NONE` = 2^64 − 1 marks an absent optional input. f64
scalars (eps, attention scale) are stored as their IEEE-754 bits.

| op | name (ONNX) | arguments |
|---|---|---|
| 1 | RMSNormalization (axis −1) | 1 x, 2 scale, 3 y, 4 rows, 5 cols, 6 eps |
| 2 | LayerNormalization (axis −1) | as 1, 7 bias or NONE |
| 3 | Softmax (axis −1, max-subtracted) | 1 x, 3 y, 4 rows, 5 cols |
| 4 | RotaryEmbedding | 1 x [B,H,S,D], 2 cos [maxpos, rot/2], 3 y, 4 sin, 5 position ids [B,S] (uint32), 6 B, 7 H, 8 S, 9 D, 10 rotary dim (0 = D), 11 interleaved |
| 5 | SiLU (Sigmoid·x) | 1 a, 3 y, 4 n |
| 6 | Gelu | as 5, 6 approximate (0 none/erf, 1 tanh) |
| 7 | SwiGLU silu(a)·b | 1 a, 2 b, 3 y, 4 n |
| 8 / 9 | Add / Mul | 1 a, 2 b, 3 y, 4 n |
| 10 | Add of a row bias b[x mod cols] | 1 a, 2 b, 3 y, 4 n, 5 cols |
| 11 | Relu | 1 a, 3 y, 4 n |
| 12 | MatMul (+ bias) | 1 A [M,K], 2 B [K,N] (or [N,K] with 8 = 1), 3 C [M,N], 4 M, 5 N, 6 K, 7 bias [N] or NONE |
| 13 | Attention prefill | 1 q [B,Hq,Sq,D], 2 k [B,Hkv,Skv,D], 3 o, 4 v, 5 B, 6 Hq, 7 Hkv, 8 Sq, 9 Skv, 10 D ≤ 512, 11 causal (top-left), 12 scale bits, 13 key block 1..64 (0 = 64) |
| 14 | Paged-KV decode attention | 1 q [B,Hq,D], 2 k cache [nblk, bs, Hkv, D], 3 o, 4 v cache, 5 block table [B,maxb], 6 lengths [B], 7 B, 8 Hq, 9 Hkv, 10 D ≤ 512, 11 bs ≤ 64, 12 maxb, 13 scale bits |
| 15 | Conv (2-D, NCHW) | 1 x, 2 w [O,C/g,kh,kw], 3 y, 4 bias or NONE, 5 scratch (im2col: C/g·kh·kw·OH·OW), 6 N, 7 C, 8 H, 9 W, 10 O, 11 packed kh kw sh sw ph pw dh dw (one byte each, low first, symmetric pads), 12 groups, 13 algorithm (0 direct, 1 im2col + GEMM) |
| 16 / 17 | MaxPool / AveragePool (2-D) | 1 x, 3 y, 4 N·C, 5 H, 6 W, 7 packed kh kw sh sw ph pw, 8 count_include_pad |
| 18 | GlobalAveragePool | 1 x, 3 y, 4 N·C, 5 H·W |

**Fusion.** With word 2 bit 0 set, a producer at node i absorbs the following elementwise nodes
i+1..e while each one reads the previous output as its `a` (argument 1) and that output is
virtual: not marked keep, not named by any later node's argument, and not the next node's `b`.
The fused chain is applied in f64 to each produced value and stored once, so a fused result is
rounded once where the unfused list rounds after every node. `dnn_run` returns the number of passes.

Traps: 1 bad magic, node count or op; 2 a parameter above its bound (D > 512, block > 64, bs > 64,
unknown elementwise op); 4 a loop over 1,000,000 passes (the seeded compiler's bound; elementwise
nodes run in 65,536-element chunks).
