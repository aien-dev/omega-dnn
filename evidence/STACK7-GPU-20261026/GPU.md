# OmegaDnn GB10 GPU rows: Omega-emitted DNN kernels vs PyTorch CUDA / cuDNN

Lane OmegaDnn.GpuDnn. Machine: DGX Spark, NVIDIA GB10 (sm_121), driver 580.173.02.
Omega source: aien-dev/omega-mv `e5b32567a` (worktree /tmp/OmegaDnn/gpu/omega-mv), physics at the
`physics.lock` pin `048e5f5e8` (worktree /tmp/OmegaDnn/gpu/physics).
Baseline: /home/drakestapleton/.venv (read-only): torch 2.14.0+cu130, CUDA 13.0, cuDNN 92400 (9.24.0), numpy 2.5.3.

**Status of the kernels under test: INTERIM.** Every Omega GPU kernel below is emitted as SASS by C code in
omega-mv (`src/omega_gpu_*_api.c` IR builders and `src/omega_blackwell_codegen_*.c`). That is C codegen, not
`.om`. It is the code the engine runs today, measured as it stands.

Evidence label for any chip row here: QUALIFIED-on-GB10 candidate (one machine, one driver). It is not
QUALIFIED in general.

## 1. Inventory

| Op | Omega kernel emitter (omega-mv e5b32567a) | Host API | Test program (Makefile target) | Measured here |
|---|---|---|---|---|
| RMSNorm f32 | IR built in `src/omega_gpu_elementwise_api.c`, lowered by `src/omega_blackwell_codegen.c` | `omega_gpu_rmsnorm_f32` (`src/omega_gpu_elementwise_api.h`) | `tests/gpu_elementwise_test.c` → `build/gpu_elementwise_test` | yes |
| RoPE f32 (rotate-half) | same file | `omega_gpu_rope_f32` | same | yes |
| SwiGLU f32 | same file | `omega_gpu_swiglu_f32` | same | yes |
| GQA attention, f32 KV (decode, 1 query) | IR built in `src/omega_gpu_attention_api.c` | `omega_gpu_gqa_attention_f32` (`src/omega_gpu_attention_api.h`) | `tests/gpu_attention_test.c` → `build/gpu_attention_test` | yes (hd 64, 128) |
| GQA attention, f32 KV, causal prefill rows | same file | `omega_gpu_gqa_attention_rows_f32` | same | yes (hd 128) |
| Paged decode attention, bf16 KV | same file | `omega_gpu_paged_attention_bf16` (+ `_batch_`, resident pool `omega_gpu_attention_pool_alloc`) | same | yes (resident pool, hd 128) |
| Fused decode attention hd128 (qk-norm + RoPE + KV write + attend, 32q/4kv) | `src/omega_blackwell_codegen_attn.c` | `omega_gpu_decode_attn` (`src/omega_gpu_decode_attn_api.h`) | `tests/gpu_decode_attn_test.c` → `build/gpu_decode_attn_test` | inventory only |
| Fused decode attention hd256 (24q/4kv, partial RoPE) | `src/omega_blackwell_codegen_attn256.c` | `omega_gpu_decode_attn256` | `tests/gpu_decode_attn256_test.c` → `build/gpu_decode_attn256_test` | inventory only |
| Prefill attention hd128 (tensor cores) | `src/omega_blackwell_codegen_pattn.c`, `_pf.c`, `_tc.c` | `omega_gpu_prefill_attn` | `tests/gpu_prefill_attn_test.c` → `build/gpu_prefill_attn_test` | inventory only |
| Prefill attention hd256 | `src/omega_blackwell_codegen_pattn256.c`, `_attn256.c`, `_tc.c` | `omega_gpu_prefill_attn256` | `tests/gpu_prefill_attn256_test.c` → `build/gpu_prefill_attn256_test` | inventory only |
| Gated DeltaNet decode step | `src/omega_blackwell_codegen_gdn.c` | `omega_gpu_gdn_step` (`src/omega_gpu_gdn_api.h`) | `tests/gpu_gdn_test.c`, `tests/gpu_gdn_multi_test.c` | inventory only |
| MoE router (one launch) | `src/omega_blackwell_codegen_router.c` | `omega_gpu_router_api.h` | `tests/gpu_router_test.c` → `build/gpu_router_test` | inventory only |
| MoE expert window matmul / FP8 GEMV | `src/omega_blackwell_codegen_moe.c`, `_moe2.c` | `omega_gpu_moe_api.h`, `omega_gpu_fp8_api.h` | `tests/gpu_moe_window_test.c`, `tests/gpu_fp8_api_test.c` | inventory only |

"Inventory only" means NOT_RUN in this pass. The fused engine kernels take engine state (a KV cache object, norms,
GDN layer weights), so a like-for-like torch baseline needs a separate harness. They are not measured here, and
nothing below makes a claim about them.

## 2. Protocol (frozen before the first timed run)

Arms:
- **A, Omega**: `/tmp/OmegaDnn/gpu/bench/omega_dnn_bench` (throwaway driver, `bench/omega_dnn_bench.c`) linked
  against `build/libomega_gpu.a` from e5b32567a. Built with the Makefile's CFLAGS plus `-ffp-contract=off`.
  Session spin left at the library default (`omega_gpu_session_spin_us() == 0`).
- **B, PyTorch CUDA**: `bench/torch_arm.py` under /home/drakestapleton/.venv.
- Both arms read the same inputs. The Omega driver generates them with a fixed LCG (seed 0x0d1ec0de) and writes
  them to disk; the torch arm reads those files.

Cases (dtype f32 unless noted):

| Case | Shape | Predeclared baseline (decides the ratio) | Reference-only baselines |
|---|---|---|---|
| rmsnorm_1x2048 | rows 1, dim 2048, eps 1e-5 | `torch_rms_norm` (F.rms_norm) | none |
| rmsnorm_64x2048 | rows 64, dim 2048 | `torch_rms_norm` | none |
| rope_32x128 | 32 heads x hd 128, rotate-half, pos 1234, theta 1e6 | `torch_ops` (mul/sub/cat) | none |
| swiglu_5632 | n 5632 | `torch_silu_mul` (F.silu(g)*u) | none |
| swiglu_131072 | n 131072 | `torch_silu_mul` | none |
| gqa_f32_32q8kv_hd128_ctx2048 | 1 query, 32 q / 8 kv heads, hd 128, ctx 2048 | `sdpa_efficient_f32` | `sdpa_{cudnn,flash,math}_f32` (cudnn/flash expected to refuse f32), `sdpa_{cudnn,flash}_bf16_REFONLY` |
| gqa_f32_32q4kv_hd64_ctx2048 | 1 query, 32 q / 4 kv, hd 64, ctx 2048 | `sdpa_efficient_f32` | same |
| gqa_rows_f32_32q8kv_hd128_m256 | 256 causal rows, base 0, 32 q / 8 kv, hd 128 | `sdpa_efficient_f32` (is_causal) | same |
| paged_bf16_32q8kv_hd128_ctx2048_bs16 | bf16 KV pool, block 16, 128 scattered blocks, 1 seq, 32 q / 8 kv, hd 128, q f32 | `sdpa_cudnn_bf16_pregathered` (cuDNN, KV already contiguous, q rounded to bf16) | `gather+sdpa_cudnn_bf16`, the flash/efficient/math variants, `sdpa_efficient_f32q_f32kv_pregathered` |

Metrics, per timed call:
- **Primary, device time.**
  - Omega: `info.elapsed_ns`, the library's submit-to-completion-marker time for the launch(es).
  - Torch: `torch.cuda.Event` elapsed between records placed just before and just after the op, with
    device-resident inputs.
  - The two mechanisms are not identical. Omega's figure includes channel submit. The event figure includes the
    launch gap between the two records.
- **Secondary, host-to-host wall.**
  - Omega: `clock_gettime` around the synchronous API call. That covers staging copy-in, launch, marker wait and
    copy-out. The f32 GQA paths stage the whole KV every call. The paged path reads the resident pool in place.
  - Torch: `perf_counter_ns` around the H2D copy of the per-call inputs (same set Omega stages: the full KV for
    f32 GQA, q only for paged), the op, and the D2H copy.

Design:
- Paired and interleaved. One warm-up round is discarded, then 5 measured rounds.
- In each round both arms run, one process each, in an order drawn from `random.Random(20261026)` and recorded
  in `rounds.jsonl`.
- Per process and case: 10 untimed warm-up calls, then 50 timed calls.
- Each arm process takes `/tmp/aien-gb10.lock` for its own lifetime only (`flock -w 900` around the GPU command).
- Both arms run under `taskset -c 8-9`. loadavg is recorded before and after every process.

Statistics (Wolfram):
- Per case and arm: median and IQR (Q1 to Q3, Wolfram `Quantile` with the default definition) of the 250 pooled
  samples.
- Ratio = Omega median / baseline median, on pooled samples. Also reported: the median over the 5 rounds of the
  per-round ratio of medians.
- A ratio above 1 means Omega is slower. It is reported as SLOWER with the factor and not softened.
- There is no pass threshold. This is a characterization, not a gate.

Correctness:
- Every arm's first output is compared with a float64 numpy reference computed from the same input files
  (`bench/check.py`).
- Reported: max abs error, and normwise rel = max abs / max|ref|.
- Violations are counted against the omega-mv tolerances: rmsnorm and swiglu 1e-5 rel + 1e-6 abs, rope 1e-6 rel
  + 1e-6 abs, attention 2e-4 rel + 2e-5 abs.
- The Omega arm also checks every timed output bitwise against its first output.
- The baseline bf16 rows are reported but not gated.

Negative control:
- `omega_dnn_bench --mutant` arms the library's TEST-ONLY mutant for each op: RMSNORM (drops the last SHFL step),
  ROPE (wrong sign term), SWIGLU (exp(+g)), attention KV_HEAD (reads kv head ^ 1).
- `check.py` must report violations > 0 for every mutated case.
- An input-side control is also run: one q element of the paged case perturbed in the checker's view must be
  flagged.

No re-run selection. A re-run happens only for a documented machine fault, and both runs are then reported.
Machine load is recorded, since the Spark load1 was ~18-20 when this was written.

Protocol sha256, recorded before the first timed run (`gpu-bench/protocol.sha256`):
- GPU.md sections 1-2: `5f8bdad9…e31ee368`
- omega_dnn_bench.c: `4c2a9360…2378ec`
- torch_arm.py: `65f3ffe9…5ed2fda0`
- check.py: `b9812d70…0ee684ee`
- libomega_gpu.a: `886646bd…6e40b3`

check.py later gained only the `--perturb-q` flag, which is the input-side control declared above. The driver later
gained only the post-hoc `--spin-us` flag (section 6). Both new hashes are appended to the same file.

## 3. Runs: commands, dates, lock holds, compute-apps check

- Date 2026-10-27 UTC. Driver 580.173.02. Window 01:23:44Z to 01:26:45Z.
- Every GPU command ran through `gpu-bench/gpu_run.sh LABEL CMD`, which does four things:
  - writes the BOARD START line;
  - runs `flock -w 900 /tmp/aien-gb10.lock held.sh CMD` (the lock is held only for CMD);
  - writes the BOARD END line;
  - runs `nvidia-smi --query-compute-apps=pid --format=csv,noheader`.
- The paired campaign used `run_paired.py`, which takes one flock per arm process.
- Logs are in `gpu-runs/`. Load during the window: load1 19-37 on 20 cores. **Every timing below was measured under load.**

| UTC end | Command (cwd /tmp/OmegaDnn/gpu/omega-mv) | rc | lock held (s) | compute-apps after |
|---|---|---|---|---|
| 01:23:48 | `./build/gpu_elementwise_test --out …/elementwise.json` | 0 | 0.357 | empty |
| 01:24:05 | `./build/gpu_elementwise_test --timing --out …` | 0 | 0.677 | empty |
| 01:24:06 | `./build/gpu_attention_test --out …` | 0 | 0.994 | empty |
| 01:24:14 | `./build/gpu_attention_test --timing --out …` | 0 | 0.809 | empty |
| 01:24:15 | `./build/gpu_attention_test --hd 128 --out …` | 0 | 1.153 | empty |
| 01:24:21 | `bench/omega_dnn_bench runs/corr/omega` | 0 | 1.167 | empty |
| 01:24:31 | `taskset -c 8-9 ~/.venv/bin/python bench/torch_arm.py runs/corr/omega runs/corr/torch` | 0 | 4.649 | empty |
| 01:24:46 | `bench/omega_dnn_bench runs/mutant/omega --mutant` | 0 | 0.749 | empty |
| 01:25:23 | `python bench/run_paired.py runs/paired 5 1`: 12 arm processes, holds 0.66-3.53 s each (`rounds.jsonl`) | 0 | ≤3.531 per process | empty after every process |
| 01:26:43-45 | `bench/omega_dnn_bench_spin runs/spin/rN --spin-us 5000` (post-hoc, ×3) | 0 | 1.086 / 0.555 / 0.554 | empty |

The longest single lock hold was 4.65 s, far under the 20 min cap. BOARD has 24 OmegaDnn.GpuDnn lines, a START and an
END for each row; the campaign row has one pair.

Output excerpts:
```
gpu_elementwise_test --out:  PASS swiglu_131073 n=131073 rc=OK violations=0 … | mutant rc=OK violations=131059 CAUGHT
                             PASS: 35 checks, 0 failed
gpu_elementwise_test --timing: PASS timing rmsnorm_1x2048 … ok_calls=100/100 violations=0 median_ms=0.105 … chip_us_mean=101.6
                             PASS timing swiglu_5632 … median_ms=0.109 … chip_us_mean=101.7 ;  PASS: 4 checks, 0 failed
gpu_attention_test --out:    mutant Q_ROW: caught 8 of 8 ; mutant OUT_ROW: caught 8 of 8 ; PASS: 219 checks, 0 failed
gpu_attention_test --timing: PASS timing gqa_f32_tinyllama_32q4kv_ctx2048 … median_ms=0.652 p90_ms=1.167 … chip_us_mean=567.0
gpu_attention_test --hd 128: hd128 dominant token 299 (warp 1 of chunk 2): rc OK, 0 bad, worst 0 ; PASS: 225 checks, 0 failed
torch_arm (f32 cases):       sdpa_cudnn_f32 / sdpa_flash_f32: "RuntimeError: No available kernel. Aborting execution."
```

## 4. Correctness (first output of each arm vs float64 numpy reference, `gpu-runs/corr/check.jsonl`)

rel = max abs / max |ref| (normwise). Columns: Omega max abs, Omega rel, Omega violations, then the same three for
the baseline.

| Case | Omega max abs | Omega rel | Omega viol. | Baseline (arm) | Baseline max abs | Baseline rel | Baseline viol. |
|---|---|---|---|---|---|---|---|
| rmsnorm_1x2048 | 1.73e-7 | 6.86e-8 | 0 | torch_rms_norm | 1.73e-7 | 6.86e-8 | 0 |
| rmsnorm_64x2048 | 3.95e-7 | 1.51e-7 | 0 | torch_rms_norm | 4.44e-7 | 1.70e-7 | 0 |
| rope_32x128 | 9.85e-8 | 7.25e-8 | 0 | torch_ops | 9.85e-8 | 7.25e-8 | 0 |
| swiglu_5632 | 5.54e-7 | 9.73e-8 | 0 | torch_silu_mul | 6.15e-7 | 1.08e-7 | 0 |
| swiglu_131072 | 7.63e-7 | 1.34e-7 | 0 | torch_silu_mul | 6.73e-7 | 1.18e-7 | 0 |
| gqa_f32_32q8kv_hd128_ctx2048 | 1.08e-7 | 1.40e-6 | 0 | sdpa_efficient_f32 | 4.44e-8 | 5.75e-7 | 0 |
| gqa_f32_32q4kv_hd64_ctx2048 | 6.39e-8 | 1.61e-6 | 0 | sdpa_efficient_f32 | 2.54e-8 | 6.40e-7 | 0 |
| gqa_rows_f32_32q8kv_hd128_m256 | 1.44e-7 | 1.44e-7 | 0 | sdpa_efficient_f32 | 2.08e-7 | 2.09e-7 | 0 |
| paged_bf16_32q8kv_hd128_ctx2048_bs16 | 1.67e-8 | 2.11e-7 | 0 | sdpa_cudnn_bf16_pregathered | 2.16e-4 | 2.73e-3 | 1816 (not gated) |

Notes:
- **Omega has 0 violations and 0 non-finite values on all 9 cases.** In every round of every Omega process, all
  timed outputs were bitwise equal to the first output (`timed_outputs_differing_from_first = 0`, 45 of 45
  case-processes).
- The cuDNN bf16 rows round q (and, for the f32 cases, KV) to bf16 and return bf16. Their 2.7e-3 to 6.3e-3
  normwise error is a dtype effect, not a defect.
- Omega's paged path keeps q in f32 and accumulates in f32.
- The f32-q, f32-KV torch control on the same gathered pool (`sdpa_efficient_f32q_f32kv_pregathered`) has max abs
  4.08e-8 and 0 violations.

## 5. Negative controls (`gpu-runs/mutant/`)

- `omega_dnn_bench --mutant` (TEST-ONLY library mutants) was checked against the same reference. **Every case was
  flagged.**
  - rmsnorm_1x2048: 2048 of 2048 violations (max abs 1.05).
  - rmsnorm_64x2048: 131072 of 131072.
  - rope: 2048 of 4096 (the sin-term half).
  - swiglu_5632: 5628 of 5632.
  - swiglu_131072: 131004 of 131072.
  - gqa hd128: 4089 of 4096.
  - gqa hd64: 2046 of 2048.
  - rows: 1048265 of 1048576.
  - paged: 4088 of 4096.
- Input-side control (`check.py … --perturb-q`, q[0] += 0.5 in the reference only): the correct Omega paged output
  was flagged with 121 violations (max abs 9.29e-4). The checker rejects a reference computed from the wrong input.
- The in-tree batteries' own mutants were also caught (excerpts in section 3).

## 6. Performance (paired, 5 measured rounds × 50 calls = 250 samples per arm per case, `gpu-runs/paired/`)

Statistics came from Wolfram. Inputs are in `gpu-runs/paired/wolfram/<case>.wl`. Per case the query was
`{st[o], st[t], N[Median[Flatten[o]]/Median[Flatten[t]]], N[Median[MapThread[Median[#1]/Median[#2]&,{o,t}]]], st[ow], st[tw], N[Median[Flatten[ow]]/Median[Flatten[tw]]] (, st[x], N[Median[Flatten[o]]/Median[Flatten[x]]])}`
with `st[x_]:=N[{Median[Flatten[x]],Quantile[Flatten[x],1/4],Quantile[Flatten[x],3/4]}/1000]`. Units are µs.

All rows are dtype f32 except paged, which has bf16 KV and an f32 q.

**Primary: device time.** Omega submit→marker (spin 0, the library default) vs torch cudaEvent.

| Op | Shape | Omega median µs | Omega IQR [Q1, Q3] | Baseline (predeclared) | Baseline median µs | Baseline IQR | Ratio Omega/baseline | Median per-round ratio | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| RMSNorm | 1×2048 | 104.289 | [103.712, 104.801] | F.rms_norm | 10.432 | [10.176, 10.656] | 9.997 | 10.03 | SLOWER 10.0× |
| RMSNorm | 64×2048 | 107.617 | [106.849, 108.593] | F.rms_norm | 10.368 | [10.176, 10.624] | 10.380 | 10.45 | SLOWER 10.4× |
| RoPE | 32×128 | 104.577 | [103.745, 105.729] | torch ops | 35.456 | [35.168, 36.320] | 2.949 | 2.94 | SLOWER 2.9× |
| SwiGLU | 5632 | 104.257 | [103.809, 105.073] | silu·mul | 12.400 | [12.064, 12.896] | 8.408 | 8.43 | SLOWER 8.4× |
| SwiGLU | 131072 | 107.417 | [106.929, 108.417] | silu·mul | 12.224 | [12.000, 12.768] | 8.787 | 8.77 | SLOWER 8.8× |
| GQA decode | 32q/8kv hd128 ctx2048 | 422.412 | [417.475, 498.821] | SDPA mem-efficient f32 | 236.160 | [235.104, 237.312] | 1.789 | 1.80 | SLOWER 1.8× |
| GQA decode | 32q/4kv hd64 ctx2048 | 418.019 | [413.699, 514.212] | SDPA mem-efficient f32 | 171.648 | [170.848, 172.384] | 2.435 | 2.44 | SLOWER 2.4× |
| GQA causal prefill | 256 rows, 32q/8kv hd128 | 1481.469 | [1379.740, 1598.078] | SDPA mem-efficient f32 | 91.040 | [90.496, 91.808] | 16.27 | 16.13 | SLOWER 16.3× |
| Paged decode | bf16 KV bs16 ctx2048, 32q/8kv hd128 | 104.729 | [104.033, 105.537] | cuDNN SDPA bf16 (pre-gathered) | 29.920 | [29.408, 30.720] | 3.500 | 3.49 | SLOWER 3.5× |

Reference-only rows (they do not decide the verdict):

| Case | Omega median µs | Other arm | That arm's median µs [IQR] | Omega/that |
|---|---|---|---|---|
| GQA hd128 ctx2048 | 422.412 | cuDNN SDPA bf16 (q, KV rounded to bf16) | 93.968 [89.952, 99.456] | 4.495 |
| GQA hd64 ctx2048 | 418.019 | cuDNN SDPA bf16 | 36.704 [36.160, 37.312] | 11.39 |
| GQA rows m256 | 1481.469 | cuDNN SDPA bf16 | 52.064 [51.296, 52.896] | 28.45 |
| Paged ctx2048 | 104.729 | gather (index_select) + cuDNN SDPA bf16 | 208.672 [201.152, 220.864] | **0.502** |

The one case where Omega is faster is the reference-only paged comparison. When the torch arm must gather the
scattered pool blocks itself, Omega's resident-pool paged kernel takes half the device time of gather + cuDNN.
Against cuDNN on KV that is already contiguous, Omega is 3.5× slower.

**Secondary: host-to-host wall** (µs median [IQR]; ratio of medians from the same Wolfram calls):

| Case | Omega wall | Torch wall (H2D + op + D2H) | Ratio |
|---|---|---|---|
| rmsnorm_1x2048 | 106.801 [106.193, 107.409] | 18.920 [18.336, 19.712] | 5.645 |
| rmsnorm_64x2048 | 210.106 [208.370, 211.970] | 37.776 [37.296, 38.192] | 5.562 |
| rope_32x128 | 108.697 [107.857, 109.921] | 44.481 [43.856, 45.856] | 2.444 |
| swiglu_5632 | 109.649 [109.105, 110.609] | 31.192 [29.824, 32.465] | 3.515 |
| swiglu_131072 | 226.778 [222.386, 230.002] | 56.464 [56.017, 57.168] | 4.016 |
| gqa hd128 ctx2048 | 1211.235 [1136.458, 1305.500] | 554.773 [546.005, 572.277] | 2.183 |
| gqa hd64 ctx2048 | 550.573 [538.804, 651.798] | 267.043 [265.971, 268.322] | 2.062 |
| gqa rows m256 | 2531.270 [2421.349, 2676.775] | 293.419 [288.435, 298.995] | 8.627 |
| paged ctx2048 | 173.554 [171.746, 175.425] | 42.113 [41.088, 43.792] | 4.121 |

**Diagnosis of the ~104 µs floor.** The small Omega ops sit at about 104 µs whatever their size. This matches the
library's documented marker sleep-poll (`omega_gpu_session.h`: "50 us, which Linux timer slack rounds to about
102 us per launch on GB10, omega#328"). The frozen protocol measured the default.

**POST-HOC, reference-only, not paired, not part of the verdict:** 3 Omega processes with
`omega_gpu_session_set_spin_us(5000)` (`gpu-runs/spin/`, 150 samples per case). Correctness was identical: 0
violations, and timed outputs equal to the first. Ratio vs the paired-campaign baseline medians:

| Case | Omega spin-5000 median µs [IQR] | Omega/baseline |
|---|---|---|
| rmsnorm_1x2048 | 16.688 [16.624, 16.816] | 1.600 |
| rmsnorm_64x2048 | 21.784 [20.833, 23.184] | 2.101 |
| rope_32x128 | 6.528 [6.432, 6.688] | 0.184 |
| swiglu_5632 | 6.376 [6.160, 6.672] | 0.514 |
| swiglu_131072 | 17.160 [15.745, 17.888] | 1.404 |
| gqa hd128 ctx2048 | 404.452 [387.300, 467.956] | 1.713 |
| gqa hd64 ctx2048 | 355.339 [348.755, 379.940] | 2.070 |
| gqa rows m256 | 1095.650 [1047.642, 1193.067] | 12.03 |
| paged ctx2048 | 55.465 [54.801, 56.208] | 1.854 |

Wolfram query (`wolfram/spin_posthoc.wl`):
`st[x_]:=N[{Median[x],Quantile[x,1/4],Quantile[x,3/4]}/1000]; {…{"case", st[spin samples], N[Median[spin]/Median[baseline]]}…}`

Caveats:
- The spin figure still comes from a different timer mechanism than cudaEvent.
- The torch RoPE baseline is 3 to 5 unfused ops, so it is not a like-for-like fused kernel.
- The attention gap (1.7 to 12×) persists with spin. It is kernel time, not polling.

## 7. Verdict

- Correctness: Omega matches the float64 reference on all 9 cases with 0 violations, timed outputs are
  deterministic, and the negative controls caught every mutant and the perturbed input.
- Speed, as shipped (spin 0): Omega is slower than the PyTorch/cuDNN baseline on all 9 predeclared cases, from
  1.8× (GQA decode hd128) to 16.3× (causal prefill rows).
- The only win is reference-only: 0.50× device time vs gather + cuDNN on a scattered paged pool.

Evidence level: QUALIFIED-on-GB10 candidate (single machine, driver 580.173.02, 2026-10-27, under load1 19-37). It
is not QUALIFIED in general. Every kernel is INTERIM C codegen emitting SASS, not `.om`.

## 8. NOT_RUN

- Perf and torch comparison for fused decode attention hd128/hd256, prefill attention hd128/hd256 (tensor core),
  GDN decode step, MoE router and MoE window/FP8. They are inventory only: no matching torch harness exists yet.
- OmegaTile tsoftmax (omega-mv PR #869): not attempted (optional item).
- cuDNN on the f32 cases: torch refused with "No available kernel"; the cuDNN SDPA path has no f32 kernel.

Reproduce: from /tmp/OmegaDnn/gpu/omega-mv,
`nice -n 19 taskset -c 8-9 make -j2 PHYSICS_DIR=/tmp/OmegaDnn/gpu/physics build/gpu_elementwise_test build/gpu_attention_test`.
Then compile the driver (P=/tmp/OmegaDnn/gpu/physics, N=$P/third_party/nvidia-open-580.173.02) and run the rows in
section 3:
`nice -n 19 taskset -c 8-9 cc -std=gnu11 -Wall -Wextra -Werror -D_GNU_SOURCE -O2 -Isrc -I$P/m16 -I$P/nvrm -I$N/src/common/sdk/nvidia/inc -I$N/kernel-open/common/inc -I$N/kernel-open/nvidia-uvm -I$N/src/nvidia/arch/nvalloc/unix/include -ffp-contract=off -o omega_dnn_bench ~/handoffs/OmegaDnn/gpu-bench/omega_dnn_bench.c build/libomega_gpu.a -lpthread -lm`
