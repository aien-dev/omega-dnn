# Tolerance contract

The reference is torch float64 on the same f32 inputs. Every output element y of every op must
satisfy

    |y − ref| ≤ u32·|ref| + 2·γ(n + 64)·M + 2^-149,    u32 = 2^-24,  u64 = 2^-53,  γ(k) = k·u64 / (1 − k·u64)

- **u32·|ref|**: the unit computes in f64 and rounds once to f32 when it stores (round to nearest:
  at most half an f32 ulp, relative 2^-24).
- **2·γ(n + 64)·M**: f64 rounding inside the unit plus inside the reference. n is the length of the
  op's longest reduction (row length for the norms and softmax, K for MatMul, D + keys for attention,
  C/g·kh·kw for Conv, the window for pooling) and M is the op's absolute-value magnitude computed in
  f64 by the test (for example |A|·|B| + |bias| for MatMul; Σp·|v|·(1 + 2·max scaled |q|·|k|) for
  attention; (1 + |x − max|)·p for softmax). The extra 64 covers the fixed-length transcendental
  evaluations (exp, erf, tanh, sqrt and divisions) per element.
- **2^-149**: the smallest f32 subnormal, for outputs that underflow.

Numbers, all from Wolfram (`WolframLanguageEvaluator`, 2026-10-27):

| query | result |
|---|---|
| `N[2^-24, 17]`, `N[2^-53, 17]` | 5.9604644775390625e-8, 1.1102230246251565e-16 |
| `N[gam[2^20 + 64], 6]` (largest reduction allowed by the loop bound) | 1.16422e-10 |
| `N[gam[2^20 + 64]/2^-24, 6]` | 0.00195324 — the f64 term stays below 0.2% of u32 per unit of M |
| `N[gam[64], 6]` | 7.10543e-15 |
| `N[(Log[2]/2)^14/14! * Exp[Log[2]/2], 6]` (exp: Taylor remainder after degree 13 on \|r\| ≤ ln2/2) | 5.85118e-18 |
| `N[Erfc[6], 6]` (erf clamps to ±1 for \|x\| ≥ 6) | 2.15197e-17 |
| `N[72/(2*70 + 3), 6]`, `N[r/(1 - r)*10^-17, 6]` (erf series: term ratio at q = 6 once k ≥ 70; tail after the 1e-17 stop) | 0.503497, 1.01408e-17 |
| `N[Product[72/(2 j + 3), {j, 36, 70}], 6]` (erf series at q = 6 has not stopped before k = 70) | 9.25083e-7 |
| `N[2^-149, 6]` | 1.40130e-45 |

The approximation errors of exp and erf (≤ 2.2e-17 relative) are below γ(64), so they sit inside
the f64 term. ReLU and MaxPool are exact (measured worst err/tol 0).

**Tightness.** The u32 term is the binding one: the measured worst err/tol ratio is 0.96–1.00 for
every rounding op (tests/test_ops.py output). With every bound halved
(`OMEGA_DNN_TOL_SCALE=0.5`, negative control) the suite reports violations.

**Scope.** A defect smaller than about one f32 rounding of the output is invisible by construction:
for example perturbing the low half of the Cody–Waite ln2 constant. tests/mutants.py lists the
mutants that the suite must kill.

## GPU tier (tests/test_gpu.py)

The GPU tier runs CUDApp's native GB10 kernels on the same f32 inputs and holds each chip output
element y to the omega-dnn host unit h (the gate) and to torch float64 r (a diagnostic):

    gate:        |y − h| ≤ T_chip + T_host,     T_host = u32·|r| + 2·γ(n + 64)·M + 2^-149   (contract above)
    diagnostic:  |y − r| ≤ T_chip + 2·γ(n + 64)·M + 2^-149

T_chip bounds the kernel against exact arithmetic. Each one below is derived from the kernel source
at the pinned CUDApp sha, does not depend on summation order, and was fixed before any chip run.
Each also carries an absolute 2^-124 for products that the chip may flush below 2^-126. Notation:
u = 2^-24, g(n) = n·u/(1 − n·u) (the f32 γ).

Hardware constants come from PTX ISA 9.4. `ex2.approx.f32` is within 2 ulp of the correctly
rounded result, which gives at most 5u relative. `rcp.approx.f32` is within 1 ulp, which gives 2u.
`rsqrt.approx.f32` is within relative 2^-22.9, and Wolfram confirms 138·2^-30 ≥ 2^-22.9, so that
value is used.

- **rmsnorm** (`omega_gpu_rmsnorm_f32`, dim % 128 = 0, dim ≤ 16384). The sum of squares is a
  sequential sum over dim/128 elements per thread, followed by a shuffle and CTA tree, so its
  relative error is g(dim/128 + 11). An input relative error g moves x^-1/2 by at most g/(2(1 − g)).
  The MUFU rsqrt with one Newton step contributes 3.0000007u, and the two output products add
  (1 + u)^2:
  R = (1 + gs/(2(1 − gs)))·(1 + 3.0000007u)·(1 + u)^2 − 1, with gs = g(dim/128 + 11), and T = R·|r|.
  The host term uses n = dim and M = |r|. The test passes eps = 2^-20 to both sides.
- **rope** (`omega_gpu_rope_f32`, rotate-half, rot = D). Each output is x1·c − x2·s or x2·c + x1·s,
  so T = g(2)·(|x1||c| + |x2||s|), which holds with or without FMA. The host term uses n = 2.
- **swiglu, silu** (`omega_gpu_swiglu_f32`, |g| ≤ 80; silu is the same kernel with up = 1).
  exp(−g) is computed as ex2(−g·log2e). With ε = exp(|g|·g(2))·(1 + 5u) − 1, the bound is
  R = (1 + u)^2·(1 + 2u)·(1 + ε/(1 − ε))/(1 − u) − 1 and T = R·|r|.
- **attention**: decode (`omega_gpu_gqa_attention_f32`), causal prefill rows
  (`omega_gpu_gqa_attention_rows_f32`), paged batch decode
  (`omega_gpu_paged_attention_batch_bf16`), and softmax computed through the decode kernel.
  - Notation: Λ is the row maximum of scaled |q|·|k|, nch = ⌈ctx/D⌉, and S is the split count.
    S = 1 for the direct paths. For the paged path S is the batch maximum of the kernel's
    own_log2split.
  - Logit error: a logit carries absolute error at most Λ·g(D + 3), and the max subtraction
    doubles it, so each weight has relative error ε_p = exp(2Λ·g(D + 3))·(1 + 5u) − 1.
  - Rescales: each chunk rescale adds ε_α = exp(2Λu)(1 + 5u) − 1.
  - Split combine (S > 1): the host combine with host_exp2 (Cephes in f32, 24.27u, Wolfram below)
    adds ε_w = exp(2Λu)(1 + 25u) − 1.
  - Combined: η = (1 + ε_p)(1 + ε_α)^(nch − 1)(1 + ε_w) − 1, α = (1 + η)(1 + g(ctx + 2nch + 2S + 8)) − 1,
    and φ = (1 + 2u)(1 + u) − 1.
  - Result: R = (φ + α(2 + φ))/(1 − α) and T = R·A, where A = p·|v|.
  - The host term uses n = D + keys and M = A(1 + 2Λ) + |r|, the same as test_ops.
  - Paged inputs: K and V are bf16-representable f32, so the bf16 pool holds exactly the values
    the host sees.
  - Softmax mapping: there is no standalone softmax kernel, so softmax(x) for c ≤ 64 logits runs
    through the decode kernel with D = 64, q = 8x, and K = V = I[:c]. The kernel's scale 1/8 is
    exact, so out[:c] = softmax(x) and A = |r|. One case in two multiplies x by 300, far past exp
    overflow.
- **matmul** (`omega_gpu_gemm_ex`, f32 inputs, compute DEFAULT, which routes to bf16x3).
  - Split: each f32 splits exactly into three bf16 planes (the source comment at
    src/omega_gpu_gemm_ex.c:85), and all 9 plane products are exact.
  - Accumulation: one HMMA.16816 accumulation step costs at most
    c = (17·2^-25 + 2^-23)(1 + 2^-20) relative under the chip-exact model {16,2,RZ,nominal}.
  - Steps: there are steps(K) = 9(⌈K/16⌉ + p) + p accumulation steps, with p = ⌈K/1808⌉ + ⌈K/16384⌉.
  - acc = ((1 + c)^steps − 1)·|A||B| + 2^-124·(rowsum|A| + colsum|B|) + K·2^-140. The last two terms
    cover the split parts below 2^-133, as in CUDApp#139.
  - T = acc + 2^-52(|A||B| + |bias|) + u(|r| + acc) + 2^-149. The bias enters through
    C = 1·acc + 1·C.
  - The host term uses n = K and M = |A||B| + |bias|.

**Negative control on chip.** Every op runs CUDApp's own test mutants on cases from the same
sweep, and each mutant must fail the gate. A mutant joins a case only where it changes the
function; an equivalent mutant cannot fail any gate.

- elementwise: 3 drops the last rmsnorm shuffle step, 4 computes rope as q0c + q1s, and 5 computes
  swiglu with exp(+g).
- attention:
  - 1 (NO_MAX) skips the max subtraction. It joins only softmax rows whose max logit x has
    x·log2 e outside [−126, 128] (half the cases are ×300). Softmax is shift-invariant, so inside
    the f32 exp range NO_MAX computes the same function.
  - 2 (KV_HEAD) uses the wrong kv head. Needs nkv ≥ 2.
  - 3 (SLOT) shifts K by one slot against V. Paged calls only, every sequence with ≥ 2 tokens.
  - 5 (Q_ROW) and 6 (OUT_ROW) swap rows h and h ^ 1. Need nq ≥ 2.
  - 4 (NO_RESCALE) leaves the accumulator unscaled when the running max moves. The host model
    `nores_output` runs CUDApp's chunk loop (D tokens per chunk, src/omega_gpu_attention_api.c:571-572)
    without the rescale. The mutant joins a case only if that model puts it at gate ratio ≥ 4.
    Paged calls run it with one CTA per head (`omega_gpu_attention_test_set_split(0)`), because
    cut J gives every split CTA one chunk and the rescale never runs (CUDApp
    tests/gpu_attention_test.c:283-285).
- gemm_ex: 1 drops the last k (used only for K ≤ 256), and 2 reads op_b as N (trans_b cases).

Each receipt records, per mutant run, the shape, rc, gate violations, the measured ratio and,
for NO_RESCALE, the predicted ratio. A first attention run placed NO_MAX and NO_RESCALE on
equivalent cases, and 10 of 32 mutant runs survived. That run is kept in
evidence/GPU-9841bb3a/LOSSES.md.

Numbers, all from Wolfram (`WolframLanguageEvaluator`, 2026-10-27; u, g as above):

| query | result |
|---|---|
| `138/2^30 >= 2^-22.9` | True |
| `Maximize`/`Minimize` of (1+e)(3/2 − (1+e)^2(1+d1)(1+d2)/2)(1+d3)(1+d4) − 1 over \|e\| ≤ 138/2^30, \|di\| ≤ u (rsqrt + Newton) | 3.0000002384u, −3.0000006522u → use 3.0000007u |
| `Reduce[ForAll[g, 0<g<1/2, (1-g)^(-1/2)-1 <= g/(2(1-g)) && 1-(1+g)^(-1/2) <= g/(2(1-g))], Reals]` | True |
| host_exp2: max \|P(f)/2^f − 1\| on [−1/2, 1/2] with the f32 Cephes coefficients; Horner rounding max g(12)·ΣP\|/P | 1.5920354e-8; 1.4305125e-6; total 1.4464328e-6 = 24.267u → use 25u |
| `(17 2^-25 + 2^-23)/2^-22`; `c/2^-22` | 21/8; 2.62500 |
| R_rms(128)/u, R_rms(16384)/u | 11.0000, 74.5012 |
| R_silu(\|g\| = 4)/u, R_silu(\|g\| = 80)/u | 18.0000, 170.003 |
| R_attn(ctx 64, D 64, S 1, Λ 10); R_attn(ctx 4096, D 128, S 16, Λ 10); R_attn(ctx 64, D 64, S 1, Λ 900) | 1.69597e-4; 9.12223e-4; 1.45434e-2 |
| steps(1), steps(300), steps(2048) | 29, 191, 1182 |
| (1 + c)^steps − 1 at K = 300, K = 2048 | 1.19544e-4, 7.40027e-4 |

**Scope.** The bounds are worst case, so measured chip errors sit well below them. The gate proves
that the chip result is a correct f32 evaluation of the same math within its own rounding
behaviour. It does not prove bitwise agreement, and the mutants show how far it is from vacuous.
Ops and shapes with no CUDApp kernel are listed in README.md ("Not on the GPU") as NOT_RUN or unsupported.
