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
