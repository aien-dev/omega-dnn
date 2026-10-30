# Earlier chip runs that failed (kept, not hidden)

Both ran on the GB10 at CUDApp 9841bb3a under the same lock discipline as the receipts in this
directory. Their logs were overwritten by the reruns (`tee`). The lines below are copied from
the session's tool output.

## Run 1, group ew, 2026-10-27 08:27:45Z (BOARD GPU-START/GPU-END), FAIL before any case

`AttributeError: ... libcudapp_dnn.so: undefined symbol: omega_gpu_gemm_ex_test_set_mutant`.
Cause: a harness bug. `set_mutant` built a dict of all three mutant setters eagerly, and the main
CUDApp library has no gemm_ex. Fixed: setters are resolved by name only when used, and every hook a
run needs is resolved before the hold is taken. No case result came from this run.

## Run 2, group attn, FAIL: every gate passed, 10 of 32 mutant runs survived

```
softmax_via_attn           cases= 32 gate_worst=0.0129 gate_viol=0 ...
attention_decode           cases= 32 gate_worst=0.00447 gate_viol=0 ...
attention_prefill_causal   cases= 32 gate_worst=0.01 gate_viol=0 ...
paged_decode_batch         cases= 32 gate_worst=0.00432 gate_viol=0 ...
paged_decode_single        cases= 32 gate_worst=0.00311 gate_viol=0 ...
MUTANT softmax_via_attn:attn:1            runs=4 killed=1 refused=0 survived=3
MUTANT paged_decode_single:attn:4         runs=4 killed=1 refused=0 survived=3
MUTANT attention_decode:attn:4            runs=1 killed=0 refused=0 survived=1
(every other mutant row: killed=runs)
RESULT FAIL hold=2.6s
```

The survivors were equivalent mutants. The harness had placed them on cases where the mutated
kernel computes the same function within the gate, so no gate could tell them apart.

- **NO_MAX (1)** on softmax rows whose logits stay inside the f32 exp range. Softmax is
  shift-invariant, so dropping the running-max subtraction changes nothing until 2^(x log2 e)
  overflows or underflows. CUDApp's own test notes the same thing (tests/gpu_attention_test.c:523).
  The three survivors had max |x| of 4.4, 1.6 and 3.2; the one kill had 1165.6.
- **NO_RESCALE (4)** on paged calls. Cut J gives each split CTA one chunk, so the in-kernel rescale
  never runs (tests/gpu_attention_test.c:283-285). On decode, the one case (1q1kv, hd 64, ctx 155)
  had a running max that did not move enough between chunks to leave the gate.

Fix, in the harness only, with no gate change: a mutant joins a case only when the host predicts
it is visible there.

- NO_MAX joins a case only when the max logit passes the f32 exp range.
- NO_RESCALE joins only when the host model `nores_output` of the mutant predicts gate ratio ≥ 4.
  The paged NO_RESCALE runs are forced to one CTA per head (split 0), as CUDApp's test does.
- The head-wiring mutants KV_HEAD, Q_ROW and OUT_ROW were added, and SLOT on paged calls whose
  every sequence has at least 2 tokens.

Each receipt records the predicted ratio of every mutant run beside the measured one.
