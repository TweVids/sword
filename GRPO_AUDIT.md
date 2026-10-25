# GRPO, bridge, scorer, and Model B audit — 2026-10-25

Scope: GRPO gradients, trainer/rollout policy alignment, LoRA synchronization,
Model B scoring, vLLM compatibility patches, and memory lifecycle. Sword serving
engine optimization is outside this audit.

## Confirmed findings

- Fallback nucleus sampling draws positions in a sorted probability tensor but
  treats them as vocabulary IDs; sampled tokens and recorded probabilities are wrong.
- Fallback reports temperature/top-p processed probabilities while default vLLM
  reports raw log-probabilities; detached importance weights compare different policies.
- Bridge entropy is computed under `no_grad`, so its floor penalty cannot train
  the policy. Microbatch loss metrics also average already normalized losses.
- vLLM wakes before trainer offload, and failed vLLM generation invokes fallback
  without restoring the trainer or sleeping vLLM. Memory errors are swallowed.
- Model B evaluation starts while vLLM is awake; harness follow-up generation can
  run alongside a resident judge. The memory guard's result is ignored.
- Multi-turn harness output replaces originally sampled token IDs and pads
  rollout log-probabilities with zero. The resulting gradient targets were never
  sampled under the recorded prompt/policy.
- Initial and resumed trainer adapters are not synchronized before first rollout.
  Missing vLLM token log-probabilities silently become probability-one values.
- Model B object/array parsers accept nonfinite and unbounded scores, and interpret
  the string `"false"` as true. Empty/malformed responses receive high default scores.
- GDPO sends pre-aggregated column names through the component mapping and treats
  formatting, efficiency, and safety as accuracy; nested score dictionaries fail.
- RL chunked loss first runs the full causal LM with all hidden states and logits,
  undermining the claimed memory savings. Its length alignment silently truncates.
- vLLM rotary fallback ignores sequence offsets, packed positions, partial rotary
  dimensions, and interleaving. Heterogeneity patches hard-code indentation and
  can corrupt Python or silently choose an incompatible head dimension.

## Implemented fixes

- Fallback nucleus samples map back to vocabulary IDs. Both rollout engines now
  supply raw model log-probabilities; incompatible processed-probability bridge
  configurations are rejected. Missing, nonfinite, or misaligned rollout
  probabilities trigger fallback or an explicit error. Stale policy versions
  cannot be reused for a new bridge update.
- Shared `sword/rl/policy.py` runs supported HF text decoders without the full
  causal-LM output head. Checkpointed vocabulary chunks avoid retaining their
  large log-softmax tensors for backward. Other model architectures use their
  own forward to preserve custom output behavior. Token values and parameter
  gradients are compared against full HF forwards, including real PEFT adapters.
- Entropy floors now provide gradients and reduce by global valid-token count.
  Microbatch losses are summed, and monitoring statistics are token weighted.
  Reference and old-policy targets are detached. Incorrect tensor lengths and
  nonfinite updates are rejected. Optional importance-weight score centering
  requires a single microbatch because its statistics span the complete group.
- Trainer offload precedes vLLM wake, including initial startup. Failed vLLM
  rollouts sleep the engine and restore the trainer before fallback. Sleep/offload
  failures propagate. VRAM free-space reporting reads actual device free memory.
- Model B is loaded after every harness generation finishes, with vLLM asleep.
  Its local forced-GPU memory guard controls whether evaluation may issue a
  request. Cleanup unloads the judge on exit. OOM retry clears gradients and
  traceback-held graphs and halves microbatch/chunk sizes once.
- Optimizer updates snapshot trainable weights and optimizer state on CPU, then
  restore both if a partial update fails. Bridge adapter export participates in
  that transaction, so failed export cannot leave a new policy behind an old
  rollout version. Continuous RL and standalone math use the same update helper.
- Harness final text supplies the outcome reward, while the original sampled
  first turn supplies gradient targets. Follow-up text is no longer substituted
  into the first turn or given fabricated probability-one entries.
- Initial/resumed LoRA adapters are exported before first vLLM rollout. Failed
  exports do not advance policy versions. Manual keys use PEFT naming. Full-weight
  updates use the shared in-process fallback. Long-trainer adapter files live
  under its output directory and are cleaned up, avoiding dependence on a full
  system temporary drive. Requested LoRA target modules are honored by HF setup.
- Model B object/array/regex results are bounded and finite, quoted booleans parse
  correctly, partial results give no unearned default scores, and empty results
  trigger deterministic fallback. The Ollama payload requests JSON. Connection
  checks match the exact requested model/tag. Numeric substring false positives,
  decimal extraction, and scientific notation were fixed in ground-truth scoring.
  Numeric-zero dataset solutions remain `"0"` instead of being treated as missing.
- GDPO accepts raw components, aggregated columns, and nested score dictionaries
  without folding formatting/efficiency/safety back into accuracy.
- Continuous RL receives exact sampled prompt/response IDs and raw probabilities
  from serving metadata. Requests needing probabilities disable the speculative
  route that omitted them. Legacy text-only servers use shared-policy fallback.
  Adapter resume updates the existing model so optimizer references stay valid.
  Serving edits are restricted to this training metadata contract.
- Both standalone math entry points use the shared chunked statistics and retain
  exact generated IDs. Training now uses the sampled formatted prompt. Single-row
  loss respects response padding.
- Rotary fallback implements offsets, packed boundaries, partial dimensions,
  interleaved rotation, conjugation, and in-place calls. Patches validate syntax
  before atomic writes, preserve indentation, upgrade the old rotary block using
  Python AST boundaries, and are idempotent. Global heterogeneous dimensions are
  accepted only when every layer agrees. The patcher no longer rewrites Transformers
  to suppress heterogeneous-configuration errors. vLLM installation targets the
  running Python interpreter. Automatic partial `text_config` overrides were removed.
- Failed real-model loads raise instead of silently training a synthetic model.
  Synthetic trainers remain available through explicit mock/test configuration.

## Validation

Baseline: 108 passed, 10 failed because `C:` had zero free disk space. Validation
uses `D:/sword/scratch/audit-temp` via `TEMP` and `TMP`, and `HF_HUB_OFFLINE=1`.
Final full suite: **146 passed** across `test_grpo_audit.py`, `test_grpo_bridge.py`,
`test_new_trainer.py`, `test_rl_engine.py`, and `test_offline_math_grpo.py` (136.23 s).
That run explicitly mocked judge availability as false and set CPU threads to one.
After the final numeric-zero fix, the audit and bridge regression rerun passed
**36 tests** (17.28 s). The new audit file contains 29 parameterized test cases.

Environment: CPU PyTorch, Transformers, PEFT, pytest; no vLLM installation or
live GPU/Ollama evaluation. Rotary math, patch injection, LoRA dispatch, memory
handoffs, score parsing, first-turn preservation, and OOM retry are covered by
CPU fixtures or mocks. Compilation and `git diff --check` pass.

## Remaining limits and follow-up targets

- Live vLLM/Unsloth/CUDA/Ollama behavior and peak VRAM still need hardware testing.
  Unknown model architectures retain full-forward logits; chunking their CE does
  not eliminate that allocation. No fixed GB-memory claim is justified here.
- These loops perform one update per rollout. The bridge's detached current-policy
  baseline is intentionally identity-valued on that update; reusable PPO epochs
  would require a separately frozen trainer-policy snapshot and different version
  validation. Calibrated clipping has no variance signal in this one-update flow.
- Importance correction compares raw backend log-probabilities. Sampling still
  uses temperature/top-p; this is not a general correction for arbitrary changes
  to the behavior sampler. Bridge processed-logprob modes are rejected explicitly.
- Multi-turn training currently gives first-turn actions the terminal harness
  reward. Training every assistant turn needs per-turn prompt/ID/logprob records
  and masks that exclude user/tool tokens; no probabilities are fabricated.
- Update rollback consumes host memory proportional to trainable parameters and
  optimizer state. Real CUDA OOM recovery still needs hardware validation; CPU
  tests verify restoration after partial weight/state mutation and export failure.
  A rollback failure raises an explicit error and suppresses automatic retry.
- Explicit `auto_unload_B_model=False` retains the judge as requested; memory
  co-location then depends on the caller's GPU budget. Cache clearing does not
  free live trainer parameters or optimizer state.
- Deterministic judge fallback remains a heuristic for non-math truth and rule
  compliance. A live judge is necessary for semantic validation.
- Truly heterogeneous head dimensions require native per-layer vLLM support.
  Unsupported patch layouts and old malformed head-dimension patches return
  errors rather than silently selecting the first layer or corrupting code.

Both math entry-point files remain identical. Final Python compilation and
`git diff --check` passed. No live package installation or external publication
was performed during validation.
