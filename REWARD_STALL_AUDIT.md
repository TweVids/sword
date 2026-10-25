# Zero loss / zero gradient audit

Investigated the report:

```text
Step 0001/0005 | Loss: 0.0000 | GradNorm: 0.0000 | Avg Reward: 0.225 | Truth: 0.00 | Harness: opencode | Diff: medium
```

The producer is `sword/long_trainer.py`. The average reward alone does not prove
that every rollout received the same reward. A controlled reproduction with
uniform rewards of 0.225 does produce zero advantages and zero gradients.
Separately, decimal rounding could conceal small gradients or losses. A centered
surrogate loss can also equal zero while its derivative is nonzero.

## Confirmed bugs and changes

- The first two diagnostic rows are coding tasks, but were marked for exact
  numeric verification. This made the scorer ignore Model B's semantic truth
  scores and compare generated code with a reference implementation through
  `MathVerifier`. Both rows now use semantic judging; the numeric math row
  continues to use exact verification.
- The long trainer duplicated group normalization and bypassed the bridge's
  historical-baseline handling and degeneracy warning. Both entry points now
  call `GRPOBridge.compute_advantages`, which validates rewards and records
  within-group standard deviation, reward bounds, maximum absolute advantage,
  and counts of uniform groups and historical-baseline uses. The long-trainer
  argument now reaches the bridge configuration.
- Disabling `use_cross_step_baseline` after an EMA had been seeded still used
  the EMA. Baseline selection now respects the configuration at calculation
  time. The first uniform batch also previously claimed to use a baseline that
  was only seeded after its zero advantages were calculated; its warning now
  reports the actual absence of reward signal.
- AdamW applied momentum and weight decay with all-zero gradients, changed
  weights, and exported another adapter version. Such steps now preserve policy
  weights and optimizer state, keep the policy version, and explicitly report
  `optimizer_step_skipped=True`. Loop-step counting continues independently.
- Loss metrics were rounded to five decimals and gradient norms were rounded
  before logging. Metrics now retain their precision; logs use six significant
  digits and show `RewardStd`, judge source, and optimizer-update status.
- `b_model=False` passed `None` to the scorer, which constructed a default
  Ollama judge anyway. Explicit `None` now selects local deterministic scoring;
  omitting the constructor argument still selects the default judge.
- Live diagnostics accepted entirely zero-gradient runs because they checked
  only finiteness and increasing adapter versions. They now reject runs without
  nonzero gradients or, when available, nonzero reward advantages. A skipped
  first step and genuine zero-loss updates remain valid. OOM retries restore
  the reward baseline instead of treating failed attempts as reward history.

## Verification

`test/test_reward_stall.py` exercises actual forward/backward and optimizer
updates with a small CPU policy, plus the long-trainer scoring path. It checks
Adam state preservation after prior momentum, tiny metrics, a valid centered
zero-loss update, both baseline settings, changed rewards after an OOM retry,
semantic coding truth, judge disabling, and five constant-reward steps. The
five-step reproduction preserves weights, reports five skipped updates, and
fails live-training validation.

The final focused regression run passed **137 tests**, with two explicitly
deselected broader-suite checks listed below. The additional trainer/math/RL
suite passed **111 tests**. Validation used the workspace CPU PyTorch environment,
one CPU thread, offline Hugging Face access, workspace temporary files, and
mocked judge availability for the latter suite. Python compilation and
`git diff --check` pass.

## Remaining limits

- A historical baseline supplies a signal only when reward levels change;
  it cannot distinguish equally rewarded answers. Disable it for strict GRPO.
  If `RewardStd` and all advantages remain zero, investigate the candidate
  responses and individual judge subscores instead of expecting a gradient.
- This audit did not rerun the user's live GPU/vLLM/Ollama workload. Its exact
  reward distribution and judge output remain unverified.
- The existing microbatch-equivalence regression fails with identical weights
  on `HEAD:sword/grpo_bridge.py` and this patch. TIS weights are normalized
  separately in each microbatch, so changing the microbatch size changes the
  estimator; this is not merely a floating-point tolerance issue. It remains
  outside this zero-update patch.
- The head-dimension regression cannot import
  `transformers.integrations.heterogeneity` in the workspace's Transformers 5.5
  installation. No dependency installation was performed.
- Dataset completion bookkeeping records visited loop rows; it does not certify
  learning on each row. Use the new update counts and live diagnostic gate to
  assess learning. `allow_retest=True` permits another pass over such records.
