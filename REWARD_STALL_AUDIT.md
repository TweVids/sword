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

## Follow-up: persistent 0.200 rewards

The user reported another uniform 0.200 reward on step three with the live judge,
and confirmed that raw judge JSON was not printed. This demonstrates a persistent
learning stall; it does not establish whether the candidate outputs or the judge
are at fault. No live response/JSON artifact was available in this workspace.

Further source inspection confirmed these issues:

- The bridge and shared PyTorch fallback passed plain prompts to the instruction
  model without its configured chat template. Both now include the assistant
  generation prefix; vLLM receives exact token prompts to avoid duplicate BOS.
  The original prompt text still drives the harness and the exact sampled prompt
  IDs still drive training. Token budgets reserve template overhead.
- Harness trimming estimated four characters per token without verifying the
  final count, and could retain an oversized first tool-definition line. It now
  recounts actual tokens, removes complete tool lines that cannot fit, preserves
  the complete task, and explicitly rejects tasks that cannot fit at all.
- Five-step diagnostics allowed only 128 completion tokens, potentially ending
  during reasoning. Their default is now 512, with explicit overrides honored.
- Empty output received full structural-formatting credit, potentially yielding
  0.200 for agentic responses whose other scores were zero. Empty formatting
  scores and empty deterministic-fallback correctness/rule scores are now zero.
- Live judging accepted partial JSON objects and reported the judge model as
  successful while missing accuracy/rule fields defaulted to zero. The permissive
  parser remains available, but live evaluation retries responses missing any
  of its four primary scoring fields. Complete all-zero judgments remain valid.

Stalled-step logs now print candidate previews and judge JSON, plus response
diversity, empty/truncated counts, tool-call totals, policy/entropy loss, and
gradient-buffer connectivity. `rollout_details.jsonl` preserves the evidence;
diagnostics also save full responses, exact prompt/response IDs, logprobs, and
tool history. File IO occurs after committing the loop step, so trace errors
cannot checkpoint an updated policy under the preceding step number.

The follow-up regression tests use a real local HF chat tokenizer, exercise
vLLM token-prompt dispatch and fallback, test incomplete versus legitimate
all-zero judge JSON, verify prompt budgets, and read back persisted rollout
evidence. The focused suite passed **149 tests** with the same two broader-suite
checks deselected. The additional trainer/math/RL suite passed **111 tests**
with judge availability mocked offline, for **260 passing CPU tests** in this
follow-up. Compilation and `git diff --check` pass.
Live GPU/vLLM/Ollama output still requires verification after
loading this updated bridge in the target runtime.

## Follow-up: harness continuation, workspaces, and loss sign

The user reported repetitive bitsandbytes cast notices and suspected that
harnesses ended after issuing tools. Source inspection found that Codex stopped
after any turn without failed tools, even when there was no final answer. The
other adapters used short default turn limits, and all four reported completion
even when the last turn was still executing tools. They now default to eight
turns, feed observations back to the model, and report a turn-limit result as
incomplete. The long trainer passes `harness_max_turns` and scores unfinished
results with an empty final answer rather than the original tool-call text.
Explicitly labelled final code answers can finish after successful implicit code
validation; a final label inside closed thinking tags does not trigger that exit.

One workspace already persisted across all turns of each `harness.run` call;
it was not wiped between turns. Each rollout still creates a separate directory,
now under `<output_dir>/harness_workspaces`, and removes it in `finally` by
default. `keep_harness_workspaces=True` retains separate directories for
inspection. Explicit caller-owned `workspace_dir` projects remain reusable and
are never automatically deleted. Cleanup verifies the owned temporary target,
and Python snippets use unique temporary runner names instead of overwriting
the project file `_agent_runner.py`.

Increasing the turn limit also requires bounding follow-up context. The trainer
rebuilds follow-up prompts with the original problem and tool observations,
removing the oldest turns only when needed. An oversized latest turn retains
its beginning and end with an explicit trim marker. Turn counts, stop reasons,
workspace lifecycle, and assistant responses are recorded in rollout evidence.

The exact bitsandbytes cast notice is filtered for both logging-based and older
warnings-based releases, without suppressing other warnings or ERROR records.
The package installs the filter before loading the training dependencies.

No loss clamp was added: a negative GRPO policy loss is mathematically valid.
A real forward/backward test demonstrates that minimizing negative policy loss
increases the probability of a positively rewarded token. Uniform-reward logs
now show the weighted entropy penalty alongside policy loss, so a tiny residual
total loss is not mistaken for reward-driven learning. Zero gradients and skipped
optimizer steps still indicate no policy update.

The combined focused suite passed **168 tests**, with the same two broader-suite
checks excluded above. The new harness tests execute write/edit/read operations
through all four adapters, verify persistence between turns and isolation between
runs, and cover turn-limit scoring, final code validation, retained projects,
runner collisions, targeted notice filtering, and follow-up prompt budgets.
The trainer/math/RL compatibility suite passed **111 tests** with judge
availability mocked offline, for **279 passing CPU tests** in this follow-up.
Python compilation and `git diff --check` also passed.
The target GPU/vLLM/Ollama workload has not been rerun; these changes must be
loaded in that runtime before assessing live rewards or gradients.

## Follow-up: reward variation with zero gradients

The next GPU report showed reward standard deviation `0.0661438`, total loss
`4.95836e-06`, and another zero-gradient skipped update. The average within-group
reward variation means the uniform-reward explanation is insufficient. The
status line alone does not identify whether gradients cancel, the training
adapter is disconnected, or the token statistics/importance weights are
degenerate. The nearby diagnostics and active bridge version were requested.

Source inspection found a training gap: only the first sampled turn's tokens
received policy gradients, even though scoring used the final harness outcome.
Later actions and answers were generated without preserving their token/logprob
data for training. A controlled CPU reproduction with identical first turns and
different rewards produces exactly zero gradients despite nonzero advantages.
Adding different follow-up tokens produces a nonzero update that increases the
rewarded token's probability. This demonstrates a real failure mode, not proof
that it explains this particular live run.

The bridge now retains follow-up samples and trains all assistant turns with
their exact prompt tokens, response tokens, raw logprobs, and policy version.
Grouping and reward normalization remain at trajectory level; each trajectory's
advantage applies to its sampled assistant tokens. Tool observations are prompt
context rather than gradient targets. Global token reduction counts every
assistant turn, and version/logprob validation covers follow-ups. This increases
training work with the number of harness turns while preserving microbatch
memory limits. Single-turn training retains its previous behavior.

Skipped steps now emit detailed evidence even when rewards are nonuniform. The
logs separate missing gradient buffers from connected all-zero buffers, show
maximum advantage, first-turn and trajectory diversity, trained token counts,
policy/rollout logprobs, and zero importance-weight fraction. Autograd-disconnected
losses fail explicitly. Exact follow-up data and advantages are written into the
trace. The new startup stamp is `sword-grpo-bridge trajectory-gradients/v4`.

`test/test_trajectory_gradients.py` includes actual backward/optimizer tests for
first-turn cancellation, later-turn reward preference, and a real local PEFT
LoRA adapter. It also exercises importance-weight underflow, disconnected
trainable parameters, follow-up validation, trainer capture of exact sampled
data, and logging a skipped step with nonuniform rewards.

The expanded focused suite passed **176 tests**, with the same two known checks
deselected above. The trainer/math/RL compatibility suite passed **111 tests**
with judge availability mocked offline, for **287 passing CPU tests**. Python
compilation and `git diff --check` passed. The user's GPU run has not been
reproduced locally; this change fixes the demonstrated trajectory-training gap
and exposes the remaining evidence needed to diagnose that runtime.

## Follow-up: warning resets, visible generations, and complete agent loops

The new audit inspected the current worktree rather than assuming previous fixes
were loaded in the GPU process. The native bitsandbytes 0.50.2 `MatMul8bitLt`
forward emitted the exact cast notice and completed backward with nonzero input
gradients on CPU. Its logging-based emitter is independent of the warning filters
used by older releases. The earlier installation also happened only before
dependency imports, and logger/warning-policy resets could bypass it. A targeted
guard now protects the emitting logger, preserves other warnings and ERROR
records, handles older UserWarning emission, and is reinstalled at runtime
boundaries. Tests exercise the actual emitting forward/backward path, logger
filter removal, and a subsequent `warnings.simplefilter("always")` change.

Generation visibility previously depended on stalled-step printing. Model A now
prints each actual initial/follow-up generation immediately after it returns,
including backend, policy version, prompt/response token counts, truncation,
and bounded text. Model B prints every request and received response before
parsing, including empty final content, thinking previews, generated token counts,
retry failures, parsed scores, and deterministic fallback. Tool results and turn
labels print during the loop. Tracing preserves all training samples and judge
attempts, including failed attempts rather than only the successful JSON.

Further harness inspection found that nonempty thinking/planning or unrecognized
tool text could still end a run as if it were a final answer. Gemini's brace-based
regex could miss valid calls containing braces or nested arguments; Pi's Action
parser could miss a final line without a newline. Shared JSON decoding now accepts
tagged, fenced, bare, and function-style calls with nested/stringified arguments.
Thinking is excluded from executable actions. Incomplete calls, plans, and
thinking-only output receive observations and another model turn; completion
requires an explicit final label and result. Real tool results feed the next
generation, and every assistant turn remains part of trajectory training.

The strongest local integration runs the actual PyTorch sampler against a tiny
controlled causal policy, executes write/edit/read/verify operations through each
of the four adapters, grades responses through a real local HTTP server using the
Model B client, then performs backward/optimizer updates. For each adapter, four
rollouts generate five assistant turns each, execute sixteen successful tools,
receive four judge responses, and train all twenty sampled turns. The rewarded
final-token probability increases. Generation, tool execution, scoring transport,
and training are not mocked; the policy's behavior and HTTP judge's score rule are
controlled fixtures, so this verifies dataflow rather than real coding ability.
A separate test attaches a real PEFT LoRA adapter to bitsandbytes 8-bit projections
in a tiny Qwen model and confirms reward-driven gradients and adapter changes.

The active source stamp is `sword-grpo-bridge agent-output/quantization-v5`.
At the start of this audit, `HEAD` was `0f1a508` and these fixes were uncommitted.
Older checkouts and installed packages need the updated source. The active GPU
stamp and installation method were requested. No live
Gemma/vLLM/Qwen-judge run has been performed in this CPU workspace.

The final focused suite passed **195 tests**, with the same two known checks
deselected. The four sampled-policy integration tests passed, and the
trainer/math/RL compatibility suite passed **111 tests** with judge availability
mocked offline, for **310 passing CPU tests**. Python compilation and
`git diff --check` passed.

`artifacts/grpo_agent_debug_patch.zip` contains the current changed runtime source,
new tests, documentation, a hash manifest, and an installer. The installer was
verified against an isolated `HEAD` package: check-only left it unchanged, copying
installed all **14 runtime files**, all copied hashes matched, the newly required
modules were present, and original files were preserved as backups. It finds the
installed package without importing Sword; the target process must be restarted
after installation. The user's `run_prep_logs.py`, model weights, checkpoints,
and credentials are excluded.

| Requested behavior | Authoritative local evidence |
| --- | --- |
| Hide the MatMul8bitLt notice while retaining errors | Native bitsandbytes forward/backward test, logger-reset test, and old UserWarning reset test |
| Print both models' responses and judge retries | Real PyTorch sampling and local HTTP client/server tests, including thinking-only output and final JSON |
| Execute and train the full agent loop | Four adapter integrations each sampled/trained 20 assistant turns and executed 16 real tools |
| Deliver the actual local source to another runtime | Bundle hashes and installation over an isolated original package, with backups |

These checks establish the implementation and its tested dataflow. They do not
establish the large GPU policy's coding quality, the live judge's responses, or
the exact cause of that runtime's earlier zero-gradient status line.
