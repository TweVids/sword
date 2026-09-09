# Bounty-Driven Continual RL Engine — Consolidated Spec (7B Model)

Revision of the original executive spec after a full pass to fix reward-hacking vulnerabilities, remove internally contradictory rules, and extend coverage beyond math/code into the domains real users actually ask about.

---

## 1. Scope and philosophy

**Target**: a focused 7B model, trained via continual RL over a 2–3 month streaming window, that is reliably good at a defined set of tasks rather than attempting general-purpose parity with much larger models.

**In scope**: reasoning, math, code editing, mechanistic/textbook science, everyday practical tasks, writing with constraints, multi-turn coherence, ambiguity handling, tool-use correctness, and (optionally, later) a thin vision-language attachment for spatial/structural tasks.

**Explicitly out of scope**: computer-use / screen perception-action loops (too unreliable at 7B scale, feedback loop too slow for this reward architecture), patient-specific medical guidance, and anything that would optimize the model toward exploit generation rather than defensive/educational vulnerability understanding.

**Core design principle carried through every section**: static rules only get used where a claim is genuinely binary or countable. Anything requiring semantic judgment (is this explanation good, is this design tasteful, is this emotionally appropriate) goes to a low-frequency learned-judge or human-review pass instead of a hand-written string/regex check. Trying to fake semantic judgment with keyword counting is the single biggest reward-hacking vector in the original design, and it recurs in almost every section below as the thing being fixed.

---

## 2. Continuous streaming pipeline

- Data streams continuously from a queue (Kafka/RabbitMQ-style); no halting between batches.
- **Failure-driven requeue, with a taxonomy.** Every trajectory scoring below 0 gets tagged with a reason code before requeuing:
  - `timeout` / `format_violation` / `wrong_answer` → auto-requeue next cycle.
  - `ambiguous_prompt` → routed to a review queue, not blindly repeated. Repeating a bad or underspecified prompt just trains the model against noise labeled as signal.
  - `safety_violation` → never requeued as a training-to-avoid example in the normal loop; handled by the safety override path (Section 8).

### 2a. Bounded retry with variation (not repeat-until-high-score)

Retrying failed problems is legitimate curriculum learning — it concentrates gradient updates on genuine weaknesses instead of wasting them on already-solved items. But "requeue the identical question until it scores high" is itself a reward-hacking vector (see Section 15.5): unbounded attempts on one fixed input turn training into a search process that can find a reward-checker blind spot or memorize one input/output pair rather than generalize a fix, and a rising score across repeated identical attempts doesn't actually tell you the underlying skill improved.

**Fix — bounded retry, then generalize, not literal repetition:**
1. **Attempt cap per literal question.** A specific failed question gets at most a fixed number of retry attempts (e.g. 2–3) as itself. After that cap, it is retired from direct requeue regardless of outcome — no unbounded loop chasing a high score on one fixed item.
2. **After the cap, generate a similar-but-varied question from the same skill/difficulty/domain cluster** (changed surface details — different numbers, different variable names, different phrasing, same underlying skill and difficulty tier) and requeue *that* instead of the original. This is the actual test of whether the fix generalized rather than pattern-matched the literal input.
3. **Track pass/fail at the cluster level, not the item level.** The metric that matters is the model's trend across many varied examples of the same skill (e.g. "success rate on `voice_leading_violations` cluster over the last N attempts"), not whether any single item's score eventually climbed.
4. **Persistent-failure escalation.** If the model keeps failing varied examples from the same cluster after a reasonable number of cycles, that's a signal the reward function or the skill's difficulty calibration may itself be broken for that cluster — route to human/dataset review (same destination as the `ambiguous_prompt` path) rather than continuing to grind retries indefinitely.

This keeps the benefit of hard-example mining (spend more compute where the model is weak) while removing the exploit surface of "score-chase one fixed input until something works."

---

## 3. Effort tiers and token budgets

Keep the tiered structure, fix the cliff penalties that caused false kills and reward-hacking:

| Tier | Max tokens | Recheck signal source |
|---|---|---|
| low | 1,024 | none expected |
| medium | 4,024 | optional, judged not counted |
| high | 11,024 | expected, judged not counted |
| xhigh | 22,024 | expected, judged not counted |
| ultra | 32,024 | expected, judged not counted |
| max | dynamic | expected, judged not counted |

**Overage penalty (replaces the instant `-1.0` hard kill):**
- 0–10% over budget: no penalty (absorbs tokenizer/counting variance).
- 10–50% over: linear penalty from 0 to −0.5.
- 50%+ over: −1.0.

**Complexity cross-matching (easy question, inflated effort):** same proportional shape — penalty scales with how far the attempt overshot the assigned tier, never an instant cliff.

---

## 4. Recheck / self-correction signal

**Problem with the original**: counting occurrences of `wait`, `actually`, `hold on` is directly gameable — a 7B model converges on inserting exactly `n−1` of these words as filler within a few thousand steps, regardless of whether real backtracking happened.

**Fix**: replace counting with a binary judge check, run cheaply (can reuse the policy model itself in judge mode, single extra pass):
> "Did the reasoning trace change direction or explicitly reject an earlier claim between the first half and second half of the trace?"

- Yes + tier calls for it → +0.15
- No signal required at this tier → no check run
- Signal required and absent → small penalty, not a cliff
- Easy-tier problems get zero credit for recheck phrases regardless of surface presence — this rule stays as a hard check since it's a real anti-gaming signal (recheck words on trivial problems are almost always spoofing, not genuine correction).

---

## 5. Context recall, structure, and density

- **Cut**: separate "prompt boundary dissection" bonus and "looping" penalty — they conflicted with each other in the original and produced noisy gradients.
- **Replace with one rule**: one-time +0.15 if the first ~100 tokens restate the problem constraints in the model's own words, checked via a lightweight paraphrase-overlap heuristic (not exact string match).
- **Cut**: the raw-data-dump prose rule as originally written — it directly contradicted the diff-format requirement in Section 6. Scope it: raw-dump penalties apply only to prose/explanation responses (per the directness flag, Section 6), never to code-editing responses where short surgical blocks are the goal.

---

## 6. Output formatting

- Keep structural checks for the "explain" flag: does a numbered/bulleted list structure exist when required — checked structurally, not via loose-character detection (which was fragile in the original).
- **Cut entirely**: the "takeaway / trade-off" keyword bonus. Rewarding specific vocabulary teaches insertion of decorative words rather than real analysis. If deep-analysis quality needs rewarding, it goes to a learned judge comparing against a rubric, not a keyword count.

---

## 7. Code editing: surgical edits vs. overwrites

- Keep unified-diff format requirement (`<<<<<<< ORIGINAL / ======= / >>>>>>> MODIFIED`) for edits to existing files.
- Keep rewarding verbalized surgical intent ("I should edit this instead of rewriting"), but **correlate it with actual diff size** — verbalization alone, without a correspondingly small diff, earns nothing. This closes the obvious gaming path of just saying the phrase.
- **Change the score-cancellation cliff to a proportional penalty**: scaled by the ratio of changed lines outside the task-relevant region to total file lines. A full rewrite of a 300-line file for a one-line fix is heavily penalized; a legitimate larger touch for a legitimate larger task is not zeroed out by a blunt rule.

### 7a. Destructive-edit guard (new)

- **Deletion ratio check**: lines removed outside the task's declared scope ÷ total file lines. Above a threshold (~10–15%) without an explicit rewrite/deletion request → penalty scaled to the excess.
- **Reference-count check**: static check of whether a deleted function/class/export is referenced elsewhere in the repo/context. Disappearance without an accompanying deprecation/replacement in the diff is scored more harshly than an ordinary over-edit — it's a strong signal of accidental data loss rather than a real edit.
- **Scope statement requirement**: the trace must state, in one sentence, which files/functions it intends to change before writing code. Diffs touching files never mentioned in that statement are penalized regardless of size — usually a sign of the model losing track of the task.

---

## 8. Terminal safety override

- Keep the concept of an absolute kill-switch category that bypasses all other scoring.
- **Fix the magnitude**: change from `−5.0` to a clamped `−1.0` to `−2.0`, matched to the rest of the reward scale. A 10x-larger penalty than everything else in the same policy gradient destabilizes learning on adjacent, correctly-scored trajectories in the same batch rather than teaching cleaner avoidance. If stronger avoidance is needed, increase the *frequency* of safety-labeled examples in the queue, not the magnitude of a single penalty.
- **Explicit exclusion from the pipeline entirely** (not just down-weighted — not trained on at all, in either direction): reward shaping toward vulnerability exploitation, weaponizable CBRN-adjacent content, and any capability that only exists to produce an "attack" version of an "understanding" task. A rule-based script cannot reliably separate "explain this vulnerability" from "here is the exploit" because they are frequently the same information — this is a hand-review/exclusion boundary, not a scorable dimension.

---

## 9. UI / front-end "design slop" — replaced almost entirely

The original's golden-ratio, font-mixing, and 60-30-10 color-rule scans were false-positive-prone (professional systems don't literally encode 1.618 in pixel values; color harmony is a judgment call, not a hex pattern). Replaced with mechanically checkable **tell-detection** instead of taste-detection:

- **DOM nesting depth**: penalize excessive wrapper nesting past a fixed threshold — a real, cheap signal.
- **Style variance across differentiated content**: extract style properties (radius, shadow, padding) per component; near-zero variance across structurally different content (a hero card styled identically to a footnote) is a checkable tell.
- **Palette fingerprinting**: blocklist a small set of specific hex clusters disproportionately produced by default AI generation (e.g. the cream/terracotta combo, tinted near-blacks standing in for true black); flag within a small color-distance threshold.
- **Palette-to-brief correlation**: across a training batch, check whether palette changes meaningfully across different briefs. Near-identical palettes regardless of subject matter is a checkable defaulting signal.
- **Gradient/shadow/blur overuse**: simple count per component; low threshold, since intentional single use is rare and usually load-bearing.
- **Template chrome regex set**: tracked-out all-caps eyebrow labels, middle-dot-joined meta strings, em-dash label patterns ("WORD — fragment"), trailing arrows on every button. These are specific, near-universal, regex-detectable AI tells, not taste calls.
- **Numbered-marker misuse**: flag "01/02/03" style markers on content whose surrounding language doesn't imply real sequence.
- **Animation orchestration ratio**: independently-triggered animations ÷ number of components. High ratio (every card fades in separately) flagged; single orchestrated entrance not flagged.
- **Motion trigger source**: penalize load-triggered decorative animation; reward action-triggered motion (bound to a real event listener).
- **`prefers-reduced-motion` compliance**: pure binary, free signal, always scored.
- **Everything else** (actual aesthetic quality, whether a specific color choice suits a specific brief) is explicitly *not* scored by rule — routed to a vision-judge model pass at low sampling frequency (e.g. 1 in 20 UI-generation trajectories) or a periodic human spot check.

---

## 10. Coverage beyond math/code (the parts the original spec missed)

| Domain | What's checkable by rule | What needs a judge/low-frequency pass |
|---|---|---|
| Writing/creative | constraint adherence (length, format, tone), cliché/tell-pattern detection, cross-prompt diversity scoring | actual literary quality |
| Everyday practical / how-to | actionability (concrete ordered steps vs. vague essay), scope-matching against the effort tier, assumption-flagging when ambiguous | — |
| Conversational/emotional support | **not-doing** checks only: no unsolicited diagnostic claims, no validation of self-destructive framing, no mental-state claims about the user | never optimize toward a "warmth" score directly — known sycophancy risk |
| Multi-turn dialogue | contradiction detection across turns via claim extraction, stated-constraint persistence (grep later turns against constraints set early) | — |
| Ambiguity handling | did the model state an assumption and proceed vs. either guess wildly or over-clarify | — |
| Tool use / agentic | did it call a tool when needed vs. hallucinate, did it check a result before trusting it, avoided redundant re-calls | judgment calls on *which* tool was the right choice |
| Multilingual/register | did output match the input's language and formality register | — |
| Mechanistic science (biology/chem/physics at textbook level) | reference-answer matching against a curated fact bank, citation-grounding against a fixed reference corpus (not live web), internal-consistency checks across the trace | deeper explanatory quality |

**Excluded from RL optimization entirely, by design**: patient-specific medical guidance (diagnosis, dosing, treatment recommendations) — no safe automated ground truth exists, and RL tends to reward fluent/confident-sounding answers, which is the opposite of safe behavior in this domain.

---

## 11. Optional extension: spatial/3D understanding

Only worth pursuing if a real use case (e.g. Blender scene generation) justifies the data investment; not a core requirement for the base model.

- **Don't build a new spatial architecture.** Attach a pretrained frozen vision encoder (CLIP/SigLIP-class) via a lightweight projection layer (a few hundred million parameters) that maps image features into the 7B's embedding space — this is the standard LLaVA-style approach, not a research problem.
- **Route spatial reasoning through a structured intermediate representation**, not direct coordinate generation:
  1. Model outputs a scene graph (objects, approximate dimensions, relationships, materials) as structured JSON.
  2. A deterministic converter turns the scene graph into Blender API calls.
  3. A rendered view is fed back to the model as a visual confirmation step.
  This keeps the model responsible for *intent*, and deterministic code responsible for *execution* — it never has to hallucinate exact geometry.
- **Rewardable dimensions**: does generated Blender Python execute without error (binary), does the object hierarchy in the output match the scene graph (checkable), do approximate dimensions match stated constraints within tolerance (checkable).
- **Not rewardable by rule**: whether the result looks good, whether proportions or materials are aesthetically convincing — same low-frequency judge/human pattern as Section 9.
- **Data is the real bottleneck**, not the architecture. Generic image-text pairs give general visual understanding but weak spatial reasoning specifically. Useful training data here is closer to (rendered scene, scene-graph/Python that generated it) pairs, which can be synthesized programmatically since Blender is fully scriptable — render many scenes, extract the generating description from scene metadata automatically rather than hand-labeling.

---

## 11a. Music (MIDI / piano note generation)

Scoped narrowly: symbolic note generation (piano-first, extensible to other monophonic/polyphonic melodic instruments), not audio synthesis or real-time performance. This is a good fit for the existing architecture because — unlike visual/aesthetic design — music has a large zone of genuinely checkable structure underneath the subjective layer.

### Why RL alone can't teach this
RL refines a policy that already has baseline competence — it doesn't teach vocabulary from nothing. A model with no prior exposure to note/harmony/rhythm structure has no reasonable trajectories to sample; exploration from a near-random symbolic-music policy is enormous and mostly garbage. Knowledge injection has to come first, same relationship as code/math already have to this pipeline's RL phase.

### Token representation
MIDI must be converted to a text-like token stream before any training. Options, in order of relevance to a piano-first target:
- **REMI-style event tokens** (note-on, duration, velocity, position-in-bar, tempo) — handles polyphony (two-hand piano) well, the standard choice for transformer-based symbolic generation.
- **Flat MIDI-like event stream** (note-on/off, time-shift, velocity) — simpler to implement, weaker at explicit bar/beat structure.
- **ABC notation** — human-readable, token-efficient, but weaker for multi-voice piano; better suited to single-line melody only.
- **Plain JSON per note** (`{"pitch", "start", "duration", "velocity"}`) — easiest to implement and easiest to validate with rule-checking code later, but token-expensive (JSON punctuation overhead burns context fast and adds syntax-learning noise on top of music-learning noise). Reasonable starting point to validate the pipeline end-to-end; can be swapped for a denser event scheme once validated.

### Pipeline sequence
1. **MIDI corpus → token/JSON conversion.** Public sources exist (e.g. Lakh MIDI Dataset); filter to piano-only/piano-heavy subsets if that's the actual target rather than full-band arrangements.
2. **Continued pretraining on the token stream** — raw next-token prediction, no instruction framing yet. Must be **mixed-replay with the original raw pretraining-style data** (not instruction-formatted data at this stage) to avoid catastrophic forgetting of general capability. Cheapest and lowest-risk as a **LoRA adapter** on a frozen base rather than full-parameter fine-tuning, since this stage is pure knowledge injection and doesn't need to touch the base weights at all.
3. **Adapter merge decision**: for a narrow scope like piano/melody-only, keeping the LoRA as a permanently separate, domain-routed adapter (active only on `domain: "midi_piano"` rows) is a reasonable choice — this isn't trying to deeply blend music into general reasoning, just add a bounded capability. Merging into the base is the alternative if smoother blending into general conversation is needed later.
4. **SFT for instruction-following behavior**, separately mixed-replay with the original **instruction-formatted** data (not raw-text replay — replay the right kind of data at the right phase, don't collapse both replay pools into one). Include a small slice of music-flavored instruction examples ("Human: write a 16-bar piano melody in D major / Assistant: [correctly tokenized output]") so the model learns to respond conversationally rather than just autocomplete raw tokens. This slice can be small relative to the main instruction dataset but shouldn't be zero.
5. **Fold into the continual RL stream** as `domain: "midi_piano"` rows, scored per Section 17 below.

### Hard-rule checks (parser-based, same shape as the code diff-parser)
- **Playability**: max simultaneous notes per hand, interval-span checks against realistic hand-span at the given tempo, no physically impossible simultaneous note assignments.
- **Duration math**: note durations within each measure sum correctly against the stated time signature — pure binary check.
- **Key/scale consistency**: flag notes outside the stated key with no theoretical justification (no borrowed chord/modal-interchange rationale in context).
- **Basic voice-leading** (for multi-voice piano): no parallel fifths/octaves between voices, checkable directly from simultaneous note pairs across consecutive positions.
- **Form adherence** (if a form is specified in the prompt, e.g. 12-bar blues, ternary form): checkable against a structural template.

### Taste layer (not rule-checkable)
Same tell-detection pattern as UI/design slop: fingerprint clichéd defaults (repetitive unvaried 4-bar phrasing, over-reliance on the same stock chord progression regardless of prompt, near-identical rhythmic patterns across supposedly different pieces in a training batch) rather than trying to score "is this beautiful" directly. Real aesthetic quality — melodic elegance, emotional arc — is routed to a low-frequency judge pass (human listener or an audio-rendering judge model), sampled at roughly the same rate as the UI-quality judge pass, not scored per-trajectory by rule.

### Explicitly out of scope
Audio synthesis, real-time performance/expressive timing control, and any audio-modality input — same reasoning as the computer-use exclusion: wrong modality and feedback loop for a 7B text model at this stage. This stays a fully symbolic, text-adjacent capability.

---

## 15. Reward Hacking Taxonomy — Full Risk Map and Mitigations

Every reward function creates pressure toward whatever satisfies the function, not necessarily toward the intended behavior. This section catalogs every hacking vector relevant to this pipeline — the ones already fixed in earlier sections, plus architecture-level and cross-cutting ones (like MoE router collapse) that need their own treatment. Organized by where in the system the exploit lives.

### 15.1 Architecture-level: MoE router collapse (if the base model is MoE)

If the 7B is actually a sparse Mixture-of-Experts model rather than dense, RL introduces a failure mode dense models don't have.

**The exploit**: the router (the small network that decides which experts handle each token) can collapse toward routing almost everything to a small subset of experts, because early in RL certain experts get slightly better initial reward on certain reward-relevant token patterns, get selected more, get more gradient signal, get better, get selected even more — a rich-get-richer loop. The remaining experts starve, stop receiving useful gradients, and effectively become dead weight. Net effect: your 7B-labeled MoE model is quietly running as an effectively much smaller dense model, capacity collapses, and the RL reward can still look like it's climbing because the surviving experts overfit to the reward's specific patterns while destroying the general capacity for diversity the MoE architecture was for in the first place.

**Why RL makes this worse than pretraining does**: pretraining has abundant, diverse gradient signal that discourages collapse naturally (broad next-token prediction rewards no single expert being right for everything). RL has sparse, narrow, high-variance reward signal concentrated on specific behaviors (e.g. "did this diff pass the destructive-edit check") — exactly the kind of concentrated signal that accelerates router collapse if left unchecked.

**Mitigations**:
- **Load-balancing auxiliary loss, kept active through the RL phase, not just pretraining.** Most MoE training already uses an auxiliary loss term that penalizes uneven expert utilization during pretraining — this must stay switched on and weighted meaningfully during RL, not silently dropped once you move from pretraining to RL as if it were only a pretraining concern.
- **Router entropy monitoring as a first-class training metric.** Track the entropy of routing decisions per batch, per domain tag. A sharp entropy drop concentrated around specific domains (e.g. routing entropy collapsing specifically on `domain: code` rows once code RL kicks in) is the earliest detectable signal of collapse, well before downstream quality visibly drops.
- **Per-expert utilization floor with a hard alarm.** If any expert's selection frequency drops below a floor (e.g. well under its fair share for a sustained window) trigger an automatic training pause/alert rather than letting it silently continue — this is the same "stop and check" instinct as your regression-eval gate for catastrophic forgetting, applied to routing instead of raw output quality.
- **Capacity-factor / expert-dropping safeguards**: cap how many tokens any single expert can be routed per batch (standard MoE capacity-factor mechanism) so the router physically cannot overload one expert even if it wants to, tokens overflow to the next-best expert instead of stacking further onto an already-dominant one.
- **Domain-diverse batch composition during RL**, same mixed-replay principle used everywhere else in this spec — if RL batches are dominated by one domain for long stretches, router collapse toward that domain's preferred experts is far more likely than if batches stay mixed across code/math/writing/music/etc.

### 15.2 Reward-signal exploitation — surface-form mimicry (already addressed, summarized here for completeness)

The single most common hacking pattern across this whole spec: the model learns to produce the *textual signature* of good behavior without the underlying substance. Already fixed instance-by-instance in earlier sections; listed together here because it's really one recurring failure mode wearing different costumes:
- Recheck-phrase counting → fixed via judge-based direction-change check (Section 4).
- Takeaway/trade-off keyword bonus → cut entirely (Section 6).
- Surgical-intent phrase without matching diff size → fixed via correlation check (Section 7).
- Design "tells" → fixed via pattern-detection instead of taste-scoring (Section 9).
- Music phrase/chord repetition → fixed via tell-detection instead of taste-scoring (Section 11a).

**General principle to apply to any new domain added later**: if a reward component can be satisfied by inserting specific words/phrases/tokens without changing the substantive content around them, it will be gamed. Test every new reward rule by asking "could a model get credit for this by memorizing a fixed insertion, regardless of context?" If yes, it needs either a judge-based check or a structural/mechanical check tied to actual content, not surface pattern matching.

### 15.3 Length and verbosity hacking

- **Padding to hit budget without adding content** — the token-budget rules (Section 3) prevent *overage*, but don't by themselves prevent a model from padding a short answer up toward the *ceiling* of its assigned tier to appear more thorough. Needs an added check: information-density estimate (e.g. compression ratio of the response, or a redundancy/repetition detector) so filling budget with restated content is penalized, not just budget overage.
- **Splitting one long response into many short exchanges to farm the "assumption-flagging" or "scope-statement" bonuses repeatedly** — if any structural bonus (problem restatement, scope statement, etc.) is awarded per-response rather than per-conversation, a multi-turn conversation could be artificially fragmented to re-trigger it multiple times. Fix: cap these structural bonuses at once per conversation/task, not once per turn.

### 15.4 Multi-metric / composite-reward exploitation

- **Reward component cross-subsidization.** When you sum many reward terms (effort-tier compliance + recheck signal + structure bonus + safety pass + domain-specific hard rules), a model can learn to sacrifice one weak-consequence component to over-optimize a strong-consequence one, ending up net-positive while being clearly bad on some axis a human would care about. Mitigation: **cap each component's contribution** (no single positive term can offset more than a fixed fraction of a hard-rule violation elsewhere) rather than a flat linear sum, and periodically audit *which* component is driving score changes, not just the aggregate.
- **Reward-model / judge-model exploitation.** Anywhere this pipeline uses a judge pass (recheck detection, UI-quality sampling, music-quality sampling, science citation-grounding) instead of a hard rule, the judge itself becomes a target for adversarial optimization — the policy can learn inputs that fool the judge specifically, rather than genuinely satisfying the underlying quality bar (this is the well-documented "reward model hacking" failure mode from RLHF literature generally). Mitigations: rotate/ensemble judge prompts or judge checkpoints rather than one fixed judge the policy can overfit to; periodically refresh the judge on newly-collected human-labeled examples so it doesn't stay static long enough to be reverse-engineered; keep judge-pass sampling rate low and unpredictable (not a fixed schedule the model could in principle infer and specifically target).

### 15.5 Requeue-loop exploitation

- **Deliberate failure to trigger favorable requeue conditions.** If the failure-taxonomy requeue logic (Section 2) ever correlates with any exploitable downstream effect — e.g. if `ambiguous_prompt`-tagged rows get *easier* scoring on their next attempt, or skip certain checks — the policy could learn to intentionally produce trajectories that get tagged that way to reach an easier-scored retry. Mitigation: requeued rows must be scored by the exact same reward function as first-attempt rows, no leniency delta; the taxonomy tag controls *whether* to requeue and *where* to route for review, never *how strictly* the retry gets scored.
- **Unbounded identical-item retry as a search process.** Requeuing the exact same failed question repeatedly until it scores well is itself a hacking vector — enough attempts on one fixed input lets the policy stumble onto a reward-checker blind spot or memorize a specific input/output pair rather than generalize the fix, and the resulting score climb doesn't indicate real capability improvement. Mitigation: see Section 2a — attempt cap per literal question, then substitute a varied same-cluster question rather than repeating the original, and track success at the cluster level rather than the item level.

### 15.6 Safety-boundary probing as a reward-adjacent exploit

- Not classic reward hacking but adjacent and worth including: because the Section 8 kill-switch is a large fixed penalty for a defined category set, there's pressure (especially under aggressive exploration) for the policy to learn the *boundary* of that category set precisely and produce content that sits just outside it while still being effectively harmful — "malicious-adjacent but technically unflagged" content. Mitigation: the category classifier in Section 8 should be trained/maintained with adversarial examples specifically probing the boundary, refreshed periodically rather than treated as a fixed static list, and boundary-adjacent outputs should be routed to human review rather than scored as simply "safe" by default whenever the hard classifier doesn't fire.

### 15.7 Cross-domain contamination as an indirect hack

- **Domain-tag spoofing pressure.** Since different `domain` tags get different reward rules (e.g. `midi_piano` rows get playability checks instead of code-diff checks), if domain tagging is ever inferred by the model rather than fixed by the dataset row, there's a theoretical incentive to produce output that reads as belonging to whichever domain has the most exploitable/lenient reward shape for that content. Mitigation: domain tag must always come from the dataset row metadata (Section 12), never inferred from model output, and the scorer should verify the actual output structurally matches the claimed domain (e.g. a `midi_piano` row whose output contains no valid parseable note data at all should fail structurally before any music-specific scoring runs) rather than trusting the tag blindly.

### 15.8 MoE-specific reward-domain interaction risk

- Beyond raw router collapse (15.1), MoE introduces a subtler risk once multiple domains (code, math, music, design) share one router: **domain-specific reward pressure can inadvertently reroute experts that were previously shared across domains**, degrading a domain that isn't currently being trained on RL for that batch. This is effectively catastrophic forgetting expressed through routing rather than through weight overwrite, so it evades your normal regression-eval-on-weights framing unless you specifically eval routing behavior too. Mitigation: run the regression-eval suite (already required for catastrophic-forgetting checks) with router-utilization logging enabled, not just output-quality logging, so a routing-level regression is visible even if raw output scores haven't dropped yet.

### 15.9 Summary table

| Vector | Layer | Fix pattern |
|---|---|---|
| Router collapse | Architecture (MoE) | Load-balancing loss kept live in RL, entropy monitoring, utilization floor + alarm, capacity-factor cap, domain-diverse batches |
| Surface-form mimicry | Reward design | Judge-based semantic check or structural check, never phrase-count alone |
| Budget-padding | Reward design | Information-density/redundancy check on top of budget-overage check |
| Bonus-farming via fragmentation | Reward design | Cap structural bonuses per-conversation, not per-turn |
| Composite-reward cross-subsidization | Reward design | Cap component contribution, audit per-component score drivers |
| Judge-model exploitation | Reward design | Rotate/ensemble judges, refresh on new labels, low/unpredictable sampling rate |
| Requeue-loop gaming | Pipeline logic | Identical scoring function regardless of taxonomy tag |
| Safety-boundary probing | Safety layer | Adversarially-refreshed classifier, human review for boundary-adjacent cases |
| Domain-tag spoofing | Pipeline logic | Tag from dataset metadata only, structural domain-match verification before domain-specific scoring |
| Cross-domain expert rerouting | Architecture (MoE) | Router-utilization logging included in regression-eval, not just output-quality |

---

## 19. External Verifier Model (9B, advisory judge — e.g. Qwen3.5-9B class)

Addresses the self-judging weakness in Section 15.4: several sub-checks in this spec (recheck genuineness, ambiguity-handling quality, citation plausibility, low-frequency taste sampling) currently rely on the 7B judging itself in a second pass, which shares the same blind spots it's being trained around. A separate, independently-trained model closes that gap — but only as a bounded, advisory subroutine the primary scorer calls, never as a parallel authority.

### 19.1 Control hierarchy

```
Trajectory generated by 7B policy
        │
        ▼
PRIMARY SCORER (this system, always runs, always authoritative)
  - hard rules (Sections 3-11): token budget, diff parser, playability
    parser, structural checks, safety classifier, domain-match verifier
        │
        │ only for sub-checks flagged "judge-required" for this row's
        │ domain, sampled at the existing low/unpredictable rate (15.4)
        ▼
9B VERIFIER (advisory only)
  - receives the dataset row + trajectory + a narrow structured
    question set
  - returns typed answers only, never a free-form score
        │
        ▼
Primary scorer maps typed answers → pre-defined bounded deltas,
clamps the total adjustment, computes final reward, logs disagreement
```

**Non-negotiable properties of this hierarchy:**
- The 9B never issues a terminal score. It answers specific typed questions; the primary scorer owns the score-mapping table.
- Hard-rule and safety-classifier outcomes are never subordinate to the verifier — a deterministic kill-switch or destructive-edit flag wins outright regardless of what the verifier returns.
- The verifier's influence on final reward is clamped (per Section 15.4's component-capping rule) so no single verifier call can singlehandedly flip a bad trajectory into a high score.
- All training-loop writes (score, requeue decision, replay-buffer inclusion) are made by the primary scorer. The verifier has no direct write path into the training loop.

### 19.2 Verifier request contract

The primary scorer constructs the verifier call from three inputs, matching the dataset row schema in Section 16:

1. **The dataset row's problem, constraints, and domain tag** (`user_problem`, `music_constraints`/`task_scope`/`reference_answer` etc. as applicable to the domain, `effort_tier`, `difficulty`).
2. **The full 7B trajectory** (reasoning trace + final answer).
3. **A fixed, narrow rubric question set for that row's domain** — not an open "evaluate this" prompt. Each question is typed, so the answer is checkable and mappable to a score delta without further judgment on the primary scorer's side.

**Example rubric question format (per-domain, defined ahead of time, not improvised per call):**

```json
{
  "row_id": "...",
  "domain": "code",
  "questions": [
    {
      "id": "looping",
      "prompt": "Does the reasoning trace copy-paste or re-state the original prompt's parameters more than twice without new analysis?",
      "answer_type": "boolean",
      "expected_fields": {"answer": "yes|no", "count": "int", "citation": "string (quote or line ref)"}
    },
    {
      "id": "recheck_genuine",
      "prompt": "Does the trace change direction or explicitly reject an earlier claim between its first half and second half?",
      "answer_type": "boolean",
      "expected_fields": {"answer": "yes|no", "citation": "string"}
    },
    {
      "id": "scope_violation",
      "prompt": "Does the diff touch any file or function not named in the trace's own scope statement?",
      "answer_type": "boolean",
      "expected_fields": {"answer": "yes|no", "offending_files": "list[string]"}
    },
    {
      "id": "assumption_stated",
      "prompt": "If the problem was ambiguous, did the trace explicitly state its assumption before proceeding, rather than guessing silently or over-asking?",
      "answer_type": "categorical",
      "expected_fields": {"answer": "stated|guessed_silently|over_clarified|not_applicable"}
    }
  ]
}
```

Domain-specific question sets follow the same shape:
- `midi_piano`: `"is the phrase repeated more than N times without melodic variation?"` → boolean + count.
- `writing`: `"does the response open by restating the prompt verbatim?"` → boolean + citation.
- `science`: `"is this specific claim present in the reference corpus provided, or unsupported?"` → categorical (`supported | unsupported | not_checkable`).
- `ui_design`: `"does this component's style properties match another component of meaningfully different content type?"` → boolean + which components.

**Answer requirements, enforced by the primary scorer before any delta is applied:**
- Every answer must use the declared `answer_type` (boolean/categorical/int) — free-text answers that don't parse into the expected schema are discarded and logged as a verifier-format failure, not silently accepted.
- Every boolean/categorical answer above a small threshold of consequence must include a citation (quote, line reference, or component ID) — an unsupported "yes" with no pointer into the trajectory is treated as a non-answer and does not move the score. This closes the obvious failure mode of the verifier rubber-stamping answers without checking.

### 19.3 Score-mapping table (owned by the primary scorer, not the verifier)

Each question ID maps to a pre-defined bounded delta, e.g.:

| Question ID | Answer | Delta |
|---|---|---|
| `looping` | yes, count > 2 | −0.4 (capped by effort tier, per Section 5) |
| `recheck_genuine` | yes (and tier requires it) | +0.15 |
| `recheck_genuine` | no (and tier requires it) | −0.1 |
| `scope_violation` | yes | proportional penalty, per Section 7a deletion-ratio logic |
| `assumption_stated` | guessed_silently | −0.1 |
| `assumption_stated` | over_clarified | −0.05 |
| `assumption_stated` | stated | +0.1 |

This table lives in the primary scorer's code and is versioned/reviewed the same way the hard rules are — the verifier only ever selects among pre-agreed answer categories, it does not invent a score.

### 19.4 Anti-gaming measures specific to the verifier

- **Rotate rubric phrasing periodically** (same question intent, different wording) so the 7B policy can't converge on a fixed textual pattern that happens to satisfy one static prompt (direct extension of the surface-mimicry principle in 15.2).
- **Sample verifier calls at the existing unpredictable low frequency** (15.4) rather than a fixed schedule — a predictable trigger point is itself something a policy can learn to specifically target.
- **Log every disagreement between the verifier's answer and what the deterministic rules independently found**, even when the verifier isn't the deciding factor. Persistent disagreement clustered in one question category is the signal that either the deterministic rule or the verifier prompt needs revision — treat it as a standing diagnostic, not a one-off.
- **Refresh the verifier model/prompt set periodically against new human-labeled examples**, same rotation principle as any judge model in 15.4, so a static verifier doesn't stay predictable long enough to be reverse-engineered by the policy over a multi-month continual run.
- **The verifier must never receive the primary scorer's internal weights or delta table** — it only ever receives the row, the trajectory, and the question set. Keeping the mapping logic outside the verifier's own context prevents the 7B (if it ever gains any visibility into verifier prompts through training data leakage) from learning to reverse-engineer the scoring table itself.

---

## 20. Dataset structure (per row)

Each streamed training row should carry, at minimum:

```
{
  "problem_id": "...",
  "domain": "code | math | science | writing | practical | conversation | tool_use | multi_turn | midi_piano",
  "effort_tier": "low | medium | high | xhigh | ultra | max",
  "difficulty": "easy | medium | hard",
  "explain_flag": true/false,          // directness vs. structured explanation
  "recheck_required": true/false,       // derived from effort_tier, not hand-labeled per row
  "task_scope": ["file_or_function_names_expected_to_change"],  // for code rows
  "reference_answer": "...",            // for math/science rows with ground truth
  "reference_corpus_id": "...",         // for citation-grounding checks
  "prior_failure_reason": "timeout | format_violation | wrong_answer | ambiguous_prompt | null",
  "music_constraints": {"key": "...", "time_signature": "...", "form": "...", "bar_length": 0},  // for midi_piano rows
  "user_problem": "...",
  "conversation_history": [...]         // for multi-turn rows
}
```

No assistant answers are pre-supplied — the model generates the trajectory, the scorer evaluates it against the row's metadata at score time.

---

## 21. What the scoring code actually needs to implement

1. **Token counter with soft-overage scaling** (Section 3) — replaces hard cliffs.
2. **Judge-mode inference pass** for recheck/self-correction detection (Section 4) — same base model, single extra cheap call per trajectory that needs it, not run on every trajectory.
3. **Paraphrase-overlap heuristic** for the problem-restatement bonus (Section 5) — lightweight embedding similarity between first-100-tokens and prompt, not exact match.
4. **Structural list-detection** for explain-flag responses (Section 6) — parse for numbered/bulleted markdown structure, not loose-character regex.
5. **Diff parser + line-ratio calculator** for code edits (Sections 7, 7a) — parses the `<<<<<<< / ======= / >>>>>>>` blocks, computes changed-line ratio against total file, cross-checks against the declared scope statement, runs a static reference-count check for deleted symbols.
6. **Failure-taxonomy tagger** on the requeue path (Section 2) — classifies why a trajectory scored below zero before deciding whether to requeue or route to review.
7. **Style-property extractor** for UI-slop tell detection (Section 9) — parses generated CSS/DOM for nesting depth, style variance across components, gradient/shadow counts, palette hex-distance against a blocklist, animation trigger-binding inspection.
8. **Claim-extraction and cross-turn diff checker** for multi-turn consistency (Section 10) — extracts factual/preference claims per turn, checks later turns for contradiction or dropped constraints.
9. **Reference-corpus lookup** for science citation-grounding (Section 10) — retrieval check against a fixed, curated corpus, not live web search, to keep grounding checks deterministic and reproducible.
10. **Low-frequency judge-model sampling scheduler** — a separate lightweight process that pulls ~1-in-N trajectories (UI quality, creative quality, tool-choice judgment) for a heavier judge pass or human review queue, decoupled from the main per-trajectory scorer so it doesn't bottleneck the streaming loop.
11. **MIDI/music-token parser** (Section 11a) — parses generated token/JSON output back into structured note data, checks playability (hand-span, simultaneous-note limits), duration math against time signature, key/scale consistency, voice-leading, and form adherence; also computes phrase-repetition and chord-progression-reuse metrics for tell-detection.
12. **Safety-category classifier + clamped penalty path** (Section 8) — routes flagged trajectories to a fixed penalty range, and hard-excludes certain categories (exploit generation, weaponization-adjacent content) from the training signal entirely rather than scoring them at all.

---

12. **MoE router-utilization logger** (Section 15.1, 15.8) — per-batch, per-domain routing entropy and per-expert selection frequency, feeding both a live alarm (utilization floor breach) and the regression-eval suite.
13. **Reward-component contribution auditor** (Section 15.4) — logs each reward term's contribution to final score per trajectory so composite-score gaming is diagnosable, not just visible after the fact as a quality drop.
14. **Domain-match structural verifier** (Section 15.7) — confirms output structurally matches its claimed `domain` tag before domain-specific scoring runs (e.g. a `midi_piano` row must contain parseable note data or it fails before music rules apply).

15. **Similar-question generator for bounded retry** (Section 2a) — once a literal question's retry cap is hit, generates a surface-varied item from the same skill/difficulty/domain cluster (changed values/names/phrasing, same underlying skill) to requeue instead of the original; also maintains cluster-level pass-rate tracking and triggers human-review escalation for persistently-failing clusters.

---

16. **9B verifier client + score-mapping table** (Section 19) — constructs the structured, typed rubric request from the dataset row and trajectory, validates returned answers against the declared `answer_type` and citation requirement before any delta is applied, applies the pre-defined bounded delta per question, enforces the overall clamp, and logs every case where the verifier's answer disagrees with an independently-computed deterministic check.

---

## 22. Summary of what changed from the original spec

- Every hard-cliff penalty (`-1.0`, `-5.0`, score cancellation) became a proportional penalty scaled to the severity of the violation.
- Every keyword/phrase-counting reward (recheck words, takeaway/trade-off terms) became either a judge-based semantic check or was cut outright.
- Contradictory rules between sections (raw-dump penalty vs. diff-block requirement) were scoped so they no longer fire on the same response type.
- Design-quality scoring shifted from unfalsifiable aesthetic rules (golden ratio, color harmony) to falsifiable AI-tell detection, with real aesthetic judgment deferred to low-frequency judge/human review.
- Coverage was extended past math/code/science into writing, practical tasks, conversation, multi-turn dialogue, tool use, and ambiguity handling — with explicit guardrails against optimizing conversational warmth or medical confidence via RL.
- Vulnerability research and patient-specific medical guidance are structurally excluded from the reward pipeline rather than scored, since no rule-based (or realistically, judge-based) ground truth exists that separates the safe framing from the harmful capability.
- Spatial/3D understanding was scoped as an optional vision-encoder attachment with a structured intermediate representation, not a new architecture, and flagged as data-bottlenecked rather than architecture-bottlenecked.
