# Parameter Golf Research Log

Last updated: 2026-03-27

## Purpose

This file is the durable memory for the ongoing `parameter-golf` exploration.

It records:

- what was changed
- what hypotheses were tested
- what Aaron contributed conceptually
- which runs mattered
- which branches are dead
- what the current best live ideas are

The aim is to keep the search legible and cumulative, rather than letting it dissolve into chat history and one-off runs.

Presentation note:

- some historical links below still point back to the original local workspace
- the headline archived run logs for this standalone repo live under `results/`

## Current Status Snapshot

As of `2026-03-23`, the work has split into two distinct tracks:

- model-search track: shared-core recurrence, mirror scheduling, resonance probes, and directional `C` correction
- trainer-correctness track: making the local MLX and torch scripts tell the truth about evaluation, wallclock, and final verification

The current best capped verified local result is now an offline derived whole-tensor carrier:

- `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- exact `val_bpb = 2.35570158`
- compressed artifact size: `15,109,864` bytes via `lzma`
- carrier: rank-64 residual sidecar on `blocks.0.mlp.proj.weight`

The best over-budget local result remains:

- `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2`
- exact `val_bpb = 2.35551193`
- compressed artifact size: `16,263,292` bytes via `lzma`

That displaced the earlier local leaders:

- `mlx_full_seq_mlp3x_200_realval_vb524k` at `2.37334218`
- `mlx_full_mirror_mlp3x_dirc02_200_realval_vb524k` at `2.38131855`
- `mlx_full_mirror_dirc02_200_realval` at `2.38989686`

The current evidence says straightforward sequential capacity is still the right base family locally, but the latest capped gain now comes from offline artifact design rather than another architectural retrain.

The trainer files now default to `lzma` for the compressed int8 artifact while still being able to read older `zlib` artifacts.

The torch work on `2026-03-21` was not new model search. It was semantic cleanup and local verification:

- decouple validation batching from grad accumulation
- make wallclock accounting actually mean wallclock
- make final quantized roundtrip verification optional for local smoke
- make a single-process local MPS smoke path work on the Mac

This did not improve score directly. It improved trust in the training/evaluation loop.

### What The M4 Verification Actually Proved

The successful local torch smoke run is:

- [torch_mps_semanticfix_smoke_smallval_b8.txt](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf/logs/torch_mps_semanticfix_smoke_smallval_b8.txt)

That run proved:

- `VAL_MAX_BATCH_TOKENS` is live and logged
- `wallclock_time` is now logged separately from `train_time`
- `VERIFY_QUANTIZED_ROUNDTRIP=0` actually skips the final expensive roundtrip eval
- the torch script can now be smoke-tested locally on Apple Silicon through MPS

The run also surfaced two real bugs that were fixed:

- `shared_core_revisit_damping` was incorrectly referenced as a nonexistent local variable instead of `args.shared_core_revisit_damping`
- RoPE cache tensors created during `torch.inference_mode()` were being reused in training, causing autograd to fail

Those were not speculative issues. They were directly observed and fixed.

### What Has Not Been Proven Yet

- no CUDA verification has been run on the current `train_gpt.py` from this machine
- no official-challenge-equivalent `8xH100` run has been done
- no friend-box or remote-box run has yet validated the torch path after the semantic fixes

So the current state is:

- local MLX architecture search is real
- local torch trainer semantics are cleaner and locally smoke-verified on MPS
- CUDA promotion is still pending external hardware

## Repos And Working Context

- Main repo: `/Users/aaronday/dev/discrete-mathematics-for-formal-system-design`
- Challenge repo: `/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf`
- Current challenge branch: `codex/shared-core-recurrence-v1`

Supporting docs already created in the main repo:

- [parameter-golf-challenge-parse.md](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-challenge-parse.md)
- [parameter-golf-thinking-sheet.md](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-thinking-sheet.md)
- [parameter-golf-thinking-sheet.pdf](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-thinking-sheet.pdf)
- [parameter-golf-ooda-loop.md](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-ooda-loop.md)

Primary code paths under active modification:

- [train_gpt.py](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf/train_gpt.py)
- [train_gpt_mlx.py](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf/train_gpt_mlx.py)

Current caution:

- the upstream soft readability limit in the repo header is now exceeded
- `train_gpt.py` is over `1500` lines
- `train_gpt_mlx.py` is over `1500` lines

This is acceptable for exploration, but should be corrected before any cleanup or upstreaming attempt.

## What Aaron Has Contributed

The highest-value human contributions so far have not been hyperparameter tweaks. They have been structural reframings.

### 1. Reframing The Challenge

Aaron correctly reframed the problem as a constrained dynamical system rather than a generic “small LLM” exercise.

Core useful reframings:

- repeated operators matter more than raw parameter count
- compression survival matters more than elegant float-space behavior
- revisit order and revisit role are part of the architecture
- resonance, damping, and attractors are relevant if cashed out into measurable behavior

### 2. OODA / Signal Framing

Aaron introduced an OODA-style search framing:

- observe exact post-quantized score and internal dynamics
- orient around amplification, damping, mean reversion, and least-bad moves
- decide on one small mechanistic change
- act with a reversible run

This was the right move. It prevented the search from degenerating into random knob sweeps.

### 3. A/B/C Role Hypothesis

Aaron proposed:

- `A` as refine
- `B` as amplify
- `C` as correct
- stabilization should happen later, at tuned intervals
- the important surviving structure under quantization is likely a stable attractor

This directly led to the role-gain, correction, stabilization, and later adaptive/directional `C` correction experiments.

### 4. Lipschitz / Bounded C-Correction

Aaron then proposed that `C` correction should be bounded in a Lipschitz-like sense, and that its strength should be derived from the local “success” of `A` and `B`.

That led to:

- scalar adaptive contractive `C` correction
- probes for `lambda_C`, `cosAB`, and `amp_ratio`
- the discovery that the scalar version was coherent but had the wrong geometry

### 5. Direction Over Blanket Contraction

The next useful human push was toward directional correction instead of shrinking the whole state.

That produced the first correction family that stayed close to the mirror baseline without collapsing.

## Detailed Process History

This section records the process as a sequence of moves, not just a pile of run IDs.

### Step 1. Aaron decided not to play the obvious game

The first important move was refusing to treat `parameter-golf` as a generic “tiny LLM competition.”

Aaron pushed the framing toward:

- constrained dynamics
- repeated operators
- compression survival
- recurrence and revisit structure

That changed the search strategy immediately. The goal stopped being “tune more knobs” and became “find a compact repeated structure that earns its bytes.”

### Step 2. The repo was set up for a local exploration loop

The challenge repo was cloned into the main workspace and a local MLX path was made usable.

This included:

- local setup scripts
- a smoke-data path
- a one-command MLX smoke runner
- artifact/log capture in the challenge repo

That was essential because it made the loop fast enough for structural search on a Mac before touching CUDA infrastructure.

### Step 3. Aaron asked for the highest-EV way to engage

Instead of asking “how do I win,” Aaron asked a better question: what is the highest expected-value interaction with this environment, given what he knows and does not know.

That produced the working strategy:

- do not chase the leaderboard first
- build a clean experimental loop
- make one structural bet
- use local runs to discover mechanism before paying for GPU scale

### Step 4. The challenge was parsed into first principles

The repo gained durable framing docs:

- [parameter-golf-challenge-parse.md](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-challenge-parse.md)
- [parameter-golf-thinking-sheet.md](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-thinking-sheet.md)
- [parameter-golf-thinking-sheet.pdf](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-thinking-sheet.pdf)
- [parameter-golf-ooda-loop.md](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/examples/parameter-golf-ooda-loop.md)

This mattered because it moved the effort from chat drift to explicit hypotheses.

### Step 5. Aaron chose the first real structural bet: shared-core recurrence

The original upstream model uses `9` effective layers.

Aaron’s first meaningful architectural direction was:

- reuse a smaller number of blocks
- make schedule shape explicit
- treat the model as a trajectory through a few reusable operators

That led to:

- `SHARED_CORE_BLOCKS`
- `SHARED_CORE_SCHEDULE`
- a mirror-vs-cyclic experiment family

### Step 6. The first real result appeared: mirror beat cyclic

This was the first moment the search moved out of speculation.

On the smoke loop:

- `mlx_shared3_cyclic_cmp` scored `2.66407351`
- `mlx_shared3_mirror_cmp` scored `2.65437865`

The parameter count was the same. The artifact size was almost the same. The schedule alone mattered.

This established that reuse order is a real architectural variable in this challenge.

### Step 7. Aaron pushed on resonance rather than cosmetically on score

Aaron asked whether resonance-like meta-structure was observable and testable.

That caused a change in method:

- stop relying only on `val_bpb`
- inspect passwise norms and update magnitudes
- compare schedules as trajectories, not just outputs

This led to the resonance analysis tooling and the first practical threshold:

- pass-RMS separation over about `5%`
- delta-norm separation over about `10%`
- cosine not trusted as a steering signal on its own

### Step 8. Aaron forced the search into an OODA loop

Aaron explicitly reframed the workflow as:

- observe
- orient
- decide
- act

and asked for “least bad move” behavior rather than a proliferation of knobs.

That was the point where the project became a legible experiment program instead of a chat-driven branch explosion.

### Step 9. Aaron introduced the A/B/C role hypothesis

The next shift came from Aaron assigning possible jobs to the three shared blocks:

- `A = refine`
- `B = amplify`
- `C = correct`

and proposing that:

- stabilization should happen late
- the interval should be tunable
- the most important information to preserve under quantization is stable attractor structure

That produced the role/correction/stabilization family.

### Step 10. Aaron identified the right failure mode in the stabilization idea

The stabilization experiments were useful because they failed in an informative way.

They showed:

- explicit late stabilization slightly improved compressibility
- but it hurt exact post-quantized score

That sharpened the question from “should we stabilize?” to “what exactly is safe to stabilize?”

### Step 11. Aaron proposed Lipschitz-like C correction from A/B success

This was the strongest conceptual move after the original mirror hypothesis.

The new idea was:

- `C` should not correct blindly
- `C` should correct only when `A` and `B` were locally successful
- the correction should be bounded
- the surviving structure should be a stable attractor, not fragile float detail

That led to scalar adaptive `C` correction driven by:

- `cosAB`
- `amp_ratio`
- bounded max gain

### Step 12. The scalar adaptive branch failed, but for a precise reason

The scalar adaptive branch did not improve the exact metric, but it was still valuable.

It showed:

- the gate was real
- the gate fired in a specific place
- compressibility improved strongly
- exact score got worse

This is the point where the project learned that “contract inward” is too blunt.

### Step 13. Aaron then pushed from scalar correction to directional correction

Instead of shrinking the whole state, Aaron pushed toward correcting only the bad direction.

That led to the directional `C` branch, where the system subtracts the projected over-amplified `A/B` mode instead of pulling the entire state toward an anchor.

That is the first correction family that remained close enough to the mirror baseline to matter.

### Step 14. A concrete implementation bug was found and fixed

The first adaptive MLX run failed during save because non-array source metadata had been stored in model state.

The bug was diagnosed and corrected by:

- checking the MLX state tree directly
- finding integer leaves in adaptive source metadata
- storing source identifiers as strings instead of tuples of integers

This mattered because it kept the exploration path reproducible instead of leaving a hidden checkpoint hazard in the code.

### Step 15. Aaron asked for outside minds at the right time

The search used other voices when they were useful, not decoratively.

Important external conceptual contributions:

- Singer: keep correction logic in the `GPT` forward seam, not inside `Block`; keep role deltas ephemeral
- Gödel: the relevant invariant may be a quantized orbit, not continuous similarity

Aaron’s role here was not passive. He chose when to bring in those angles and what part of the search they were meant to clarify.

### Step 16. The schedule result survived a real promotion step

The strongest non-smoke proof so far is that the mirror schedule beat cyclic on a longer real-data run:

- `mlx_full_mirror200_realval = 2.40028927`
- `mlx_full_cyclic200_realval = 2.40292304`

That means the core architectural effect is not merely a toy smoke artifact.

### Step 17. Directional correction survived promotion and improved the real metric

The directional branch did not merely survive smoke ranking.

It survived the same `200`-step real-data promotion protocol and beat the best promoted mirror line:

- `mlx_full_mirror200_realval = 2.40028927`
- `mlx_full_mirror_dirc02_200_realval = 2.38989686`

That changed the status of the project. Directional `C` correction stopped being a speculative side branch and became the best promoted result so far.

### Step 18. The first orbit-gated attempt failed in a useful way

After Gödel’s quantized-orbit framing, the first cheap orbit-gated implementation was tested on top of `dirc02`.

What happened:

- the gate was expensive
- the gate was highly selective
- the gate suppressed too much of the useful correction

The branch got smaller on disk, but its exact smoke score got clearly worse.

That was still useful because it told us the problem is not “add a hard quantized-future gate and everything improves.”
It told us the present proxy is too blunt and too costly.

## What The Machine Established

The machine side of the work has reduced the search space substantially.

Known good:

- `3` shared blocks is viable
- mirror reuse beats cyclic reuse
- the effect survives promotion from smoke to a more serious `200`-step real-data run

Known weak or dead:

- `1` shared block is too aggressive
- `pass_x0` reinjection hurts
- blanket revisit damping hurts
- naive phase-split revisit gains hurt badly
- the custom `A A C B B C A A A` schedule underperforms mirror
- scalar adaptive contractive `C` correction improves compressibility but hurts exact score

Known alive:

- mirror schedule
- role-biased `A/B/C` interpretation as a near-tie refinement
- directional `C` correction as a live branch

## Important Reference Runs

### Smoke Baseline Family

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_shared3_cyclic_cmp` | `3` shared blocks, cyclic | `2.66407351` | `3749035` | baseline shared-core line |
| `mlx_shared3_mirror_cmp` | `3` shared blocks, mirror | `2.65437865` | `3744737` | best simple smoke baseline |
| `mlx_shared1_cyclic_cmp` | `1` shared block, cyclic | `2.69754880` | `1576008` | too tied, clearly worse |
| `mlx_shared3_mirror_x0_cmp` | mirror + `pass_x0` | `2.66370193` | `3747526` | worse than mirror |

### Revisit-Gain Family

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_shared3_mirror_revisit_cmp_v2` | mirror + revisit gain | `2.65537863` | `3735691` | coherent but slightly worse |
| `mlx_shared3_mirror_revisit_damp015_cmp` | mirror + revisit gain + damping | `2.66775012` | `3729617` | damping hurts |
| `mlx_shared3_mirror_revisit_phase_cmp` | mirror + phase-split revisit gain | `2.71027250` | `3090581` | collapses score |

### Custom A/B/C Schedule Family

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_custom_aacbbcaaa_cmp` | custom `AACBBCAAA` path | `2.67979177` | `3740198` | worse than mirror |
| `mlx_custom_aacbbcaaa_countgain_cmp` | custom path + revisit-count gain | `2.67346598` | `3736444` | slight improvement, still not competitive |

### Role / Correction / Stabilization Family

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_mirror_rolecorr_cmp` | role gains + fixed `C` correction | `2.65546615` | `3744043` | near tie, alive |
| `mlx_mirror_rolecorr_stab7e2_cmp` | role/correct + sparse late stabilization | `2.65697448` | `3740005` | stabilization hurts |
| `mlx_mirror_rolecorr_stab7e1_cmp` | role/correct + dense late stabilization | `2.65910760` | `3740726` | hurts more |

### Scalar Adaptive C-Correction

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_mirror_role_adaptc04_fix_cmp` | scalar adaptive `C`, max gain `0.04` | `2.66780102` | `3639092` | coherent but clearly bad for score |
| `mlx_mirror_role_adaptc01_cmp` | scalar adaptive `C`, max gain `0.01` | `2.66223291` | `3639589` | less bad, still dead |

Probe files:

- [adaptive_gate_mlx_mirror_role_adaptc04_fix_cmp.json](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf/logs/adaptive_gate_mlx_mirror_role_adaptc04_fix_cmp.json)
- [adaptive_gate_mlx_mirror_role_adaptc01_cmp.json](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf/logs/adaptive_gate_mlx_mirror_role_adaptc01_cmp.json)

Main read:

- the scalar gate fired only on pass `7`
- it improved compressibility but degraded exact post-quantized score
- conclusion: the mechanism was real, but the correction geometry was wrong

### Directional C-Correction

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_mirror_role_dirc04_cmp` | directional `C`, max gain `0.04` | `2.65513593` | `3741527` | much better than scalar branch |
| `mlx_mirror_role_dirc02_cmp` | directional `C`, max gain `0.02` | `2.65466476` | `3741104` | best correction branch so far |
| `mlx_mirror_role_dirc015_cmp` | directional `C`, max gain `0.015` | `2.65685054` | `3743438` | too weak / less good than `0.02` |

Probe file for the best directional smoke run:

- [gate_probe_mlx_mirror_role_dirc02_cmp.json](/Users/aaronday/dev/discrete-mathematics-for-formal-system-design/parameter-golf/logs/gate_probe_mlx_mirror_role_dirc02_cmp.json)

Important directional probe read:

- the gate still fires only on pass `7`
- `mean_active_lambda ~= 0.01344`
- `mean_active_cos_ab ~= 0.99373`
- `mean_active_amp_ratio ~= 2.26765`

Interpretation:

- only one revisit site needs correction
- suppressing the offending direction is much safer than shrinking the whole state
- this is the first correction branch that stayed genuinely close to the mirror baseline

### Real-Data Promotion Runs

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_full_mirror200_realval` | mirror, full real validation, `200` steps | `2.40028927` | `4082425` | best proven promoted line |
| `mlx_full_cyclic200_realval` | cyclic, full real validation, `200` steps | `2.40292304` | `4093619` | mirror still wins |
| `mlx_full_mirror_dirc02_200_realval` | mirror + directional `C`, full real validation, `200` steps | `2.38989686` | `4074385` | directional correction survived promotion and beat mirror |

Key conclusion from promotion:

- mirror beat cyclic by `0.00263377` `bpb` on the exact post-quantized real-data comparison
- that means the schedule effect survived a meaningful promotion step
- directional `C` then beat the promoted mirror line by `0.01039241` `bpb`
- that means the first correction branch was not just alive on smoke; it survived promotion and improved the real metric

### Orbit-Gated Directional Correction

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_mirror_role_dirc02_orbit3_cmp` | `dirc02` + orbit gate, smoke, `3` future steps | `2.67923063` | `3289450` | dead branch in current form |

Important orbit-gate read:

- the orbit gate roughly doubled smoke step time, from about `422ms` to about `942ms`
- exact smoke score got much worse, from `2.65466476` to `2.67873100`
- replay analysis showed mean orbit-match only `0.25` at the active `C` revisit
- effective gate strength collapsed from about `0.01344` to about `0.00314`

Interpretation:

- the current orbit proxy is too selective and too expensive
- it is not yet the right operationalization of Gödel’s idea
- the quantized-basin hypothesis may still be correct, but this implementation is not

### Switched Attractor Pulse

| run_id | idea | exact val_bpb | artifact bytes | verdict |
| --- | --- | ---: | ---: | --- |
| `mlx_mirror_role_dirc02_pulse2_cmp_v2` | `dirc02` + thresholded attractor pulse, `T=2` | `2.66092593` | `3667363` | alive as control law, worse than plain `dirc02` |

Important pulse read:

- this was much cheaper than the hard orbit gate
- smoke step time rose to about `553ms`, not `~940ms`
- replay showed the pulse selected on every active revisit under the current threshold
- replay stress delta was positive, about `0.10065`, so the pulse branch did reduce the cheap quant-stress proxy
- exact score still got worse, from `2.65466476` to `2.66092593`

Interpretation:

- the switched-control idea is coherent
- the current attractor target is still too blunt
- reducing the cheap quant-stress proxy is not enough by itself to preserve the real objective

## Resonance Findings

Resonance-like structure is real in this system, but it does not show up in every metric equally.

Useful:

- pass RMS
- relative delta norm
- schedule-conditioned drift

Not very useful as a steering metric:

- cosine alone

Working heuristic established during the resonance passes:

- pay attention when a schedule causes pass-RMS separation `> 5%` on at least `3` post-divergence passes
- and delta-norm separation `> 10%` on at least `3` post-divergence passes
- ignore cosine unless it becomes much larger than what was observed in the current mirror/cyclic split

## Singer’s Useful Contribution

Singer’s best contribution was architectural hygiene:

- patch shared-core correction logic in the `GPT` forward seam, not inside `Block`
- keep block-role deltas ephemeral, not persistent module state
- avoid introducing non-array state into the MLX checkpoint tree

This prevented a worse design and also helped localize the MLX serialization bug that appeared during the first adaptive gate attempt.

## Gödel’s Idea

Gödel’s contribution was the best conceptual reframing since the original mirror schedule win.

Plain-English version:

- do not correct based only on continuous closeness
- correct only inside regions whose short quantized orbit is already stable
- move inward only up to the point where the quantized orbit would change

The core object is a short quantized orbit under the mirror-step operator:

- `Omega(x) = (Q(x), Q(Mx), Q(M^2 x))`

Where:

- `M` is the mirror schedule operator
- `Q` is the effective quantizer relevant to the final evaluation

Gödel’s suggested invariant:

- if `Omega(A) = Omega(B)`, then a local contraction might be safe
- if not, do nothing

The first cheap proxy has now been implemented once, and it failed.

So the idea itself is not dead, but the current proxy implementation is not the right next operational form.

## Current Best State

### Best Simple Smoke Baseline

- `mlx_shared3_mirror_cmp`
- exact `val_bpb = 2.65437865`

### Best Correction Branch On Smoke

- `mlx_mirror_role_dirc02_cmp`
- exact `val_bpb = 2.65466476`
- gap to mirror baseline: `+0.00028611`

This is close enough to matter.

### Best Real-Data Proven Result

- `mlx_full_mirror_dirc02_200_realval`
- exact `val_bpb = 2.38989686`

### Current Promotion Target

The next open question is no longer whether `dirc02` survives promotion. It did.

The next serious target is:

- a cheaper, better-grounded quantized-basin gate
- or a more targeted quantization-aware correction that preserves the `dirc02` win without doubling runtime

## What Aaron Has Actually Done In Practice

This is the practical summary of Aaron’s part in the loop so far.

Aaron has:

- chosen to prototype locally before paying for serious GPU time
- insisted on a first-principles parse instead of leaderboard imitation
- repeatedly redirected the search away from random tuning and toward mechanism
- supplied the key architectural metaphors that generated the most useful branches
- identified stable attractors as the relevant surviving object under quantization
- asked for instrumentation when raw scores were no longer enough
- pushed the correction idea from blanket stabilization to bounded correction and then to directional correction
- requested durable documentation so the work does not vanish into chat memory

That is a real research contribution, not merely “prompting an assistant.”

## Dead Branches

Unless new evidence emerges, these branches should be treated as closed:

- `1` shared block
- `pass_x0`
- blanket revisit damping
- naive phase-split revisit gain
- custom `AACBBCAAA` schedule as currently implemented
- scalar adaptive contractive `C` correction
- periodic stabilization as currently implemented
- aggressive mirror-cancellation pulse
- lowered `dirc02` max gain at `0.015`
- quant-stress lambda guard as currently implemented

## Mirror-Cancellation Follow-Up

This branch tested a literal phase-cancellation trigger on top of `dirc02`.

The mechanism:

- compare the current mirrored revisit against its earlier mirror partner
- measure uncancelled residual
- only allow the extra attractor pulse when that residual stays above a threshold

### Aggressive Mirror-Cancellation Pulse

- `mlx_mirror_role_dirc02_mirrorpulse_aggr_cmp`
- exact `val_bpb = 2.67254228`
- step average `617.79ms`

This is worse than plain `dirc02` on both score and speed.

### Conservative Mirror-Cancellation Pulse

- `mlx_mirror_role_dirc02_mirrorpulse_cons_cmp`
- exact `val_bpb = 2.64605443`
- step average `519.00ms`

This looked promising on smoke score.

However, the gate probe showed something important:

- pulse selection rate at the active pass was effectively `0.0`
- the extra attractor did not actually fire

So this run should **not** be interpreted as evidence that phase inversion is helping. It is better read as a nearby no-op variant with slight smoke variance.

### Gate Probe Read

Probe output:

- `gate_probe_mirrorpulse_compare.json`

Key read:

- aggressive branch active pass `7`
- aggressive pulse selection mean `0.0`
- aggressive mirror residual mean about `0.9937`
- conservative branch active pass `7`
- conservative pulse selection mean `0.0`
- conservative mirror residual mean about `0.9955`

Interpretation:

- the mirror-cancellation trigger is not yet a live steering mechanism
- the aggressive and conservative variants changed compile/numeric behavior, but did not actually activate the extra pulse

## Lower-Lambda Follow-Up

After the mirror-cancellation probe showed a dead gate, the next direct test was:

- keep plain `dirc02`
- lower `SHARED_CORE_DIRECTIONAL_CORRECT_MAX_GAIN` from `0.020` to `0.015`

Result:

- `mlx_mirror_role_dirc015_cmp`
- exact `val_bpb = 2.66421949`

This is a clean regression.

So the current best operational read is:

- keep `dirc02`
- keep the existing `0.020` max gain
- do not adopt the mirror-cancellation pulse as a working branch yet

## Quant-Stress Lambda Guard

The next branch tried to stay on the live `dirc02` seam and only reduce `lambda` when the directional correction itself increased local quantization stress.

Run:

- `mlx_mirror_role_dirc02_stressguard_cmp`
- exact `val_bpb = 2.67273248`

Probe:

- `gate_probe_mlx_mirror_role_dirc02_stressguard_cmp.json`

Important read:

- stress-guard factor on the active pass was `1.0`
- stress delta on the active pass was `0.0`

So this branch is not merely bad. It is another dead control:

- it did not actually reduce `lambda`
- it still made the run worse

That is enough to close it in its current form.

## Open Questions

1. Can a quantization-aware control improve `dirc02` without destroying runtime or selectivity?
2. Should that control be soft and local rather than a hard orbit gate?
3. Is the next useful abstraction still Gödel’s quantized-orbit invariant, or something cheaper such as a quantization-sensitivity bias on the correction strength?
4. Should the current trainer logic be compacted into helpers before any further idea expansion?

## Immediate Next Step

Do not promote another branch until a probe shows a control with:

- nonzero engagement on the active pass
- bounded selectivity
- and a smoke score that is genuinely attributable to that control

Right now the only trusted promoted line remains:

- `mlx_full_mirror_dirc02_200_realval`

## Current Status Snapshot

The current trusted best promoted result is still:

- `mlx_full_mirror_dirc02_200_realval`
- exact `val_bpb = 2.38989686`
- int8+zlib artifact `4,074,385` bytes

That artifact size matters. It means the live branch is still using only about one quarter of the `16 MB` budget, so the next serious search pressure moved away from dead controls and toward spending more capacity.

### Depth-Only Follow-Up Closed

A direct effective-depth push was tested by increasing mirrored shared-core recurrence without increasing unique block capacity:

- `mlx_mirror13_dirc02_cmp`
- config: `NUM_LAYERS=13`, `SHARED_CORE_BLOCKS=3`, `mirror + dirc02`
- exact `val_bpb = 2.67693325`
- int8+zlib artifact `3,491,385` bytes

Interpretation:

- extra revisit depth alone made the branch worse
- the loop is not simply under-iterated
- the next useful spend is richer blocks, not more passes through the same mirrored carrier

### Capacity Follow-Up Alive

The next branch increased useful block capacity while keeping the proven recurrence mechanism fixed:

- `mlx_mirror_mlp3x_dirc02_cmp`
- config: `MLP_MULT=3`, `SHARED_CORE_BLOCKS=3`, `mirror + dirc02`
- model params `7,610,404`
- exact `val_bpb = 2.64687904`
- int8+zlib artifact `4,545,781` bytes

This is the first clean post-`dirc02` improvement in the current burst that is attributable to a real architectural change rather than a dead gate or smoke variance.

### Promotion In Flight

Because the `MLP_MULT=3` branch improved exact smoke score, it has been promoted to the same real-data protocol used for the current best line:

- run id: `mlx_full_mirror_mlp3x_dirc02_200_realval`
- protocol: `200` steps, real `fineweb10B_sp1024` validation, exact int8 roundtrip verification
- status: crashed during validation on `2026-03-21`

Observed failure point:

- training completed through `step:200/200`
- validation only reached `val_progress:1/30`
- no serialized model artifact was written

Working interpretation:

- the branch itself is still alive
- the most likely problem is validation-memory pressure from `VAL_MAX_BATCH_TOKENS=2097152` on the larger `MLP_MULT=3` model
- the next retry should keep the same training branch and lower validation batch size before changing architecture

Decision rule:

- if this beats `2.38989686`, it becomes the new live leader
- if it does not, the next branch should spend budget on width, not more mirrored depth and not more control logic

### M4 Retry And Outcome

The first real-data promotion of `MLP_MULT=3` crashed the whole machine during validation.

Retry:

- `mlx_full_mirror_mlp3x_dirc02_200_realval_vb524k`
- same branch, but `VAL_MAX_BATCH_TOKENS=524288`

Result:

- exact `val_bpb = 2.38131855`
- previous best was `2.38989686`
- improvement: about `0.00858 bpb`
- int8+zlib artifact `5,012,890` bytes

Interpretation:

- the branch was real
- the crash was validation-pressure, not a false-positive model
- richer block capacity beat the earlier best shared-core line

### Branch A / Branch B Tournament

With the new M4-safe wrapper in place, two follow-up smoke branches were run back to back.

#### Branch A: Width-First Shared-Core

- `mlx_mirror_dim640_mlp3x_dirc02_cmp`
- config: `MODEL_DIM=640`, `NUM_HEADS=10`, `NUM_KV_HEADS=5`, `MLP_MULT=3`, keep `mirror + dirc02`
- exact `val_bpb = 2.66895938`
- int8+zlib artifact `6,190,593` bytes

This is a clean regression. More width on top of the winning shared-core family did not help.

#### Branch B: Sequential Capacity Challenger

- `mlx_seq_mlp3x_cmp`
- config: full sequential `9` unique blocks, `MLP_MULT=3`
- exact `val_bpb = 2.61722235`
- int8+zlib artifact `12,448,165` bytes

This is a strong smoke win over the current shared-core smoke leader.

Interpretation:

- the latest gains may have come more from capacity allocation than from shared-core structure
- shared-core is still alive on promoted real-data evidence
- but it no longer deserves default priority over a stronger sequential-capacity challenger

### New Promotion In Flight

Because Branch B won clearly on smoke and still fits under the `16 MB` cap, it has been promoted:

- run id: `mlx_full_seq_mlp3x_200_realval_vb524k`
- protocol: real `fineweb10B_sp1024`, `200` steps, exact int8 roundtrip, M4-safe validation batch

## Iteration Annotation: Late-C Clarification

This iteration added a useful correction to the framing around damping and delayed correction.

Aaron's useful push was:

- reject the earlier sloppy reading of "Lipschitz" as something to implement directly
- restate the real concern as information loss from early blunt correction
- point out that shaving amplitude peaks too early can destroy relative separation that may still matter after quantization
- insist that the live question is not generic smoothness, but whether the re-encoding stage is being invoked too early and too bluntly

That contribution is consistent with the machine evidence.

It matches the dead-branch pattern:

- blanket revisit damping hurt
- scalar adaptive contractive `C` hurt
- stabilization-style controls hurt
- attractor and orbit-style pulls reduced proxies without preserving the exact roundtrip objective

So the anti-smoothing instinct was good.

Aaron's overreach in this same iteration was:

- jumping from "early blunt correction is bad" to "therefore much longer `A/B/A/B/...` carrier loops before `C` should win"

The log does **not** support that stronger claim.

Closest counterevidence already in hand:

- more mirrored recurrence without more unique capacity (`mlx_mirror13_dirc02_cmp`) regressed badly
- the custom `AACBBCAAA` schedule also regressed

So the correct update is narrower:

- preserve amplitude longer, yes
- delay correction selectively, yes
- but do **not** assume that extra carrier passes through the same reused blocks are inherently good

The real search object is probably not a single scalar like "how long do we wait before `C`."

It is more likely a small structural family:

- fixed-depth schedules that let `A/B` carry longer
- sparse late `C` placements
- at least one late `C` revisit so the proven directional seam can still engage
- correction strength tied to quantization-sensitive structure, not generic contraction

Working practical read:

- this is still partly a parameter search
- but the parameter is probably not one knob
- it is a bounded architectural micro-search over late-`C` schedule geometry plus quantization-aware correction timing

Best current bounded next-step family:

- `ABABABCAC`
- `ABABABCBC`
- `ABABACBAC`

These should be treated as **fixed-depth** tests, not invitations to "wait until God or the checkered flag."

If OpenAI compute arrives, this is a good place to spend it:

- launch `5-6` tightly chosen late-`C` variants
- keep `dirc02` alive
- require probe evidence of a real late active pass
- promote only the variants whose gain is attributable, not just smoke variance

## Iteration Annotation: Interpreting A Possible `MLP_MULT=3` Win

This iteration clarified what a successful `MLP_MULT=3` promotion would and would not mean.

If `mlx_full_mirror_mlp3x_dirc02_200_realval` beats the current trusted line by a real margin, the strongest update is:

- the shared recurrent carrier is probably real
- targeted late correction is probably real
- the next bottleneck is more likely operator capacity than "invoke `C` later and later"

That would support the broad systems hypothesis:

- post-quantized performance depends on internal organization, not just standard tuning

But it would weaken the narrower reading:

- "the main missing ingredient is longer `A/B` chewing before `C`"

If `MLP_MULT=3` wins, the cleaner explanation is:

- richer transforms beat extra reuse depth
- artifact budget should first be spent on stronger operators before more schedule cleverness

Working decision rule:

- if the promoted `MLP_MULT=3` branch improves by less than about `0.005 bpb`, treat it as encouraging but not decisive
- if it improves by about `0.01 bpb` or more, default belief should shift toward capacity-first, schedule-second
- if it fails to beat `2.38989686`, the late-`C` family gets more oxygen again

Compute-credit implication:

- with credits, we stop treating each branch as a sacred single guess
- the immediate value is bounded parallel family search, repeated promotion, and CUDA transfer checks
- the biggest direct beneficiaries would be the `MLP_MULT=3` / width-first family, the late-`C` fixed-depth family, and CUDA verification of whichever branch survives locally

## Ideation Note: Phase / Harmonics

This idea was raised as a possible explanation for why early blunt correction feels wrong:

- amplitude peaks may carry real information
- shaving them too early may reduce quantization-relevant separation
- perhaps some cheaper harmonic object survives better than the raw path

The useful part of that intuition is:

- coarse structured variation can survive lossy channels better than fragile detail

But the first literal payload reading is probably wrong for this project.

Why the full payload-harmonic idea is weak:

- the challenge score depends on the final exact int8 roundtrip model artifact, not the training-time waveform
- dense transformer weight coordinates do not live on a stable physical axis where ordinary harmonic coding is naturally meaningful
- current probes show one late amplified recurrent mode, not a rich oscillatory family
- fine phase-coded residual structure is exactly the kind of thing int8 clipping tends to destroy
- any harmonic payload scheme would also have to pay code and decode overhead inside the artifact budget

Current best critical read:

- phase / harmonics are probably the wrong object at the full-payload level
- they may still be alive as a **control-basis** idea

That narrower version means:

- use spectral or phase-like parameterizations only over meaningful ordered axes such as layer depth or revisit index
- consider a tiny basis for step scales, revisit gains, or correction strength over depth
- only pursue this if probes show real oscillatory structure across revisit count rather than simple monotone amplification

So the idea is not closed, but it has been demoted:

- probably wrong as artifact encoding
- still plausible as a compact control-law parameterization

## Iteration Annotation: Localized QAT Or Kill It

External critique in this iteration was strong and mostly correct.

Main correction:

- the current mechanism story is too confident relative to the actual challenge regime
- many of the detailed "live branch / dead branch / active late mode" claims were learned from MLX smoke-scale runs and should be treated as provisional local steering, not established frontier mechanism

Most important priority update:

- the quantization-robustness family is currently a low-priority branch on the best promoted local line
- the exact post-quant gap on the current trusted `dirc02` result is already very small
- therefore a pure "survive quantization better" idea has limited upside unless it also changes the learned float solution itself

Best critical reduction of the idea:

- "learn quantization-safe equivalence classes" is too abstract and mostly the wrong object
- the only coherent narrow version is localized QAT on the actually sensitive locus
- in plain terms: fake-quantize the measured sensitive pass, measure the induced drift, and penalize only the harmful component of that drift

That means:

- no payload coding theory
- no hidden-state class language
- no broad control-basis expansion before diagnostics

Required order of operations if this family is revisited:

1. finish and evaluate the promoted `MLP_MULT=3` branch
2. if needed later, run a diagnostic to localize where quantization damage actually enters
3. kill the whole idea quickly unless late `C` explains a meaningful share of the quantization penalty
4. only then consider one scalar localized QAT auxiliary, not a broader theory stack

Current decision:

- wait for `MLP_MULT=3`
- treat capacity-first as the current lead explanation if it promotes successfully
- defer any quantization-consistency training branch until the diagnostic case is strong enough to justify it

## Iteration Annotation: LZMA Path Verified

The `lzma` artifact path is now verified end-to-end on the standalone repo.

Verification run:

- `mlx_seq_mlp4x_lzma_cmp_v2`
- exact `val_bpb = 2.61172375`
- artifact line: `serialized_model_int8_lzma:13522952 bytes`

What this established:

- the trainer now writes `.ptx` artifacts by default
- the trainer can reload the `lzma` artifact and complete the final exact roundtrip evaluation
- the exact roundtrip score remains consistent with the earlier `MLP_MULT=4` smoke branch

Practical implication:

- the storage-format issue is no longer the active bottleneck
- future search should treat `lzma` as the default artifact path and focus back on model quality per byte, not on serializer rescue work

## Iteration Annotation: Hypothesis Worksheet Added

The search has now reached the point where additional nearby local runs are less valuable than one good new mechanism.

To make that explicit, a dedicated worksheet now lives here:

- `docs/parameter-golf-hypothesis-worksheet.md`

Its role is simple:

- record the current dead / alive / unclear map
- force a 45-minute paper session before the next run family
- require a one-slot hypothesis template with a kill criterion

This is a guardrail against another round of local thousandth-shaving on a family that is already flattening.

## Iteration Annotation: Quantizer Floor Stress Is Real, But The First Exception Still Lost

This iteration started from a useful external read rather than a new training mechanism.

The proposed update was:

- large 2D tensors may be over-quantized by the current fixed per-row floor
- maybe the next "salience field exception" is not architectural at all
- maybe one tensor family deserves a non-generic quantization rule

The concrete claim from the tensor-atlas pass was:

- for big matrix families like `mlp.proj`, `mlp.fc`, and `attn.c_q`, the natural per-row scale is below the current fixed floor
- therefore the current rule
  - `scale = max(clip_abs / 127, 1 / 127)`
  is often dominated by `1 / 127`
- this means a large share of the int8 grid is unused on those tensors

Local verification on the best full artifact confirmed the narrow factual part.

On the current best promoted line:

- `logs/mlx_full_seq_mlp4x_200_realval_vb524k_mlx_model.npz`

sample checks showed:

- `blocks.0.mlp.proj.weight`
  - natural median / max row scale about `0.00093 / 0.00170`
  - current floor scale `0.007874`
  - current max code usage only about `15-27` out of `127`
- `blocks.0.mlp.fc.weight`
  - natural median / max row scale about `0.00121 / 0.00324`
  - current max code usage about `20-52`
- `blocks.0.attn.c_q.weight`
  - natural median / max row scale about `0.00115 / 0.00203`
  - current max code usage about `19-33`

So the quantizer-floor story is real enough to matter.

But the first naïve fixes were immediately too expensive.

Offline recompression on the current best full artifact showed:

- current quantizer with `lzma`: `14985872` bytes
- tensor-floor everywhere: `19650520` bytes
- `mlp.proj` tensor-floor only: `17219364` bytes
- `mlp.proj` no-floor only: `18391732` bytes

This killed the broad reading:

- removing the fixed floor is not a free gain
- even a seemingly focused `mlp.proj` exception can blow the full `16,000,000`-byte budget

That led to the actual local test:

- keep the change default-off
- touch only `.mlp.proj.weight`
- interpolate the floor between the current `1 / 127` and a tensor-specific floor
- test only one smoke against the exact local control

Offline budget sweep for that softened family on the best full artifact:

- `alpha=0.05` -> `15059424` bytes
- `alpha=0.10` -> `15190860`
- `alpha=0.15` -> `15227864`
- `alpha=0.20` -> `15304608`
- `alpha=0.25` -> `15398136`
- `alpha=0.30` -> `15500392`

So `alpha=0.30` was a safe first smoke from a full-artifact budget perspective.

Exact smoke run:

- `mlx_seq_mlp4x_projfloor_a030_lzma_v1`
- `soft_floor_alpha = 0.30`
- pattern: `.mlp.proj.weight`

Result:

- exact smoke control:
  - `mlx_seq_mlp4x_lzma_cmp_v2`
  - exact `val_bpb = 2.61172375`
  - artifact `13522952` bytes
- tested branch:
  - `mlx_seq_mlp4x_projfloor_a030_lzma_v1`
  - exact `val_bpb = 2.61243245`
  - artifact `13628196` bytes

Decision:

- kill this first soft-floor exception
- it lost on exact post-roundtrip score by about `+0.00070870`
- it also increased artifact size by `105244` bytes

Most important update from this iteration:

- quantizer stress alone is not enough to justify an exception
- "this tensor is underusing the grid" is not the same as "this exception improves the challenge score"

So the next version of the idea must be stricter:

- stop ranking sacred candidates only by quantizer pathology
- measure actual loss delta for single-tensor or tensor-family roundtrip substitution
- treat byte cost and semantic sensitivity as separate axes

Current decision rule after this result:

- no more blind soft-floor tuning
- build a sacredness script that measures:
  - float baseline loss
  - loss after replacing exactly one tensor or one coherent tensor family with its roundtripped version
  - byte share and quantization stress only as supporting diagnostics

This iteration did not produce a new winning family, but it did tighten the search discipline:

- the artifact path is an active source of real structure
- however, local exceptions still need to earn themselves on exact score, not just on an appealing tensor-atlas story

## Iteration Annotation: Sacredness Measured A Real Tensor, Then Killed The Wrong Refinement

The next step after the failed soft-floor exception was to stop guessing from stress proxies alone.

A new local script:

- `scripts/analyze_tensor_sacredness.py`

measured the actual loss delta from restoring one tensor family or one tensor from float into the quantized-roundtrip baseline.

On a full smoke-val pass, the ranking came back:

- family sacredness:
  - `mlp.proj.weight` `dBpb ~= +0.000697`
  - `attn.proj.weight` `dBpb ~= +0.000168`
  - `tok_emb.weight` `dBpb ~= +0.000062`
- top single tensors:
  - `blocks.0.mlp.proj.weight` `dBpb ~= +0.000596`
  - `blocks.0.attn.proj.weight` `dBpb ~= +0.000121`
  - `blocks.1.mlp.proj.weight` `dBpb ~= +0.000091`

This gave the first strongly grounded sacred candidate:

- `blocks.0.mlp.proj.weight`

The first exploitation move was the simplest possible preservation path:

- keep exactly `blocks.0.mlp.proj.weight` in fp16 passthrough

Exact smoke result:

- control:
  - `mlx_seq_mlp4x_lzma_cmp_v2`
  - exact `val_bpb = 2.61172375`
- keep-float sacred tensor:
  - `mlx_seq_mlp4x_keepf_block0proj_v1`
  - exact `val_bpb = 2.61001513`
  - artifact `14858772` bytes

So this family produced a real local win:

- score gain about `-0.00170862`
- still under smoke budget

That was important because it separated:

- a real sacred tensor exploit
from
- generic "something in the artifact path is off"

A nearby backup locus was then tested to make sure this was not just random keep-float churn:

- `blocks.0.attn.proj.weight`

Exact smoke result:

- `mlx_seq_mlp4x_keepf_block0attnproj_v1`
- exact `val_bpb = 2.61456176`

This lost clearly, which narrowed the live family:

- keep-float is not a generic one-tensor win
- early `mlp.proj` is special in a way early `attn.proj` is not

The next question became:

- can the proven `blocks.0.mlp.proj.weight` exploit be made cheaper without losing the gain?

To answer that, two more local helpers were added:

- `scripts/analyze_tensor_row_sacredness.py`
- `scripts/estimate_row_subset_passthrough_size.py`

Row analysis showed the important warning:

- the row signal inside `blocks.0.mlp.proj.weight` is diffuse, not concentrated in a tiny subset
- top-8 rows only captured about `1.8%` of row-relative-error mass
- top-16 about `3.5%`
- top-32 about `6.8%`
- top-64 about `13.3%`
- top-128 about `26.3%`

So a very small row carveout was already suspect on first principles.

Even so, one disciplined refinement was worth trying because it was cheap:

- top-64 sacred rows of `blocks.0.mlp.proj.weight` kept in fp16 passthrough
- estimated smoke artifact delta: `+188008` bytes

Exact smoke result:

- `mlx_seq_mlp4x_keepf_block0proj_rows64_v1`
- exact `val_bpb = 2.62784548`
- artifact `12755236` bytes

Decision:

- kill the row-subset refinement immediately
- it lost by about `+0.01612173` versus the smoke control
- it is not a cheaper version of the full-tensor keep-float win
- it is a different family, and that family is locally bad

Current read after this chain:

- sacredness measurement was worth doing
- it found a real exploit
- the winning unit was the whole tensor, not a sparse row subset
- naive attempts to cheapen that exploit can destroy the effect entirely

So the live lesson is stricter now:

- "sacred tensor" can be real
- but the preserved semantics may be distributed across the tensor rather than localized to a small row family
- byte-cheap refinements still need exact score proof, not just a plausible decomposition story

One more cheapening attempt was still worth trying before closing this branch:

- keep the whole sacred tensor
- but only soften the quantizer floor on that exact tensor
- do not widen back out to the full `.mlp.proj.weight` family

This was the cleanest remaining version of the soft-floor idea:

- tensor: `blocks.0.mlp.proj.weight`
- `alpha = 0.30`
- estimated smoke artifact delta: `+61272` bytes

Exact smoke result:

- `mlx_seq_mlp4x_projfloor_block0_a030_v1`
- exact `val_bpb = 2.61193999`
- artifact `13181736` bytes

Compared to control:

- control exact `2.61172375`
- miss: about `+0.00021624`

Decision:

- kill tensor-specific soft-floor too
- it was cheap enough
- but it still did not beat the control on exact post-roundtrip score

That closes the soft-floor family more firmly:

- broad soft-floor lost
- tensor-specific soft-floor also lost
- the surviving exploit remains whole-tensor fp16 preservation of `blocks.0.mlp.proj.weight`
- the remaining problem is not "find a narrower soft-floor"
- it is "find a cheaper whole-tensor representation that preserves the same semantics"

Operational note after this branch:

- the live sacred-tensor smoke win
  - `mlx_seq_mlp4x_keepf_block0proj_v1`
  - exact `val_bpb = 2.61001513`
  was archived into `results/` with a normalized report bundle under:
  - `results/reports/mlx_seq_mlp4x_keepf_block0proj_v1/`

Immediate local limitation:

- a full local promotion of this sacred-tensor exploit could not be run on `2026-03-23`
- the required full dataset prefix `data/datasets/fineweb10B_sp1024/` is not currently present in this standalone repo
- so local evidence remains:
  - smoke win for whole-tensor keep-float
  - smoke loss for row-subset keep-float
  - smoke loss for tensor-specific soft-floor

If larger compute becomes available, the work should stop looking like local toy search and become a focused program:

1. Promote the live sacred-tensor exploit on real full data immediately.
   - run the exact `blocks.0.mlp.proj.weight` fp16-preserve branch on the full validation protocol
   - verify whether the smoke win survives at scale
   - measure exact artifact size and whether training adapts around the exception

2. Search cheaper whole-tensor representations, not sparse row approximations.
   - fp16 whole-tensor keep-float is the current score upper bound
   - the next real family is a cheaper whole-tensor carrier for that same information
   - candidates include:
     - tensor-specific mixed precision
     - learned sidecar residuals
     - low-rank + residual only if the actual reconstructed score earns it
     - tensor-specific packing / staging changes that preserve the whole tensor's semantics

3. Spend compute on robustness, not just novelty.
   - repeat the live exploit across seeds
   - test interaction with longer training, different warmup, and larger real validation
   - stop promoting ideas that only survive one smoke run

4. Only after the sacred exploit is characterized should the search widen again.
   - if the whole-tensor effect survives promotion, it becomes a new anchor family
   - if it collapses on full data, then sacred-tensor preservation was only a local smoke artifact

## Iteration Annotation: The Sacred Tensor Survives Promotion, But Misses The Cap

After the full published dataset prefix was downloaded locally, the whole-tensor sacred exploit was promoted to the real-data MLX protocol:

- `blocks.0.mlp.proj.weight` kept in fp16 passthrough
- same plain sequential `MLP_MULT=4` control family
- `200` steps
- full validation
- exact `lzma` roundtrip verification

The first promoted run:

- `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v1`

trained through `step:200/200` but the machine crashed at `val_progress:1/119`, before artifact save. This looked like validation-time memory pressure rather than a training failure.

The rerun:

- `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2`
- reduced `VAL_MAX_BATCH_TOKENS` from `524288` to `131072`

completed successfully.

Result:

- exact `val_bpb = 2.35551193`
- artifact `16,263,292` bytes via `lzma`

Comparison to the current best capped promoted local result:

- capped control:
  - `mlx_full_seq_mlp4x_200_realval_vb524k`
  - exact `val_bpb = 2.35796063`
  - artifact `14,849,696` bytes
- sacred-tensor promoted run:
  - `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2`
  - exact `val_bpb = 2.35551193`
  - artifact `16,263,292` bytes

So the most important update is now unambiguous:

- the sacred-tensor effect survives promotion
- it is not just a smoke artifact
- it produces a real full-data score gain of about `-0.00244870`

But:

- it misses the decimal `16,000,000`-byte cap by `263,292` bytes

This changes the search problem materially.

Before this run, the open question was:

- does whole-tensor preservation of `blocks.0.mlp.proj.weight` survive full promotion at all?

After this run, the open question is:

- can we recover roughly `263k` bytes without giving back more than `0.00245` bpb?

That is a much tighter and more promising problem.

Current recommendation after this result:

- do not go back to broad architecture wandering
- treat whole-tensor preservation of `blocks.0.mlp.proj.weight` as a live promoted family
- focus the next search on budget recovery:
  - cheaper whole-tensor carrier for that sacred tensor
  - or a paired low-salience sacrifice elsewhere that buys back about `263k`

## Iteration Annotation: Single Attention-Block Sacrifice Is Too Weak

The first concrete budget-recovery hypothesis after the promoted sacred-tensor result was:

- keep the live sacred tensor exactly:
  - `blocks.0.mlp.proj.weight` in fp16 passthrough
- try to buy back the missing `263,292` bytes by making one weak, low-salience attention tensor cheaper at serialization time
- do this offline, without retraining, on the promoted float artifact:
  - `logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz`

Why this was a disciplined next move:

- the promoted sacred-tensor branch already proved the family works on score
- the cap miss was small in absolute terms:
  - about `1.65%` over the `16,000,000` byte cap
- non-sacred passthrough bytes were far too small to solve it:
  - total passthrough in the promoted artifact: `2,179,360` bytes
  - sacred tensor alone: `2,097,152` bytes
  - everything else in passthrough combined: only `82,208` bytes

So the first recovery mechanism was:

- leave the sacred tensor untouched
- pick one quantized attention block
- apply a harsher tensor-local clip percentile only to that block
- reserialize with `lzma`
- exact-eval the reconstructed model on a fixed full-data `128`-sequence validation slice

To make this repeatable, a dedicated offline sweep script was added:

- `scripts/sweep_budget_recovery.py`

The script:

- preserves one named sacred tensor in fp16 passthrough
- sweeps a candidate tensor family block-by-block
- tests harsher local clip percentiles
- records artifact bytes, bytes over cap, and exact `val_bpb`

### First sacrifice family: `attn.c_q.weight`

This was chosen first because earlier sacredness work had consistently ranked `attn.c_q.weight` as one of the least sacred sizable families.

Baseline for the sweep:

- promoted sacred-tensor artifact:
  - `16,263,292` bytes
- fixed `128`-sequence full-data eval slice:
  - baseline keep-float `val_bpb = 2.39749507`

Best result from the `attn.c_q.weight` sweep:

- `blocks.2.attn.c_q.weight` at clip `99.5`
- artifact `16,201,572` bytes
- savings vs sacred baseline: `61,720` bytes
- still over cap by `201,572` bytes
- `val_bpb = 2.39752396`
- score delta vs sacred baseline: `+0.00002889`

Other noteworthy `attn.c_q` outcomes:

- most variants saved only a few kilobytes, or even increased artifact size
- the most aggressive clipping often hurt score without materially improving bytes
- no single `attn.c_q` block came remotely close to recovering the missing `263k`

Conclusion after the `attn.c_q` sweep:

- one harsher `attn.c_q` block is not enough
- this family is too weak as a one-block sacrifice

### Second sacrifice family: `attn.proj.weight`

This was the next large, relatively low-salience attention family to test under the same offline sweep.

Best result from the `attn.proj.weight` sweep:

- `blocks.1.attn.proj.weight` at clip `99.5`
- artifact `16,250,364` bytes
- savings vs sacred baseline: `12,928` bytes
- still over cap by `250,364` bytes
- `val_bpb = 2.39762478`
- score delta vs sacred baseline: `+0.00012971`

Other `attn.proj` observations:

- savings were generally even smaller than for `attn.c_q`
- the family was more likely to lose score for negligible byte improvement
- no single `attn.proj` block was a serious cap-recovery candidate

### What this means

The important discovery is not just that the tested blocks lost.

It is that the tactic itself looks wrong:

- making one weak attention block harsher through local clipping does not buy enough compressed bytes
- the resulting `lzma` savings are tiny relative to the needed `263,292`
- even the best single-block `attn.c_q` candidate only bought about `61.7k`
- the best `attn.proj` candidate only bought about `12.9k`

So the new live read is:

- the sacred tensor itself is real
- but "recover the budget by making one low-salience attention block cheaper with harsher clipping" is not a viable family

This should now be downranked as a recovery mechanism.

### Updated recommendation

Do not spend more local search on:

- single-block harsher clipping for `attn.c_q.weight`
- single-block harsher clipping for `attn.proj.weight`

The next budget-recovery family should be qualitatively different:

- a cheaper whole-tensor carrier for `blocks.0.mlp.proj.weight`
- or a broader, deliberately structured sacrifice that is large enough to matter in bytes

But the evidence now says clearly:

- a one-block attention sacrifice is too small a lever

## Iteration Annotation: Residual Sidecar Wins Under The Cap

After the one-block attention sacrifice family was downranked, the next carrier family tested was:

- keep the whole sacred tensor locus:
  - `blocks.0.mlp.proj.weight`
- but replace full fp16 passthrough with:
  - normal int8 quantization for that tensor
  - plus a low-rank fp16 residual sidecar

This was implemented as an offline artifact-only family:

- no retraining
- normal quantized artifact schema preserved
- one additive sidecar attached to the target tensor during dequantization

Files added / touched for this experiment:

- `train_gpt_mlx.py`
  - dequantization now accepts an optional tensor-local residual sidecar
- `scripts/sweep_residual_sidecar.py`
  - builds and exact-evals residual-sidecar artifacts offline

The important structural check before running the sweep:

- residual tensor:
  - `R = W_float - dequant(quantize(W_float))`
  for `blocks.0.mlp.proj.weight`
- relative residual norm:
  - about `0.0942`

The residual was not sharply low-rank:

- rank `64` captures about `23.7%` of residual energy
- rank `128` about `42.6%`
- rank `192` about `58.2%`
- rank `224` about `64.8%`

So this family was worth testing, but only as a bounded, empirical sweep rather than a presumed solution.

### Slice sweep on promoted artifact

Using the promoted sacred-tensor float artifact:

- `logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz`

and a fixed `128`-sequence full-data validation slice, the residual sidecar sweep gave:

- keep-float baseline:
  - artifact `16,263,292`
  - `val_bpb = 2.39749507`
- plain quant baseline:
  - artifact `14,813,572`
  - `val_bpb = 2.39789063`

Best slice result:

- rank `64`
- artifact `15,109,864`
- under cap by `890,136`
- `val_bpb = 2.39777665`

This was:

- better than plain quant by about `-0.00011398`
- but still worse than full keep-float by about `+0.00028158`

So the slice result said:

- the family is real
- but plain low-rank sidecars only recover part of the sacred-tensor gain

### Full exact offline evaluation

Because rank `64` was the best slice candidate and already comfortably under cap, it was promoted to a full exact offline validation pass on the full local validation set.

Result:

- rank `64`
- artifact `15,109,864`
- under cap by `890,136`
- exact `val_bpb = 2.35570158`

Comparison:

- current best capped promoted result:
  - `mlx_full_seq_mlp4x_200_realval_vb524k`
  - exact `2.35796063`
  - artifact `14,849,696`
- full sacred keep-float promoted result:
  - `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2`
  - exact `2.35551193`
  - artifact `16,263,292`
- residual sidecar rank `64`:
  - exact `2.35570158`
  - artifact `15,109,864`

This means:

- the residual-sidecar family **does** produce a capped promoted result better than the current best capped local run
- it gives back only about `0.00018965` relative to the over-budget full keep-float sacred result
- but it stays safely under the `16,000,000` byte cap

This is the most important update from this branch:

- we now have a cheaper whole-tensor carrier that preserves most of the sacred-tensor gain
- it is not as good as full fp16 keep-float
- but it is good enough to become the new live capped promoted leader locally

Current read after this result:

- whole-tensor carriers were the right family
- plain global low-rank is not magical, but it is good enough to matter
- the problem is no longer "does a cheaper whole-tensor carrier exist?"
- it is now "can a more structured carrier beat rank-64 residual sidecar under the same cap?"

## Iteration Annotation: Naive Column Tiling Does Not Beat Global Rank-64

The next carrier family after the capped rank-64 residual sidecar winner was a simple block-local variant:

- same sacred tensor:
  - `blocks.0.mlp.proj.weight`
- same offline artifact-only setup
- but instead of one global residual factorization, split the tensor into column tiles
- fit one low-rank residual sidecar per tile

Implementation:

- `train_gpt_mlx.py`
  - dequantizer now also supports `residual_tiled_low_rank_v1`
- `scripts/sweep_tiled_residual_sidecar.py`
  - offline sweep over tile widths and per-tile ranks

This first tiled pass intentionally stayed simple:

- tile widths:
  - `1024`
  - `512`
- per-tile ranks:
  - `16, 24, 32, 40, 48, 64`
- eval slice:
  - `128` sequences from the full local validation set

Reference points on the same slice:

- keep-float sacred baseline:
  - artifact `16,263,292`
  - `val_bpb = 2.39749507`
- plain quant baseline:
  - artifact `14,813,572`
  - `val_bpb = 2.39789063`
- global residual sidecar best:
  - rank `64`
  - artifact `15,109,864`
  - `val_bpb = 2.39777665`

Best tiled result:

- tile width `1024`
- rank `32` per tile
- artifact `14,991,684`
- `val_bpb = 2.39779253`

That means:

- it beat plain quant by about `-0.00009810`
- but it still lost to the global rank-64 sidecar by about `+0.00001588`

The smaller `512`-column tiling did worse:

- most `512`-tile candidates were at or above plain quant
- even the best `512`-tile variant did not materially challenge the global carrier

So the update is:

- naive column-local tiling is not enough
- more locality by itself did not beat the current global rank-64 carrier
- the next structured-carrier step, if this line continues, should not just be "more tiles"

Current recommendation after this branch:

- keep the capped rank-64 residual sidecar as the live leader
- downrank naive column-tiled low-rank residuals
- if Claude or future local work pushes further, the next structured carrier should likely be:
  - mixed global + local residual
  - or a codebook / PQ-style carrier
  - or a tiling aligned to learned structure rather than uniform fixed-width columns

## Iteration Annotation: Residual Sidecars Fit The Budget, But Not The Gain

The next whole-tensor carrier family after the failed attention-sacrifice sweeps was the most direct linear-algebra baseline:

- keep the normal int8 artifact for all tensors
- target the one live sacred tensor:
  - `blocks.0.mlp.proj.weight`
- compute its quantization residual:
  - `R = W_float - W_quant_roundtrip`
- approximate that residual with a low-rank fp16 sidecar
- reconstruct on load as:
  - `W_hat = W_q + A @ B`

This was tested offline on the promoted sacred-tensor float artifact using a dedicated sweep script:

- `scripts/sweep_residual_sidecar.py`

The sweep used the same full published dataset prefix, but only a fixed `128`-sequence validation slice for quick ranking.

Baselines on that slice:

- keep-float sacred tensor:
  - artifact `16,263,292`
  - `val_bpb = 2.39749507`
- plain quantized artifact:
  - artifact `14,813,572`
  - `val_bpb = 2.39789063`

So on this slice the full sacred keep-float effect is about:

- `-0.00039556 bpb`

The residual-sidecar sweep tested ranks:

- `32, 64, 96, 128, 160, 192, 224`

Important result:

- every tested rank stayed safely under the `16,000,000` byte cap
- but none of them recovered the full keep-float gain

Best quality result:

- rank `64`
- artifact `15,109,864`
- `val_bpb = 2.39777665`
- delta vs keep-float: `+0.00028158`
- delta vs plain quantized: `-0.00011398`

Best smallest result:

- rank `32`
- artifact `14,961,896`
- `val_bpb = 2.39782656`
- delta vs keep-float: `+0.00033149`
- delta vs plain quantized: `-0.00006407`

Largest tested rank:

- rank `224`
- artifact `15,849,356`
- still under cap by `150,644`
- `val_bpb = 2.39794626`
- worse than plain quantized by `+0.00005563`

The residual itself was not especially compressible in a low-rank sense:

- residual relative norm was about `0.09419`
- singular-value energy capture was gradual rather than steep

This is the key read:

- the low-rank residual sidecar is a real budget-feasible family
- but this first SVD carrier does not preserve enough of the sacred tensor's semantics
- getting under the cap was easy
- keeping the gain was the hard part

So the update to the search is:

- the problem is no longer "can we find a cheaper whole-tensor carrier at all?"
- it is "can we find a cheaper whole-tensor carrier that preserves structure better than a plain low-rank residual?"

Current recommendation after this sweep:

- do not promote plain SVD residual sidecars as the answer
- keep the family alive only in the broader sense of "whole-tensor structured carrier"
- downrank pure low-rank residuals for this tensor
- next candidates, if this line continues, should preserve local/block structure rather than only global low-rank structure

## 2026-03-23 - mixed global+local residual sidecar on the sacred tensor

After the plain rank-64 residual sidecar became the capped full leader, the next carrier family tested was:

- same sacred tensor: `blocks.0.mlp.proj.weight`
- same global carrier: rank-64 fp16 residual sidecar
- plus a small local correction on the leftover residual
- local carrier shape:
  - column tiles of width `1024`
  - per-tile local ranks `4, 8, 12, 16`

This was implemented offline via:

- `scripts/sweep_mixed_residual_sidecar.py`
- `train_gpt_mlx.py` mixed sidecar dequant support

The quick `128`-sequence full-data slice did show one interesting candidate:

- global rank `64` + local rank `4`
- artifact `15,132,276`
- `val_bpb = 2.39773432`
- improvement vs plain global rank-64 on the slice:
  - `-0.00004233 bpb`
- result artifact:
  - `results/mixed_residual_sidecar_keepf_full_v2_128.json`

Because that was a real slice win under cap, it was promoted immediately to full exact validation on all `62,021,632` validation tokens:

- global-only rank `64` control:
  - artifact `15,109,864`
  - exact `val_bpb = 2.35570158`
- mixed `g64 + tile1024 + local4`:
  - artifact `15,132,276`
  - exact `val_bpb = 2.35570394`
  - delta vs global-only:
    - `+0.00000236 bpb`
    - `+22,412` bytes
- result artifact:
  - `results/mixed_residual_sidecar_rank64_tile1024_local4_fullval.json`

So the full exact read is:

- the slice signal was real enough to justify promotion
- but it did not survive full validation
- naive mixed global+local correction does not beat the plain rank-64 global sidecar

Updated recommendation:

- keep the capped rank-64 global residual sidecar as the live leader
- downrank this first mixed-global-local variant
- if this line continues, the next structured carrier should not be "global low-rank plus uniform column tiles"
- any future local correction should have a fresher reason than this simple leftover-residual tiling

## 2026-03-24 - public-parity gap work: sliding eval, bigram hash, and mixed-bit headroom

The public records are stacking several gain classes that this repo mostly did not have yet:

- sliding-window eval
- cheap token-local bias
- lower-bit mixed precision

The first pass on those classes produced a clearer frontier, not a new winner yet.

### 1. Sliding eval is now implemented as a separate parity track

Both trainers now support:

- `EVAL_MODE=contiguous|sliding`
- `EVAL_STRIDE=<int>`

The sliding implementation scores:

- the full first window
- only the newly exposed suffix on later overlapping windows

This avoids double-counting tokens while giving a public-parity metric path.

Local validation checks:

- contiguous vs sliding with `EVAL_STRIDE=TRAIN_SEQ_LEN` matched to numerical noise
- on the capped rank-64 residual-sidecar winner, a `1024`-sequence slice showed:
  - contiguous `val_bpb = 2.37974998`
  - sliding-64 `val_bpb = 2.37880930`
  - improvement:
    - `-0.00094068`

So sliding eval is real and worth keeping, but as a separate parity track rather than a rewrite of old local history.

### 2. Cheap token-local bigram hash did not earn itself

The first public-style cheap token-local bias tested here was:

- zero-init hashed bigram embedding
- added before the first RMSNorm
- `BIGRAM_HASH_BINS=4096`

Exact smoke result:

- run:
  - `mlx_seq_mlp4x_bigram_hash_v1`
- artifact:
  - `13,448,860` bytes
- exact `val_bpb = 2.61192726`
- smoke control:
  - `2.61172375`
- delta:
  - `+0.00020351`

That is a clean kill for this first bigram-hash implementation.

### 3. Lower-bit mixed precision creates very large byte headroom, but loses badly on the current model

An offline mixed-bit export sweep was run on the promoted sacred-tensor float artifact:

- source float artifact:
  - `logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz`
- tested profile:
  - `*.mlp.fc.weight:5`
  - `*.mlp.proj.weight:5`
  - `*.attn.c_q.weight:6`
  - `*.attn.c_k.weight:6`
  - `*.attn.c_v.weight:6`
  - `*.attn.proj.weight:6`

Results:

- baseline 8-bit export:
  - artifact `14,813,668`
  - exact `val_bpb = 2.35586296`
- mixed profile export:
  - artifact `7,244,516`
  - exact `val_bpb = 2.38623309`
- delta vs 8-bit:
  - bytes `-7,569,152`
  - `val_bpb +0.03037013`

This is the key read:

- mixed bits are not a direct drop-in win on the current trained model
- but they buy enormous artifact headroom
- that means the next serious question is no longer "does mixed precision help by itself?"
- it is "can the saved bytes fund enough additional model capacity to beat the current capped leader after retraining?"

That is a borrowed-GPU question, not a local M4 question.

Updated recommendation:

- keep the current capped leader:
  - `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- keep sliding eval as a separate parity track
- kill the first bigram-hash family
- use any borrowed NVIDIA compute for one or two capacity-funded retraining runs under the mixed-bit export profile


## 2026-03-24 - fp32-factor residual-sidecar sweep on the sacred tensor

A bounded follow-up sweep tested the same global residual-sidecar family as the earlier SVD carrier, but stored the residual factors in `fp32` instead of `fp16`.

Setup:

- source float artifact:
  - `logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz`
- target tensor:
  - `blocks.0.mlp.proj.weight`
- eval slice:
  - `128` sequences from the full published validation prefix
- tested ranks:
  - `32, 48, 64`
- result artifact:
  - `results/fp32_factor_residual_sidecar_keepf_full_v2_128.json`

Reference points on the same slice:

- keep-float sacred tensor:
  - artifact `16,263,292`
  - `val_bpb = 2.39749507`
- plain quant baseline:
  - artifact `14,813,572`
  - `val_bpb = 2.39789063`

fp32-factor results:

- rank `32`:
  - artifact `15,116,588`
  - `val_bpb = 2.39775620`
  - delta vs plain quant:
    - `-0.00013443 bpb`
    - `+303,016` bytes
- rank `48`:
  - artifact `15,268,148`
  - `val_bpb = 2.39779124`
  - delta vs plain quant:
    - `-0.00009939 bpb`
    - `+454,576` bytes
- rank `64`:
  - artifact `15,419,224`
  - `val_bpb = 2.39773304`
  - delta vs plain quant:
    - `-0.00015759 bpb`
    - `+605,652` bytes

Read:

- all tested fp32-factor variants stay under the `16,000,000` byte cap
- rank `64` is the best quality point in this bounded fp32 sweep
- but fp32 factors are materially less byte-efficient than the older fp16 sidecars
- compared with the earlier fp16 rank-64 result (`15,109,864`, `2.39777665`), fp32 rank `64` improves the slice by only about `-0.00004361 bpb` while costing `+309,360` bytes

Updated recommendation:

- keep the fp16 rank-64 residual sidecar as the live capped leader in this family
- treat fp32-factor sidecars as an interesting quality nudge, not a better frontier point
- the next queued task remains the 5090 handoff refresh only if a later result changes the broader conclusion

## 2026-03-25 - scout monitor on active mixed-bit calibration sweep

During the `Golf Scout` automation run at `2026-03-25T11:58:55Z`, the repo was not idle:

- active command:
  - `.venv/bin/python scripts/sweep_mixed_precision_quant_calibration.py --output-json results/mixed_precision_quant_calibration_fullval.json`
- observed elapsed time at the last check:
  - about `9m 10s`
- branch:
  - `codex/scout-loop`
- dirty state:
  - one untracked file: `scripts/sweep_mixed_precision_quant_calibration.py`
  - that file is related to the live run, because the running Python process is executing it

Observed state:

- `results/mixed_precision_quant_calibration_fullval.json` does not exist yet
- that is expected for this script, because it writes the JSON payload only after the baseline and all calibration profiles finish
- the process has MLX / Metal libraries loaded and still has live stdio attached to a terminal, so this looks like an in-progress evaluation rather than a failed launch

Read:

- no new conclusions are available yet from the calibration sweep
- no 5090 handoff refresh is justified while the result is still pending
- once the active calibration run finishes, the automation queue should return to task `1`: the full fp32-factor residual-sidecar sweep for ranks `32,48,64`

## 2026-03-25 - mixed-bit profile calibration around the promoted 5/6 export

The earlier mixed-bit result established that the first local `5/6` profile bought huge byte headroom, but it did not answer a more useful question:

- which side of that profile is doing the real work
- whether the local knee is "give attention more bits" or "stop starving the MLP weights"

To answer that, a bounded full-validation calibration sweep was run on the same promoted sacred-tensor float artifact:

- source float artifact:
  - `logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz`
- output artifact:
  - `results/mixed_precision_quant_calibration_fullval.json`
- fixed baseline:
  - 8-bit export at `14,813,668` bytes and exact `val_bpb = 2.35586296`
- tested neighborhood:
  - `mlp4_attn6`
  - `mlp5_attn5`
  - `mlp5_attn6` (the existing center point)
  - `mlp5_attn7`
  - `mlp6_attn6`

Results:

- `mlp4_attn6`:
  - artifact `4,421,084`
  - exact `val_bpb = 2.63017452`
  - this is a hard collapse; `4`-bit MLP weights are too aggressive here
- `mlp5_attn5`:
  - artifact `6,196,420`
  - exact `val_bpb = 2.39791617`
  - worse than the current `5/6` center by `+0.01168308 bpb`
- `mlp5_attn6`:
  - artifact `7,244,516`
  - exact `val_bpb = 2.38623309`
  - reproduced the earlier single-point sweep exactly
- `mlp5_attn7`:
  - artifact `8,527,564`
  - exact `val_bpb = 2.38371061`
  - improves on `5/6` by `-0.00252248 bpb` for `+1,283,048` bytes
- `mlp6_attn6`:
  - artifact `9,521,536`
  - exact `val_bpb = 2.36480881`
  - improves on `5/6` by `-0.02142428 bpb` for `+2,277,020` bytes
  - still stays `5,588,328` bytes under the current capped leader artifact

Read:

- the dominant local quality lever is MLP precision, not attention precision
- raising attention from `6` to `7` bits helps a little
- raising MLP from `5` to `6` bits helps a lot more
- lowering attention to `5` bits is survivable but clearly worse
- lowering MLP to `4` bits is not a viable trade on this artifact family

Updated recommendation:

- keep the current local capped leader unchanged:
  - `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- if mixed-bit export is used to fund a borrowed-GPU retraining run, promote `mlp6_attn6` to the primary export profile candidate
- keep `mlp5_attn7` as a smaller backup profile if tighter byte pressure matters
- stop treating the earlier `mlp5_attn6` point as the best mixed-bit default
- the next queued task is now the 5090 handoff refresh, because the recommended mixed-bit profile changed materially

## 2026-03-25 - refreshed 5090 handoff package for the calibrated mixed-bit default

The `5090` handoff package was refreshed after the mixed-bit calibration sweep changed the recommendation from the old `mlp5_attn6` center point to `mlp6_attn6`.

Package updates:

- `handoffs/claude_5090_friend/run_parameter_golf_cuda_5090.sh`
  - default `INTX_BITS_BY_NAME` now uses `mlp6_attn6`
- `scripts/run_parameter_golf_cuda_5090.sh`
  - matching repo-side wrapper now uses the same `mlp6_attn6` default to avoid drift
- `handoffs/claude_5090_friend/README.md`
  - now explains the calibrated recommendation and names `mlp5_attn7` as the smaller backup
- `handoffs/claude_5090_friend/PASTE_TO_CLAUDE.md`
  - now tells Claude to keep the primary and secondary CUDA runs on `mlp6_attn6`
- `handoffs/claude_5090_friend/mixed_precision_quant_fullval.json`
  - now carries the calibration summary instead of only the earlier single-point `5/6` evidence

Read:

- the borrowed-GPU package is now aligned with the best offline mixed-bit calibration result we have locally
- the package still points at the same CUDA training hypothesis:
  - use mixed-bit compression headroom to fund extra capacity, not to claim an immediate export-only win
- there are no remaining incomplete items in this bounded scout queue after the handoff refresh

## 2026-03-26 - borrowed compute path marked inactive; switch to strict local-only queue

The borrowed-`5090` path is no longer an active planning item.

Reason:

- the operator path failed, not the handoff artifact
- the handoff package existed and was refreshed
- but the real bottleneck became human coordination and CLI friction
- that is negative expected value compared with continuing bounded local work

So the planning boundary is now:

- no friend-mediated compute
- no handoff maintenance as an active queue item
- no external operator assumptions in the main search loop

Updated read:

- the laptop is not the right machine for a stacked public leaderboard push by itself
- but it is still the right machine for falsification, artifact sweeps, and tooling
- the search should stay local-only until compute is directly controlled by the repo owner

The new active queue is captured in:

- `docs/local-only-active-queue.md`

Top local-only priorities:

1. tighten the mixed-bit quality/bytes knee around `mlp6_attn6`
2. build a capacity budget table for larger candidate architectures instead of guessing
3. keep sliding eval as a separate parity metric, not a new source of multi-hour laptop runs

Dead or inactive branches:

- friend / borrowed-`5090` handoff
- bigram-hash retry
- soft-floor retry
- row-subset rescue
- fp32 sidecar retry
- naive tiled or mixed local residual retry

## 2026-03-26 - scout attempted the full fp32-factor residual-sidecar promotion but MLX is unavailable here

During the `Golf Scout` automation run at `2026-03-26T19:30:10Z`, the explicit queue item from the automation prompt was:

- run the full fp32-factor residual-sidecar sweep for ranks `32,48,64`

This remained incomplete in one important sense:

- the repo already had the bounded `128`-sequence fp32-factor ranking sweep
- but it did not have the promoted full-validation JSON for the same ranks

To make that promotion reproducible again, `scripts/sweep_residual_sidecar.py` was updated to restore an explicit:

- `--sidecar-dtype {float16,float32}`

and to record the chosen sidecar dtype in the output JSON payload.

Intended promotion command:

- `.venv/bin/python scripts/sweep_residual_sidecar.py --sidecar-dtype float32 --ranks 32,48,64 --val-seqs 60568 --output-json results/fp32_factor_residual_sidecar_keepf_full_v2_fullval.json`

Observed blocker:

- even the narrower probe
  - `.venv/bin/python -c 'import mlx.core as mx; print(mx.default_device())'`
  aborts immediately in this automation context
- the failure is an `NSRangeException` inside MLX Metal device initialization
- it happens before argument parsing or any evaluation work

So the honest outcome of this run is:

- no full-validation fp32-factor JSON was produced
- no new quality conclusion was earned
- the blocker is execution environment, not an unresolved repo bug in the sweep logic itself

Current recommendation:

- treat the full fp32-factor promotion as still pending
- run it only from a Metal-capable local session where `import mlx.core` succeeds
- keep the next queued task unchanged until that exact run actually lands

## 2026-03-27 - scout rechecked the fp32-factor queue head and the MLX blocker still holds

During the `Golf Scout` automation run at `2026-03-27T07:42:36Z`, the repo was first checked for two automation guardrails:

- no long-running parameter-golf experiment was active
- the git worktree was clean on `codex/scout-loop`

So the scout took the first incomplete automation queue item again:

- run the fp32-factor residual-sidecar full-validation sweep for ranks `32,48,64`

The intended command is still:

- `.venv/bin/python scripts/sweep_residual_sidecar.py --sidecar-dtype float32 --ranks 32,48,64 --val-seqs 60568 --output-json results/fp32_factor_residual_sidecar_keepf_full_v2_fullval.json`

But before starting a multi-hour eval, the narrow environment probe was repeated in the repo venv:

- `.venv/bin/python -c 'import mlx.core as mx; print(mx.default_device())'`

Observed result:

- the process still aborts immediately with the same `NSRangeException`
- the crash still occurs inside MLX Metal device initialization
- the failure still happens before any sweep argument parsing or evaluation work

So the honest outcome of this automation cycle is unchanged:

- no new JSON result was produced under `results/`
- no new quality or bytes conclusion was earned
- the blocker is still the local execution environment, not the sweep script contract

Current recommendation:

- keep the fp32-factor full-validation sweep as the next queued automation task
- only reattempt it from a session where the repo venv can successfully import `mlx.core`
- do not advance to the mixed-bit recalibration or handoff-refresh queue items until this head task either lands or is explicitly deprioritized
