# Parameter Golf Hypothesis Worksheet

Last updated: 2026-03-22

This worksheet exists to stop local search from dissolving into nearby knob-twiddling.

Use it when the current family has flattened and the next run needs a new mechanism, not another minor variation.

## Current Baseline

Current best promoted local result:

- `mlx_full_seq_mlp4x_200_realval_vb524k`
- exact `val_bpb = 2.35796063`
- same winning quantized payload fits under the artifact cap when re-encoded with `lzma`

Current smoke baseline for this family:

- `mlx_seq_mlp4x_lzma_cmp_v2`
- exact `val_bpb = 2.61172375`

Latest near-neighbor miss:

- `mlx_seq_dim528_mlp4x_lzma_cmp_v1`
- exact `val_bpb = 2.61440332`

Interpretation:

- `MLP_MULT` helped
- extra width missed
- extra depth missed
- shared-core cleverness lost to plain sequential capacity
- storage/container work is no longer the bottleneck

## Dead / Alive / Unclear

### Dead or strongly downranked

- Blunt damping, stabilization, and attractor pulls
  Why: they often made the exact post-quantized score worse even when they looked cleaner internally.
- Extra mirrored/shared-core recurrence as the main local story
  Why: plain sequential capacity beat the best shared-core branch.
- Width-first local extensions of the current sequential winner
  Why: `MODEL_DIM=528` on `MLP_MULT=4` missed.
- Extra depth on the current sequential line
  Why: `10` layers regressed.

### Alive

- Plain sequential capacity
  Why: `MLP_MULT=3` and then `MLP_MULT=4` both improved promoted results.
- Quantization-sensitive handling of specific tensors or loci
  Why: the public records show this is high leverage, and we have not really attacked it directly.
- Boundary-aware or side-channel structure
  Why: public winners are using `BigramHash`, `SmearGate`, sliding evaluation, and similar nontrivial structure we have not tried.

### Unclear but worth sketching

- Salience-aware late correction instead of global correction
- Localized QAT on a measured sensitive locus
- Boundary-aware auxiliary features on the plain sequential winner

## 45-Minute Paper Drill

Goal:

- leave with one hypothesis family
- one minimal code change
- one kill criterion

Do not leave with five ideas. Leave with one run-worthy idea or none.

### Block 1: Failure Map (15 min)

Write down three memorable misses from this search.

For each miss, label it as one of:

- `masked detail`
- `threshold miss`
- `regime switch`

Then answer:

- what was being overcorrected?
- what was being blurred that probably mattered?
- did the miss look like global overreach or wrong placement?

### Block 2: Lever Classification (15 min)

List every lever that still feels meaningful.

Force each into exactly one bucket:

- `gain`
- `threshold`
- `time constant`
- `asymmetry`

If a lever does not fit one of those, it is probably not the real mechanism.

Then write:

- which lever class has actually paid off locally?
- which lever class have we mostly abused?

Expected answer from the current evidence:

- local wins came from coarse capacity allocation
- local losses often came from global corrective gain or recurrence stories without enough specialization

### Block 3: Salience Sketch (15 min)

Draw a simple 2x2:

- x-axis: `local salience`
- y-axis: `model uncertainty`

Now mark:

- easy/common cases
- rare/boundary cases
- places where quantization probably destroys distinctions first

Then answer:

- where should the model spend extra bits or extra corrective structure?
- where should it explicitly do less?
- what should be preserved structurally even if fine detail dies?

## Patterns To Look For

Good signs:

- the hypothesis changes *where* correction or extra structure appears, not only how much
- the idea protects high-salience or boundary cases without globally blunting the state
- the mechanism still makes sense after quantization
- the mechanism can be tested with one narrow patch and one smoke

Bad signs:

- it is another scalar gain story
- it depends on fine float-space geometry surviving exact int8 roundtrip
- it asks for many coupled knob sweeps before yielding a falsifiable claim
- it is mainly a story about recurrence even though plain sequential capacity currently wins

## One-Slot Hypothesis Template

Fill this out before touching code.

### 1. Hypothesis Family

Name:

One sentence:

What qualitative mechanism changes?

### 2. Why It Could Move By Hundredths

Why this is not a thousandth-shaving tweak:

What current evidence supports trying it:

What public evidence supports trying it:

### 3. Minimal Patch

Target file:

Target code path:

Minimal implementation:

What I am explicitly **not** changing:

### 4. Measurement

Smoke run id:

Primary metric:

Secondary metric:

Expected win signal:

### 5. Kill Criterion

Kill immediately if:

Do not promote if:

### 6. Next Decision

If smoke wins:

If smoke loses:

## Best Current Hypothesis Directions

These are the best next families to sketch, not commitments to run.

1. Boundary-aware side channel on the sequential winner
   Examples: bigram-adjacent feature path, smear-like boundary signal, salience-conditioned late gate.

2. Quantization-sensitive tensor treatment
   Examples: fp16 tied embedding/head passthrough, localized QAT on the sensitive locus.

3. Salience-aware soft-knee correction
   Only if it is clearly boundary-local and not another global damping story.

## Current Non-Goals

Do not spend the next run on:

- another width poke near `MODEL_DIM=512`
- another depth poke
- another attractor / stabilization / damping variant
- another shared-core schedule micro-variation
- more serializer/container work

## Exit Condition

If you cannot explain the new family in plain language and state a one-run kill criterion, do not run it.
