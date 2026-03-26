# Local-Only Active Queue

Last updated: 2026-03-26

## Intent

This queue assumes:

- no borrowed GPU
- no friend-mediated CLI
- no external operator in the loop

The laptop is a falsification and tooling machine, not the finish line for a
public leaderboard push.

Current capped local leader:

- `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- exact `val_bpb = 2.35570158`
- artifact `15,109,864`

## Inactive Branches

These are no longer active planning items:

- friend / borrowed-`5090` handoff
- bigram-hash retry
- soft-floor retry
- row-subset sacred-tensor rescue
- fp32 sidecar retry
- naive tiled or mixed local residual retry

They remain in the repo as history, not as current queue items.

## What The Laptop Is Good For

- offline artifact sweeps
- serializer / quantizer experiments
- exact evaluation on saved artifacts
- small smoke training runs with hard kill criteria
- keeping the research log legible

## Active Queue

### 1. Tighten the mixed-bit local frontier offline

Goal:

- keep mapping the local quality/bytes knee around the current best mixed-bit neighborhood

Current read:

- `mlp6_attn6` is the best local mixed-bit default seen so far
- attention precision matters less than MLP precision on this artifact family

Allowed work:

- offline calibration sweeps near `mlp6_attn6`
- exact eval only
- no retraining required

Kill rule:

- stop if a new point is clearly dominated on both bytes and `val_bpb`

### 2. Build a capacity budget table instead of guessing

Goal:

- estimate which larger architectures could plausibly fit under mixed-bit export without needing external compute first

Deliverable:

- one machine-readable table for candidate shapes such as:
  - `10L MLP3x`
  - `11L MLP3x`
  - nearby width/depth variants
- include predicted artifact bytes under:
  - plain 8-bit
  - `mlp6_attn6`
  - one smaller backup profile

Kill rule:

- do not open a training branch unless the table suggests a realistic under-cap path

### 3. Keep sliding eval as a parity metric, not a distraction

Goal:

- preserve sliding eval as a separate measurement track

Allowed work:

- small slice checks
- parity sanity checks
- report formatting

Not allowed:

- multi-hour full-val sliding runs on the laptop unless there is a new reason

### 4. Keep the scout bounded

The scout should do only:

- one task per cycle
- local-only tasks
- no handoff refresh work
- no new architecture wandering without a written reason

Preferred cycle order:

1. mixed-bit calibration
2. capacity budget table
3. parity metric maintenance

## Aaron's Role

Highest-value work:

- choose the queue
- kill dead classes hard
- notice asymmetries and new loci
- keep the search narrative coherent

Low-value work:

- babysitting loops
- recruiting reluctant operators
- maintaining handoff packets for dead compute paths

## Definition Of Progress

Local progress means one of:

- a better offline quality/bytes knee
- a sharper budget model for larger architectures
- a cleaner kill of a tempting but dead branch
- a more legible research trail that compounds future work

It does **not** require daily leaderboard movement.
