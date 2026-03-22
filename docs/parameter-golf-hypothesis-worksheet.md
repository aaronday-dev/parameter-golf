---
title: "Parameter Golf Hypothesis Worksheet"
date: "2026-03-22"
documentclass: extarticle
geometry: margin=0.5in
classoption:
  - 9pt
header-includes:
  - \usepackage{array}
  - \usepackage{longtable}
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{0.7ex}{0.4ex}
  - \titlespacing*{\subsection}{0pt}{0.5ex}{0.3ex}
  - \setlength{\parskip}{2pt}
  - \setlength{\parindent}{0pt}
  - \setlength{\tabcolsep}{4pt}
---

# Parameter Golf Hypothesis Worksheet

Use this when the next run needs a new idea, not another nearby tweak.

## What These Words Mean

- **Parameter Golf**: the search for a small model that still scores well after compression under a hard artifact-size limit.
- **Current winner**: the best full local run so far.
- **Smoke baseline**: the cheap, short run used as the local reference point.
- **Latest miss**: the most recent run that failed to beat the smoke baseline.
- **Exact `val_bpb`**: the final score after the model has been re-encoded the same way the challenge scores it.
- **Locus**: the exact place where the problem seems to enter. Use one of:
  - **input**: the problem starts in the tokens or features going in
  - **state**: the problem is in the model's internal representation while it is thinking
  - **boundary**: the problem appears at edge cases, transitions, or ambiguous positions
  - **quantization**: the problem appears when the trained model is compressed / re-encoded
- **Patch**: the smallest code change that tests the idea.
- **Smoke**: one quick run used to prove or kill the idea.
- **Sweep**: many nearby runs. Do not do this first.

## State Of Play

Current winner:
- `mlx_full_seq_mlp4x_200_realval_vb524k`
- exact `val_bpb = 2.35796063`

Smoke baseline:
- `mlx_seq_mlp4x_lzma_cmp_v2`
- exact `val_bpb = 2.61172375`

Latest miss:
- `mlx_seq_dim528_mlp4x_lzma_cmp_v1`
- exact `val_bpb = 2.61440332`

## Why The Current Read Is What It Is

| Claim | Why we currently believe it |
|---|---|
| `MLP_MULT` helped | Sequential `MLP_MULT=4` beat sequential `MLP_MULT=3` on smoke (`2.6117` vs `2.6172`) and on the promoted run (`2.3580` vs `2.3733`). |
| Width missed | `MODEL_DIM=528, MLP_MULT=4` lost to the `MODEL_DIM=512` smoke baseline (`2.6144` vs `2.6117`). |
| Depth missed | More mirrored depth regressed badly (`2.6769`) instead of helping. |
| Plain sequential beat shared-core cleverness | The best promoted sequential run (`2.3580`) beat the best promoted shared-core run (`2.3813`). |
| Damping / attractor controls are downranked | They repeatedly made the exact post-compression score worse, even when they looked cleaner internally. |

Short honest read:

- Coarse capacity helped.
- Nearby width and depth pokes did not.
- Shared-core recurrence is no longer the lead story on this local track.
- The open space is not "more of the same." It is a different mechanism.

## Decision Card

### 1. Name The Problem

Recent miss I am thinking from:

What changed:

Where it showed up:

Why it mattered:

### 2. One Hypothesis

Name:

Exact locus:

Exact failure mode:

Why this is plausible *now*:

What result would disprove it immediately:

### 3. Minimal Patch

What is the smallest code change that tests this?

Files / functions it touches:

What must stay unchanged:

### 4. Smoke

Smoke run id:

What counts as a win:

What exact result kills the idea:

### 5. Decision

If it wins, the next single step is:

If it loses, I stop doing:

## Quick Reality Check

Answer `yes` or `no`:

- Can I point to the exact file or function?
- Does this change where the model helps, not just how strongly it pushes?
- Does the idea still make sense after exact re-encoding?
- Can one smoke run prove or kill it?
- Am I testing a mechanism, not just a knob?

If any answer is `no`, do not run it yet.

## Example

Name:
- boundary-aware side signal on the sequential winner

Exact locus:
- boundary positions in the late correction path

Exact failure mode:
- rare or ambiguous positions are being over-smoothed by a correction that is too global

Why this is plausible now:
- the current local line improved with more useful capacity, but global corrective ideas mostly failed; public winners are also using edge-case-aware signals such as bigram-side features

Minimal patch:
- add one small boundary-conditioned gate at the sensitive late locus in `train_gpt_mlx.py`

What must stay unchanged:
- width, depth, serializer path, and unrelated trainer plumbing

What counts as a win:
- exact smoke beats `2.61172375`

What kills it:
- exact smoke does not beat baseline on the first run
