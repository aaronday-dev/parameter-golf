---
title: "Parameter Golf Hypothesis Worksheet"
date: "2026-03-22"
documentclass: extarticle
geometry: margin=0.4in
classoption:
  - 8pt
header-includes:
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{0.5ex}{0.4ex}
  - \titlespacing*{\subsection}{0pt}{0.4ex}{0.3ex}
  - \setlength{\parskip}{2pt}
  - \setlength{\parindent}{0pt}
  - \setlength{\tabcolsep}{4pt}
---

# Parameter Golf Hypothesis Worksheet

Use this only when the current line has plateaued and the next run needs a new mechanism.

## Header

Current winner:
- `mlx_full_seq_mlp4x_200_realval_vb524k`
- exact `val_bpb = 2.35796063`

Current smoke baseline:
- `mlx_seq_mlp4x_lzma_cmp_v2`
- exact `val_bpb = 2.61172375`

Latest miss:
- `mlx_seq_dim528_mlp4x_lzma_cmp_v1`
- exact `val_bpb = 2.61440332`

One-line read:
- `MLP_MULT` helped; width missed; depth missed; shared-core cleverness lost to plain sequential capacity.

## What Changed

- Helped: coarse capacity on the plain sequential winner
- Failed: width, depth, damping / attractor controls, more shared-core recurrence
- Still open: signals that help on edge cases; some parts survive re-encoding better than others

## 45-Minute Flow

### 0-5 min: Name the miss

Pick one recent bad or flat run.
Write one sentence: what changed, where it showed up, why it mattered.

### 5-15 min: Write three causes

For each cause, answer: what part is likely wrong, and what would I expect to see if I’m right?

Use this quick table:

| Miss | What changed | What likely got overcorrected | What to test next |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |

### 15-25 min: Point to the locus

For the best cause, name the exact entry point:
- `input`
- `state`
- `boundary`
- `quantization`

If you cannot point to the file or code path, the idea is not ready.

### 25-35 min: Define one patch

Write the smallest change that tests the idea: one patch, one smoke, no sweep.

### 35-45 min: Decide

If it wins, write one next step. If it loses, kill it in one sentence.

## One Hypothesis

Name:

Exact locus + exact failure mode:

One sentence:

Minimal patch:

Do **not** change: width, depth, serializer path, or anything unrelated.

Smoke run id:

Expected win signal:

Kill immediately if:

Next step if it wins:

## Yes / No Check

Before you code, answer `yes` or `no`:

- Does this change *where* the model helps, not just how hard it pushes?
- Does it still make sense after exact int8 re-encoding?
- Can it be tested with one narrow patch?
- Can you name the exact file and code path?
- Can you kill it after one smoke if it misses?

If any answer is `no`, do not run it.

## Example Stub

- Name: boundary-aware side channel on the sequential winner
- Exact locus + failure mode: late correction is too global; rare or ambiguous positions need local help instead
- Minimal patch: add one small salience-conditioned gate at the sensitive late locus in `train_gpt_mlx.py`
- Kill immediately if: exact smoke does not beat `2.61172375`
