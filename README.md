# Parameter Golf

This repository packages Aaron's active work on OpenAI's Parameter Golf
challenge into a standalone project that can be shared directly for compute
review, collaboration, and submission prep.

The project is not framed as generic LLM tuning. The working lens is:

- severe post-training quantization as the real constraint
- small-model architecture as a dynamical system
- perceptual-coding style intuition about what survives lossy channels
- careful separation between "interesting float behavior" and "behavior that
  survives exact int8 roundtrip evaluation"

## Current Best Verified Local Result

Best exact real-data result currently in this repo:

- run: `mlx_full_seq_mlp3x_200_realval_vb524k`
- exact `final_int8_zlib_roundtrip_exact val_bpb = 2.37334218`
- compressed artifact size: `13,534,421` bytes
- hardware: Apple Silicon M4 via MLX / Metal

Important earlier milestones:

- shared-core mirror + directional correction:
  `2.38989686`
- shared-core mirror + directional correction + `MLP_MULT=3`:
  `2.38131855`

The current evidence says straightforward capacity spending is now beating the
more elaborate shared-core control family.

## What This Repo Contains

- `train_gpt.py`
  Torch/CUDA trainer with the semantic fixes from the local verification pass.
- `train_gpt_mlx.py`
  Apple Silicon / MLX trainer used for the local search loop.
- `scripts/`
  M4-safe run wrappers and small analysis helpers.
- `docs/`
  Research log, next-run notes, and archived upstream challenge material.
- `results/`
  Selected run logs showing the progression from shared-core to the current
  sequential-capacity leader.
- `data/README.md`
  Expected local dataset/tokenizer layout.

## Quick Start

Create a local environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then place the tokenizer and dataset shards under `./data/` using the layout in:

- `data/README.md`

For Apple Silicon MLX runs on this M4:

```bash
scripts/run_parameter_golf_mlx_m4.sh
```

For smoke mode:

```bash
RUN_MODE=smoke scripts/run_parameter_golf_mlx_m4.sh
```

For the longer local promotion profile:

```bash
RUN_MODE=promotion scripts/run_parameter_golf_mlx_m4.sh
```

## Included Experimental Highlights

- `results/mlx_full_mirror_dirc02_200_realval.txt`
  Early promoted shared-core winner.
- `results/mlx_full_mirror_mlp3x_dirc02_200_realval_vb524k.txt`
  Shared-core capacity upgrade that survived promotion.
- `results/mlx_full_seq_mlp3x_200_realval_vb524k.txt`
  Current best local promoted result.
- `results/mlx_mirror13_dirc02_cmp.txt`
  Negative result showing that more mirrored recurrence alone was not the right
  lever.

## Data

Datasets are not vendored here.

This repo assumes the same dataset and tokenizer layout used by the upstream
OpenAI Parameter Golf codebase. See:

- `data/README.md`
- `docs/upstream-openai-readme.md`

## Research Thread

The main narrative lives in:

- `docs/parameter-golf-research-log.md`

The current practical takeaway is:

1. Shared-core mirror scheduling found a real mechanism.
2. Blunt contraction / damping / attractor-style controls mostly failed.
3. Spending more of the artifact budget on useful capacity produced the biggest
   recent gains.
4. The live frontier has shifted toward plain high-capacity sequential models
   that still fit under the `16 MB` artifact cap.
