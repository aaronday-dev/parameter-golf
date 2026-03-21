# Parameter Golf

This repository contains a standalone Parameter Golf workspace built from the
local experiment branch.

The main focus is simple:

- train small language models under the challenge constraints
- measure exact post-quantization `val_bpb`
- compare architectural changes against compressed artifact size
- keep the search legible through logs and notes

## Current Best Verified Local Result

Best exact real-data result currently in this repo:

- run: `mlx_full_seq_mlp3x_200_realval_vb524k`
- exact `final_int8_zlib_roundtrip_exact val_bpb = 2.37334218`
- compressed artifact size: `13,534,421` bytes
- hardware: Apple Silicon M4 via MLX / Metal

Earlier milestones:

- shared-core mirror + directional correction:
  `2.38989686`
- shared-core mirror + directional correction + `MLP_MULT=3`:
  `2.38131855`

Current local conclusion:

- increasing useful capacity helped
- extra recurrence and contraction-style controls mostly did not
- the best local result now comes from a plain sequential `MLP_MULT=3` model

## What This Repo Contains

- `train_gpt.py`
  Torch/CUDA trainer.
- `train_gpt_mlx.py`
  Apple Silicon / MLX trainer used for local search.
- `scripts/`
  Run wrappers and small analysis helpers.
- `docs/`
  Research log, run notes, and archived upstream challenge material.
- `results/`
  Selected run logs.
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

For Apple Silicon MLX runs:

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
  Early shared-core promoted winner.
- `results/mlx_full_mirror_mlp3x_dirc02_200_realval_vb524k.txt`
  Shared-core `MLP_MULT=3` promoted result.
- `results/mlx_full_seq_mlp3x_200_realval_vb524k.txt`
  Current best local promoted result.
- `results/mlx_mirror13_dirc02_cmp.txt`
  Negative result: more mirrored recurrence alone regressed.

## Data

Datasets are not vendored here.

This repo assumes the same dataset and tokenizer layout used by the upstream
OpenAI Parameter Golf codebase. See:

- `data/README.md`
- `docs/upstream-openai-readme.md`

## Research Thread

The main running log lives in:

- `docs/parameter-golf-research-log.md`

Current practical takeaway:

1. Shared-core mirror scheduling produced a measurable improvement over the earlier baseline.
2. Blunt contraction / damping / attractor-style controls mostly failed.
3. Spending more of the artifact budget on useful capacity produced the biggest
   recent gains.
4. The best local result currently comes from a plain higher-capacity sequential
   model that still fits under the `16 MB` artifact cap.
