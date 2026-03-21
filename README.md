# Parameter Golf Lab

This repository packages Aaron's active work on OpenAI's Parameter Golf
challenge into a standalone Git-ready project.

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

## Why This Repo Exists

The original work lived inside a broader local workspace. This repo isolates the
Parameter Golf code, wrappers, logs, and research notes so it can be:

- published cleanly on Git
- shared for compute-grant review
- extended without dragging unrelated workspace files around

## Structure

- `parameter-golf/`
  Derived training code and upstream README/license material.
- `scripts/`
  Run wrappers and small analysis tools.
- `docs/`
  Research log, next-run plan, and conceptual notes.
- `results/`
  Selected logs that show the progression of the search.

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

## Running On An M4

Use the M4-safe wrapper:

```bash
scripts/run_parameter_golf_mlx_m4.sh
```

Modes:

- `RUN_MODE=smoke`
- `RUN_MODE=promotion`

The wrapper keeps validation conservative enough to reduce whole-machine crashes
on larger MLX runs while still preserving exact int8 roundtrip checks.

## Data

Datasets are not vendored here.

This repo assumes the same dataset and tokenizer layout used by the upstream
OpenAI Parameter Golf codebase. See:

- `parameter-golf/data/README.md`

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
