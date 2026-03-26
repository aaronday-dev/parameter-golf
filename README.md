# Parameter Golf

This repository is a standalone Parameter Golf workspace focused on artifact-aware compression and evaluation under the `16,000,000` byte cap.

The main goals are:

- train small language models under the challenge constraints
- measure exact post-quantization `val_bpb`
- compare model quality against compressed artifact size
- keep the search legible through targeted tooling and result summaries

## Current Best Capped Result

- Run: `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- Exact `val_bpb`: `2.35570158`
- Compressed artifact size: `15,109,864` bytes
- Method: offline rank-64 residual sidecar on a quantization-sensitive early MLP projection tensor, `blocks.0.mlp.proj.weight`

This result is documented in:

- `docs/public-summary.md`
- `results/reports/mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1/report.md`
- `results/residual_sidecar_rank64_fullval.json`

## Mixed-Precision Calibration

The current calibrated mixed-precision branch is `mlp6_attn6`.

- `mlp5_attn6`: `2.38623309` at `7,244,516` bytes
- `mlp5_attn7`: `2.38371061` at `8,527,564` bytes
- `mlp6_attn6`: `2.36480881` at `9,521,536` bytes

The calibration evidence lives in:

- `results/mixed_precision_quant_calibration_fullval.json`

## Why This Matters

The repo’s main finding is not a new architecture family. It is that exact artifact-aware evaluation changes which ideas survive:

- generic quantization is not uniformly good
- an early MLP projection tensor is unusually sensitive
- calibrated mixed precision creates a plausible compression-funded capacity branch

The next compute-backed question is:

- can `mlp6_attn6` fund a larger `10L` or `11L` `MLP3x` retrain that beats `2.35570158` under exact roundtrip evaluation?

## Key Code Paths

- `train_gpt.py`
  Torch/CUDA trainer with sliding-eval and mixed-bit quantization support.
- `train_gpt_mlx.py`
  Apple Silicon / MLX trainer used for the local artifact-aware search loop.
- `scripts/sweep_mixed_precision_quant.py`
  Offline mixed-precision artifact sweep.
- `scripts/sweep_mixed_precision_quant_calibration.py`
  Mixed-precision calibration sweep centered on the current best export profile.

## Quick Start

Create a local environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install mlx
```

Bootstrap the smoke tokenizer and dataset prefix used by the local MLX wrappers:

```bash
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1 --build-smoke
```

Place larger dataset prefixes under `./data/` using:

- `data/README.md`

For Apple Silicon MLX runs:

```bash
scripts/run_parameter_golf_mlx_m4.sh
```

For smoke mode:

```bash
RUN_MODE=smoke scripts/run_parameter_golf_mlx_m4.sh
```

## Additional Context

Supporting historical notes remain in:

- `docs/parameter-golf-research-log.md`

That log is intentionally secondary. The public summary and linked result artifacts above are the primary entry points for this branch.
