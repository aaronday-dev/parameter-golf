# Public Summary

This repository focuses on artifact-aware compression and evaluation for small language models under the Parameter Golf `16,000,000` byte cap. The core metric is exact post-quantization `val_bpb`, measured after full artifact roundtrip rather than only in float space.

## Current Best Capped Result

- Run: `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- Exact `val_bpb`: `2.35570158`
- Artifact bytes: `15,109,864`
- Method: offline rank-64 fp16 residual sidecar on a quantization-sensitive early MLP projection tensor, `blocks.0.mlp.proj.weight`

## Over-Budget Source Result

- Run: `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2`
- Exact `val_bpb`: `2.35551193`
- Artifact bytes: `16,263,292`

This source result validated that generic quantization was damaging this tensor disproportionately, but it missed the hard cap. The capped residual-sidecar result preserves most of that gain while staying within budget.

## What Was Learned

- Exact artifact evaluation matters; several attractive branches lost once evaluated after full roundtrip.
- Generic quantization is not uniformly good; an early MLP projection tensor is unusually sensitive.
- Mixed precision is a real local lever, and MLP precision matters more than attention precision on this artifact family.
- The current mixed-precision recommendation is not a near-win by itself; it is a calibrated tradeoff that creates headroom for larger retrains.

## Mixed-Precision Calibration

| Profile | Exact `val_bpb` | Artifact bytes |
|---|---:|---:|
| 8-bit baseline | `2.35586296` | `14,813,668` |
| over-budget keep-float source | `2.35551193` | `16,263,292` |
| capped residual-sidecar leader | `2.35570158` | `15,109,864` |
| `mlp5_attn6` | `2.38623309` | `7,244,516` |
| `mlp5_attn7` | `2.38371061` | `8,527,564` |
| `mlp6_attn6` | `2.36480881` | `9,521,536` |

## Next Compute-Backed Question

Can `mlp6_attn6` fund a larger `10L` or `11L` `MLP3x` retrain that beats `2.35570158` under exact roundtrip evaluation?

## Primary Evidence

- `results/reports/mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1/report.md`
- `results/residual_sidecar_rank64_fullval.json`
- `results/mixed_precision_quant_calibration_fullval.json`
