# Parameter Golf Report: mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1

Normalized offline artifact-eval view for a derived capped result.

## Summary
- Exact offline post-roundtrip val_bpb is 2.35570158 on a derived lzma artifact.
- The derived artifact is under the decimal 16,000,000-byte cap by 890,136 bytes.
- This result was derived offline from `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2` using a rank-64 fp16 residual sidecar on `blocks.0.mlp.proj.weight`.
- Relative to the over-budget keep-float source result, it gives back +0.00018965 bpb.
- Relative to the previous capped local leader `mlx_full_seq_mlp4x_200_realval_vb524k`, it changes exact val_bpb by -0.00225905.
- This is a full-eval result over 62,021,632 validation tokens.

## Run
- Run id: `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- Source log: `/Users/aaronday/dev/parameter-golf/results/mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1.txt`
- Validation tokens: `62,021,632`
- Eval scope: `full`

## Model
- Params: `26,497,096`
- Topology: `9` layers, dim `512`, heads `8`, kv_heads `4`
- Sequence length: `1024`
- Tie embeddings: `True`
- Architecture family: `sequential`
- Shared-core schedule: `cyclic`
- Unique blocks: `9`

## Training
- Iterations: `200`
- Train batch tokens: `8,192`
- Grad accum steps: `8`
- Microbatch tokens: `1,024`
- Validation batch tokens: `131,072`
- Warmup steps: `0`
- Max wallclock seconds: `600.000`
- Final train time: `116,685` ms

## Compression
- Storage compressor: `lzma`
- Compressed model bytes: `15,109,864`
- Payload bytes: `26,962,208`
- Raw serialized bytes: `n/a`
- Payload ratio: `n/a`x
- Budget status: `within_budget`
- Budget margin: `890,136` bytes
- Storage source: `derived_json`
- Logged artifact: `lzma` at `15,109,864` bytes

## Evaluation
- Roundtrip compressor: `lzma`
- Final in-run val_bpb: `n/a`
- Exact roundtrip val_bpb: `2.35570158`
- Roundtrip penalty: `n/a`
- Roundtrip eval time: `n/a` ms

## Derived
- Compressed bytes per parameter: `0.570246`
- Budget margin percent: `5.563`%
- Savings vs saved model: `84.505`%

## Source Artifacts
- `/Users/aaronday/dev/parameter-golf/results/mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1.txt`
- `/Users/aaronday/dev/parameter-golf/results/residual_sidecar_rank64_fullval.json`
- `/Users/aaronday/dev/parameter-golf/logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2.txt`
- `/Users/aaronday/dev/parameter-golf/logs/mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2_mlx_model.npz`
