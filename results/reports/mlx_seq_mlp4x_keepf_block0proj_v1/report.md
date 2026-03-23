# Parameter Golf Report: mlx_seq_mlp4x_keepf_block0proj_v1

Normalized single-run view for architecture, compression, and post-roundtrip evaluation.

## Summary
- Exact post-roundtrip val_bpb is 2.61001513 on a lzma roundtrip.
- The archived log reports a lzma artifact at 14,858,772 bytes.
- The compressed artifact is under the decimal 16,000,000-byte cap by 1,141,228 bytes.
- Quantized roundtrip changed val_bpb by +0.00031513 relative to the final in-run validation.
- The run is a sequential architecture with 9 layers, dim 512, and schedule `cyclic`.
- This is a smoke-scale result over 1,047,552 validation tokens.

## Run
- Run id: `mlx_seq_mlp4x_keepf_block0proj_v1`
- Source log: `/Users/aaronday/dev/parameter-golf/results/mlx_seq_mlp4x_keepf_block0proj_v1.txt`
- Validation tokens: `1,047,552`
- Eval scope: `smoke`

## Model
- Params: `26,497,096`
- Topology: `9` layers, dim `512`, heads `8`, kv_heads `4`
- Sequence length: `1024`
- Tie embeddings: `True`
- Architecture family: `sequential`
- Shared-core schedule: `cyclic`
- Unique blocks: `9`

## Training
- Iterations: `80`
- Train batch tokens: `8,192`
- Grad accum steps: `8`
- Microbatch tokens: `1,024`
- Validation batch tokens: `524,288`
- Warmup steps: `0`
- Max wallclock seconds: `600.000`
- Final train time: `47,398` ms

## Compression
- Storage compressor: `lzma`
- Compressed model bytes: `14,858,772`
- Payload bytes: `27,682,080`
- Raw serialized bytes: `27,691,501`
- Payload ratio: `3.79`x
- Budget status: `within_budget`
- Budget margin: `1,141,228` bytes
- Storage source: `log`

## Evaluation
- Roundtrip compressor: `lzma`
- Final in-run val_bpb: `2.60970000`
- Exact roundtrip val_bpb: `2.61001513`
- Roundtrip penalty: `0.00031513`
- Roundtrip eval time: `13,406` ms

## Derived
- Compressed bytes per parameter: `0.560770`
- Budget margin percent: `7.133`%
- Savings vs saved model: `85.844`%

## Source Artifacts
- `/Users/aaronday/dev/parameter-golf/results/mlx_seq_mlp4x_keepf_block0proj_v1.txt`
