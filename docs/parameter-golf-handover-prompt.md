# Parameter Golf Handover Prompt

Use this prompt to continue the work in `/Users/aaronday/dev/parameter-golf`.

```text
We are continuing work in the standalone Parameter Golf repo:

/Users/aaronday/dev/parameter-golf

Treat that repo as the only source of truth.
Do not use or refer back to the old discrete-mathematics repo for Parameter Golf work unless I explicitly ask.

Current state:

- Best full local result:
  `mlx_full_seq_mlp4x_200_realval_vb524k`
  exact `val_bpb = 2.35796063`

- Current smoke baseline:
  `mlx_seq_mlp4x_lzma_cmp_v2`
  exact `val_bpb = 2.61172375`

- Latest nearby miss:
  `mlx_seq_dim528_mlp4x_lzma_cmp_v1`
  exact `val_bpb = 2.61440332`

Current local read:

- increasing `MLP_MULT` helped
- nearby width increases did not help
- extra depth did not help
- shared-core / recurrence tricks were locally beaten by a plain sequential higher-capacity model
- damping / attractor / stabilization controls mostly made the exact post-compression score worse
- `lzma` artifact export is solved enough and is no longer the active bottleneck

Important docs in this repo:

- `README.md`
- `docs/parameter-golf-research-log.md`
- `docs/parameter-golf-hypothesis-worksheet.md`
- `docs/parameter-golf-hypothesis-worksheet.pdf`
- `docs/parameter-golf-claude-hypothesis-prompt.md`
- `results/mlx_full_seq_mlp4x_200_realval_vb524k.txt`
- `results/mlx_seq_mlp4x_lzma_cmp_v2.txt`

Working assumptions:

- local M4 search is now in diminishing returns on the current family
- the next worthwhile move should be a new mechanism, not another nearby width/depth tweak
- likely live families still worth thinking about are:
  - boundary-aware side signals
  - salience-conditioned late correction
  - quantization-sensitive treatment of a specific tensor or path

Rules for continuing:

1. Stay in `/Users/aaronday/dev/parameter-golf`
2. Prefer one mechanism, one locus, one patch, one smoke, one kill criterion
3. Do not reopen dead local families without a fresh reason
4. Keep responses human-readable and concrete
5. If suggesting a run, tie it back to the current baseline and exact metric

Start by:

1. reading the README and research log
2. giving a short read of what the current results imply
3. recommending one next hypothesis family, not many
4. naming the exact file / code path where the next patch would land
```
