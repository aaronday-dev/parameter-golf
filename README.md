# Parameter Golf

Parameter Golf was an experiment in model compression and artifact design under a
hard `16,000,000`-byte artifact cap. The experiment is finished. This repo is
now the closed record of what actually worked.

## Hits First

- Verified capped winner:
  `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1` reached exact
  post-roundtrip `val_bpb = 2.35570158` at `15,109,864` bytes via `lzma` on
  Apple Silicon M4 (`MLX` / `Metal`).
- Best architecture-only local result:
  `mlx_full_seq_mlp4x_200_realval_vb524k` reached exact `val_bpb = 2.35796063`.
  By the end, plain sequential capacity beat most of the more ornate local
  mechanism branches.
- Recurrence produced real signal before it plateaued: promoted mirror beat
  promoted cyclic by `0.00263377` `bpb`, and promoted directional `C` beat
  promoted mirror by `0.01039241` `bpb`.
- Mixed-bit calibration found a real export knee: `mlp6_attn6` reached exact
  `val_bpb = 2.36480881` at `9,521,536` bytes. That created major byte headroom,
  but not a no-retrain frontier win.
- Final lesson: exact post-roundtrip `val_bpb` beat prettier proxy metrics, and
  offline artifact design delivered the capped repo leader more effectively than
  another nearby architecture retrain.

## Bottom Line

- Exact post-roundtrip `val_bpb` was the truth metric. Proxy wins and prettier
  reconstructions did not count if the roundtrip number got worse.
- Straightforward sequential capacity was the strongest local base family.
- Shared-core mirror scheduling and directional correction were genuine wins,
  but they were stepping stones, not the final answer.
- Most damping, stabilization, attractor, retry, and rescue branches died
  cleanly and should stay dead.
- The only credible unanswered path was a bounded external-compute retrain
  (`10L / MLP3x / mlp6_attn6`). It was prepared and never executed.

## Status

Closed on `2026-04-10`.

This repo is now:

- a finished experiment record
- a clean evidence surface for archived runs and normalized reports
- a repo-local CLI for reading the frozen state

This repo is not:

- an active frontier search
- a claim that every possible compute regime was exhausted
- a prompt to reopen dead families by implication

Mutable repo truth lives in [`state/current.yaml`](state/current.yaml). Study
close-out lives in [`docs/STUDY_SUMMARY.md`](docs/STUDY_SUMMARY.md). The
canonical noun is `./bin/pg`.

If this README and [`state/current.yaml`](state/current.yaml) ever disagree,
trust the YAML.

## Start Here

- [`state/current.yaml`](state/current.yaml)
- [`docs/STUDY_SUMMARY.md`](docs/STUDY_SUMMARY.md)
- `./bin/pg leaderboard --json`

Canonical commands:

```bash
./bin/pg --help
./bin/pg leaderboard --json
./bin/pg queue --json
./bin/pg state --json
./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json
```

## Evidence Trail

Primary evidence lives under:

- `results/`
- `results/reports/`
- [`docs/parameter-golf-research-log.md`](docs/parameter-golf-research-log.md)

Historical planning docs are preserved as context, not current marching orders.
