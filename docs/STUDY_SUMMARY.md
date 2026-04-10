# Parameter Golf Study Summary

Status: finished study  
Closed on: `2026-04-10`

## What This Repo Became

This repo is the closed record of a compression and artifact-design study under a
hard `16,000,000`-byte artifact cap.

It is no longer an active frontier search.

The point of the study was to answer a practical question:

- what kinds of model and artifact changes actually survive exact post-roundtrip
  evaluation under a hard byte budget?

## Study Constraints

- exact post-roundtrip `val_bpb` was the real objective
- the decimal `16,000,000`-byte artifact cap was the hard boundary
- local work was primarily Apple Silicon M4 via MLX / Metal
- artifact design and exact eval had to stay legible enough to audit later

## Final Frontier Snapshot

Best capped repo leader:

- run: `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- exact `val_bpb = 2.35570158`
- artifact bytes: `15,109,864`
- compressor: `lzma`
- provenance: offline rank-64 fp16 residual sidecar on
  `blocks.0.mlp.proj.weight`

Best architecture-only full result:

- run: `mlx_full_seq_mlp4x_200_realval_vb524k`
- exact `val_bpb = 2.35796063`

Best over-budget local result:

- run: `mlx_full_seq_mlp4x_keepf_block0proj_200_realval_v2`
- exact `val_bpb = 2.35551193`
- artifact bytes: `16,263,292`

Best no-retrain mixed-bit calibration point:

- profile: `mlp6_attn6`
- exact `val_bpb = 2.36480881`
- artifact bytes: `9,521,536`

## Findings

1. Exact post-roundtrip `val_bpb` beat prettier proxy metrics.
   Reconstruction wins, local residual improvements, and clever transforms did
   not matter if the exact post-roundtrip number got worse.

2. Straightforward sequential capacity beat most “fancier” local mechanism work.
   Shared-core recurrence helped at one stage, but the strongest local base family
   ended up being a plain sequential higher-capacity model.

3. Offline artifact design mattered more than another nearby local retrain.
   The capped repo leader came from preserving sacred-tensor gain at artifact time,
   not from a fresh architecture breakthrough.

4. Mixed-bit quantization created real byte headroom but not a direct frontier win.
   The calibration sweep clarified a useful export knee, but it did not by itself
   beat the capped leader without retraining.

5. Several plausible branches died cleanly and should stay dead.
   Basis-preconditioned quantization, naive local residual variants, bigram-hash
   retries, soft-floor retries, and fp32 sidecar escalation did not justify
   reopening.

## Dead Families

- bigram-hash retry
- soft-floor retry
- row-subset sacred-tensor rescue
- fp32 sidecar retry as a frontier move
- naive tiled local residual retry
- naive mixed local residual retry
- sign-Hadamard or basis-preconditioned quantization branch

## The One Unanswered Question

There was one remaining credible path forward when the repo was frozen:

- a bounded external-compute retrain using `10L / MLP3x / mlp6_attn6`

That lane was prepared in [docs/RUNPOD_START_HERE.md](/Users/aaronday/dev/parameter-golf/docs/RUNPOD_START_HERE.md) but never executed.

The repo is being closed without answering that question.

That means:

- there is no honest claim here that the project was fully exhausted in all
  possible compute regimes
- there is a clean claim that the local artifact-design and compression study
  reached a natural stopping point

## Why The Study Is Closed

- no new local mechanism family in this repo looks credible enough to justify
  more human time
- the remaining hope path requires fresh external compute, not more local
  ideation
- the strongest durable value is now the evidence trail, not continued churn

## How To Read This Repo Going Forward

Start here:

- [state/current.yaml](/Users/aaronday/dev/parameter-golf/state/current.yaml)
- [README.md](/Users/aaronday/dev/parameter-golf/README.md)
- [docs/STUDY_SUMMARY.md](/Users/aaronday/dev/parameter-golf/docs/STUDY_SUMMARY.md)

Then use:

- `./bin/pg leaderboard --json`
- `./bin/pg queue --json`
- `./bin/pg state --json`
- `./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json`

Primary evidence lives under:

- `results/`
- `results/reports/`
- `docs/parameter-golf-research-log.md`

Historical planning surfaces such as `docs/local-only-active-queue.md`,
`docs/parameter-golf-next-runs.md`, and `docs/RUNPOD_START_HERE.md` should now be
read as archived context, not as active marching orders.
