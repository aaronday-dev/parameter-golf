# Parameter Golf

Parameter Golf is now a finished compression and artifact-design study under a hard artifact budget.

Current state: the canonical noun is `./bin/pg`. Mutable repo truth lives in `state/current.yaml`. The study close-out and findings live in `docs/STUDY_SUMMARY.md`.

It preserves:
- the capped repo leader
- the final study status and known drift
- normalized archived-run reports

It does not launch remote compute by itself, replace the raw trainers, reopen the frontier by implication, or turn research history into fake present-state truth.

## Study Status

Closed on `2026-04-10`.

Read this repo as:
- a finished study of compression and artifact design under a hard byte cap
- a preserved evidence surface for archived runs and normalized reports
- a repo-local CLI for reading the frozen state cleanly

Start with:
- `state/current.yaml`
- `docs/STUDY_SUMMARY.md`
- `./bin/pg leaderboard --json`

## Quick Start

Use the canonical noun first:

```bash
./bin/pg --help
```

First clean success:

```bash
./bin/pg leaderboard --json
```

Machine path:

```bash
./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json
```

Current capped repo leader from `state/current.yaml`:
- run: `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- exact `val_bpb = 2.35570158`
- compressed artifact size: `15,109,864` bytes via `lzma`
- hardware: Apple Silicon M4 via MLX / Metal

Public support target for this slice is repo-local execution from a clone with Python 3.11+.
For the canonical read-only CLI surface, install `pyyaml`.
For training and broader experiment work, install the full repo requirements.

## Canonical Commands

Show the current frontier:

```bash
./bin/pg leaderboard --json
```

Show the active lane and drift boundary:

```bash
./bin/pg queue --json
```

Show the full parsed repo state:

```bash
./bin/pg state --json
```

Render one normalized run report:

```bash
./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json
```

Compatibility surfaces remain available:
- `scripts/render_parameter_golf_run_report.py`
- `scripts/archive_parameter_golf_run.py`
- `scripts/archive_parameter_golf_offline_result.py`
- `scripts/run_parameter_golf_mlx_m4.sh`
- `train_gpt.py`
- `train_gpt_mlx.py`

## JSON Contract

All `./bin/pg ... --json` commands emit one envelope shape:

```json
{"ok":true,"command":"leaderboard","result":{...}}
{"ok":false,"command":"leaderboard","error":{"type":"StateError","message":"...","details":{...}}}
```

For `state`, the nested `result.state` payload is the parsed `state/current.yaml` document.

For `leaderboard`, the nested result reports the current repo leader, nearby baselines, and the current primary lane from the frozen state.

For `queue`, the nested result reports current focus, lane status, automation state, dead families, and known drift.

For `report`, the nested result includes `report_json`, `report_md`, `report_html`, and the normalized `report` payload.

## Trust Boundary

Parameter Golf is:
- a closed study record
- a report normalizer
- a local experiment workspace with a frozen frontier state

Parameter Golf is not:
- a guaranteed best-model launcher
- a rented-compute orchestrator
- a substitute for `state/current.yaml`
- an active instruction to keep searching

If a doc disagrees with `state/current.yaml`, treat the YAML as current truth unless the user explicitly overrides it.

## Operator Notes

Read-only CLI bootstrap:

```bash
python -m pip install pyyaml
./bin/pg leaderboard --json
```

Full experiment environment bootstrap:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install mlx
```

Download the smoke tokenizer and dataset prefix used by the local MLX wrappers:

```bash
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1 --build-smoke
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

Datasets are not vendored here, but the standalone repo now includes a small
published-data bootstrap helper for the local MLX path:

- `data/cached_challenge_fineweb.py`
- `data/README.md`
- `docs/upstream-openai-readme.md`

## Research Thread

The main running log lives in:

- `docs/parameter-golf-research-log.md`
- `docs/parameter-golf-hypothesis-worksheet.md`
- `docs/local-only-active-queue.md`
- `docs/STUDY_SUMMARY.md`

Current practical takeaway:

1. Shared-core mirror scheduling produced a measurable improvement over the earlier baseline.
2. Blunt contraction / damping / attractor-style controls mostly failed.
3. Spending more of the artifact budget on useful capacity produced the biggest
   recent gains.
4. The final capped repo leader came from offline artifact design, not another
   local architecture breakthrough.
5. The repo is now closed as a study; old planning docs are evidence and context,
   not active marching orders.
