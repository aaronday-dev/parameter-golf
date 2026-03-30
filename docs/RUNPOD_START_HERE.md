# RunPod First Tranche

Use this for the first and only `$10` compute tranche.

Question to buy:

> Can a directly controlled 10-layer, MLP-3x retrain at `model_dim=512`, `num_heads=8`, `num_kv_heads=4`, exported with the calibrated `mlp6_attn6` mixed-bit profile, beat the current capped local leader of `2.35570158` `val_bpb` under the `16,000,000`-byte cap on exact contiguous post-roundtrip evaluation?

Backup question:

> If it does not beat `2.35570158`, does it land close enough to justify exactly one follow-up `11L` run?

## Ground Truth

- Current capped leader:
  - `val_bpb = 2.35570158`
  - artifact `15,109,864`
- Current best mixed-bit calibration:
  - `mlp6_attn6`
  - `val_bpb = 2.36480881`
  - artifact `9,521,536`
- Current branch to run:
  - `codex/runpod-10l-v1`

## Rules

1. Spend at most `$10` first.
2. First tranche buys exactly one real answer.
3. One dry run is mandatory.
4. One real primary run only by default:
   - `10L`
   - `MLP3x`
   - `mlp6_attn6`
5. No second run unless the first clearly earns it.
6. If the first run misses cleanly, the project cools.
7. No improvising mid-run:
   - no sidecars
   - no basis/polar experiments
   - no quantizer changes
   - no eval-mode changes
   - no random hyperparameter wandering

## Before You Open RunPod

Use the exact repo snapshot below:

- branch: `codex/runpod-10l-v1`

If you are using another machine later, verify the commit you checked out:

```bash
git rev-parse --short HEAD
```

## What To Do In RunPod

1. Add only as much credit as you intend to use.
2. Create one Pod in the RunPod console.
3. Choose an on-demand Pod, not spot.
4. Choose one CUDA GPU you directly control:
   - prefer a single `RTX 5090`
   - acceptable fallback: a single `RTX 4090`
5. Choose a PyTorch image/template with Python already available.
6. Give it enough storage to avoid drama:
   - container disk: `50 GB`
   - volume disk: `100 GB`
7. Start the Pod.
8. Connect with SSH, not the browser terminal, for the real run.

## Bootstrap On The Pod

```bash
git clone https://github.com/aaronday-dev/parameter-golf.git
cd parameter-golf
git switch codex/runpod-10l-v1
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024
```

If CUDA is false, stop.

## Mandatory Dry Run

```bash
INTX_BITS_BY_NAME='*.mlp.fc.weight:6,*.mlp.proj.weight:6,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6' \
RUN_ID=cloud_10l_mlp3x_mlp6attn6_dryrun \
RUN_MODE=primary \
ITERATIONS=20 \
VAL_LOSS_EVERY=20 \
MAX_WALLCLOCK_SECONDS=120 \
TRAIN_BATCH_TOKENS=131072 \
VAL_MAX_BATCH_TOKENS=131072 \
bash scripts/run_parameter_golf_cuda_5090.sh
```

Dry run must prove:

- training starts
- validation runs
- quantized roundtrip runs
- exact `val_bpb` and artifact bytes are emitted

If any of those fail, stop.

## Real Primary Run

```bash
INTX_BITS_BY_NAME='*.mlp.fc.weight:6,*.mlp.proj.weight:6,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6' \
RUN_ID=cloud_10l_mlp3x_mlp6attn6_v1 \
RUN_MODE=primary \
MAX_WALLCLOCK_SECONDS=600 \
bash scripts/run_parameter_golf_cuda_5090.sh
```

## Preserve Artifacts Immediately

`train_gpt.py` writes fixed filenames like `final_model.pt` and `final_model.int8.*`, so copy them out right away:

```bash
mkdir -p cloud_runs/$RUN_ID
cp logs/$RUN_ID.txt cloud_runs/$RUN_ID/
cp final_model.pt cloud_runs/$RUN_ID/
cp final_model.int8.* cloud_runs/$RUN_ID/ 2>/dev/null || true
git rev-parse HEAD > cloud_runs/$RUN_ID/commit.txt
```

Extract the result lines:

```bash
rg -n "final_int8_.*_roundtrip_exact|Total submission size int8|Serialized model int8" logs/$RUN_ID.txt
```

Record only:

- branch
- commit SHA
- `RUN_ID`
- exact contiguous roundtrip `val_bpb`
- compressed artifact bytes
- under-cap yes/no

## Interpretation Table

Reference points:

- current capped leader: `2.35570158`
- plain 8-bit baseline: `2.35586296`
- best current mixed-bit no-retrain point: `2.36480881`

Use this table:

| Result | Meaning | Action |
|---|---|---|
| dry run fails, exact roundtrip missing, or artifact `> 16,000,000` | invalid | stop |
| `val_bpb <= 2.35570158` and under cap | clear win | second tranche earned |
| `2.35570158 < val_bpb <= 2.35586296` and under cap | very strong near-win | second tranche earned |
| `2.35586296 < val_bpb <= 2.36025519` and under cap | alive enough | one `11L` follow-up justified |
| `2.36025519 < val_bpb <= 2.36253200` and under cap | weak | default stop |
| `2.36253200 < val_bpb <= 2.36480881` | barely better than old mixed-bit point | probably stop |
| `val_bpb > 2.36480881` | no real architecture-side gain | park the project |

## Only If The Primary Earns It

Then run exactly one follow-up:

```bash
INTX_BITS_BY_NAME='*.mlp.fc.weight:6,*.mlp.proj.weight:6,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6' \
RUN_ID=cloud_11l_mlp3x_mlp6attn6_v1 \
RUN_MODE=secondary \
MAX_WALLCLOCK_SECONDS=600 \
bash scripts/run_parameter_golf_cuda_5090.sh
```

## What Not To Do

Do not spend the first `$10` on:

- more quantizer microbranches
- asymmetric quantization
- polar/cartesian rewrites
- sidecar variants
- sliding-eval-only runs
- another `9L` control
- another local-only week

## Paste-Ready Agent Prompt

```text
Work only in this repo and do not improvise beyond the steps below.

Repo:
- branch: codex/runpod-10l-v1

Objective:
Answer exactly one question:
Can a directly controlled 10-layer, MLP-3x retrain at model_dim=512, num_heads=8, num_kv_heads=4, exported with the calibrated mlp6_attn6 mixed-bit profile, beat the current capped local leader of 2.35570158 val_bpb under the 16,000,000-byte cap on exact contiguous post-roundtrip evaluation?

Rules:
- One dry run first.
- One real primary run only by default.
- No extra architecture changes.
- No extra quantization experiments.
- No second run unless the first is under cap and clearly alive.
- Do not use archive_parameter_golf_run.py.

Use this mixed-bit profile exactly:
INTX_BITS_BY_NAME='*.mlp.fc.weight:6,*.mlp.proj.weight:6,*.attn.c_q.weight:6,*.attn.c_k.weight:6,*.attn.c_v.weight:6,*.attn.proj.weight:6'

Dry run:
RUN_ID=cloud_10l_mlp3x_mlp6attn6_dryrun
RUN_MODE=primary
ITERATIONS=20
VAL_LOSS_EVERY=20
MAX_WALLCLOCK_SECONDS=120
TRAIN_BATCH_TOKENS=131072
VAL_MAX_BATCH_TOKENS=131072
bash scripts/run_parameter_golf_cuda_5090.sh

Primary run:
RUN_ID=cloud_10l_mlp3x_mlp6attn6_v1
RUN_MODE=primary
MAX_WALLCLOCK_SECONDS=600
bash scripts/run_parameter_golf_cuda_5090.sh

After the primary run:
1. Copy logs/$RUN_ID.txt, final_model.pt, and final_model.int8.* into cloud_runs/$RUN_ID/
2. Report only:
   - commit SHA
   - run id
   - exact roundtrip val_bpb
   - artifact bytes
   - under-cap yes/no
   - whether the result earns the 11L follow-up
3. Only if result is at or below 2.36025519 under cap, recommend the 11L follow-up. Otherwise stop.
```
