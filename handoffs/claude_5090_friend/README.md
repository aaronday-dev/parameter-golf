# 5090 Friend Handoff

This folder is the single handoff package for a friend with one NVIDIA `5090`.

If your friend is uneasy, start here:

- open [START_HERE.md](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/START_HERE.md)
- let them choose a preflight-only pass first
- only move to training once the environment check feels boring and legible

Use this repo branch:

- `codex/public-parity-gap`

Snapshot:

- commit: `dd89789`

## Goal

Run one high-EV CUDA experiment, and optionally one second run, to test whether
mixed-bit compression headroom can fund enough extra model capacity to beat the
current capped local leader.

Current capped local leader:

- run: `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
- exact `val_bpb = 2.35570158`
- artifact `15,109,864` bytes

Why this is the right borrowed-GPU test:

- sliding eval is now implemented locally as a separate parity track
- the first bigram-hash branch already lost and should not be retried first
- mixed-bit offline quantization is not a direct win on the current model, but it buys massive headroom:
  - baseline 8-bit export:
    - `14,813,668` bytes
    - `2.35586296`
  - mixed-bit export:
    - `7,244,516` bytes
    - `2.38623309`
- that means the live question is no longer "does mixed precision help by itself?"
- it is:
  - "can mixed precision buy enough additional model capacity to win after retraining?"
- fp32-factor residual sidecars were rechecked locally and are not the borrowed-GPU path

## What Is In This Folder

- [README.md](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/README.md)
  - human-facing instructions
- [START_HERE.md](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/START_HERE.md)
  - low-pressure entry point for a nervous human
- [PASTE_TO_CLAUDE.md](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/PASTE_TO_CLAUDE.md)
  - the exact instruction block to paste into Claude
- [run_parameter_golf_cuda_5090.sh](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/run_parameter_golf_cuda_5090.sh)
  - one-command CUDA wrapper for the primary/secondary runs
- [mixed_precision_quant_fullval.json](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/mixed_precision_quant_fullval.json)
  - the offline evidence that motivates the run

## What You Send Your Friend

Send exactly this:

1. The branch name:
   - `codex/public-parity-gap`
2. This folder path:
   - `handoffs/claude_5090_friend/`
3. One sentence:
   - "Please open `START_HERE.md` in that folder. If you want the safest mode, ask Claude to do only the preflight first."

## What Success Looks Like

Lowest-stress useful outcome:

- CUDA and dataset preflight pass cleanly, even if the full run waits

Primary run:

- `10` layers
- `MLP 3x`
- `512` width
- mixed-bit export profile

Secondary run only if primary is encouraging:

- same recipe
- `11` layers instead of `10`

Encouraging means:

- exact roundtrip `val_bpb` beats `2.35570158`
  or
- it is close enough that it is clearly the best architecture-side progress in this branch

## Important Constraints

- Do not spend the borrowed GPU on bigram-hash retries first.
- Do not spend it on sacred-tensor carrier sweeps first.
- Do not spend it on fp32-factor sidecar sweeps.
- Do not use `archive_parameter_golf_run.py` to invent archived reports from mutable logs state.
- Keep the human out of run loops unless there is a real blocker.
