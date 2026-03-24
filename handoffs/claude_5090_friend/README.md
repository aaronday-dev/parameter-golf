# 5090 Friend Handoff

This folder is the single handoff package for a friend with one NVIDIA `5090`.

Use this repo branch:

- `codex/public-parity-gap`

Snapshot:

- commit: `f2be339`

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

## What Is In This Folder

- [README.md](/Users/aaronday/dev/parameter-golf/handoffs/claude_5090_friend/README.md)
  - human-facing instructions
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
   - "Please open `PASTE_TO_CLAUDE.md` in that folder and paste it to Claude from inside the repo."

## What Success Looks Like

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
- Do not use `archive_parameter_golf_run.py` to invent archived reports from mutable logs state.
- Keep the human out of run loops unless there is a real blocker.
