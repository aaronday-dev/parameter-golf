# Claude 5090 Run Package

This is the single handoff package for a friend with one NVIDIA `5090` and a solid CPU.

Goal:

- run one high-EV CUDA experiment, and optionally one second run
- do not waste time on dead local branches
- do not ask the human to copy-paste a bunch of loop commands if Claude can run them
- keep the handoff low-pressure enough that a nervous human can still start with a preflight-only pass

Use this repo snapshot:

- branch: `codex/public-parity-gap`

Current local facts that matter:

- best capped verified local result:
  - `mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1`
  - exact `val_bpb = 2.35570158`
  - artifact `15,109,864`
- sliding eval now exists as a separate parity track
- first bigram-hash token-local bias lost and is downranked
- fp32-factor residual sidecars did not change the frontier and should stay off the borrowed-GPU path
- mixed-bit offline export is not a direct win on the current model, but the full calibration sweep clarified the useful knee:
  - baseline 8-bit exact `2.35586296`, artifact `14,813,668`
  - recommended `mlp6_attn6` exact `2.36480881`, artifact `9,521,536`
  - backup `mlp5_attn7` exact `2.38371061`, artifact `8,527,564`
  - the old `mlp5_attn6` center point is no longer the default

This means the `5090` should be used for exactly one thing:

- test whether mixed-bit headroom can fund enough extra capacity to beat the current capped leader after retraining

## What Claude Should Do

Paste the block below to Claude on the friend machine, from inside the repo.

```text
Work only in the currently checked-out repository root.

Use the checked-out branch:
- codex/public-parity-gap

Objective:
- run one primary CUDA training job on the friend's single NVIDIA 5090
- optionally run one second CUDA job only if the first one is promising
- keep the human out of the loop unless there is a real blocker
- if anything looks risky or confusing, stop after preflight and report readiness instead of improvising

Important local facts:
- current capped local leader is:
  - mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1
  - exact val_bpb = 2.35570158
  - artifact = 15,109,864 bytes
- bigram hash already lost locally; do not spend time there
- fp32-factor residual sidecars were checked locally and did not justify more bytes; do not spend borrowed GPU time there
- sliding eval exists, but use contiguous eval for the main gating result unless asked otherwise
- the live hypothesis is compression-funded capacity:
  - lower-bit mixed precision buys huge byte headroom
  - the question is whether that headroom can fund a larger model that beats 2.35570158
- the current recommended mixed-bit default is mlp6_attn6
- mlp5_attn7 is the smaller backup only if tighter byte pressure matters

What is already implemented in this branch:
- sliding eval in train_gpt.py and train_gpt_mlx.py
- mixed-bit quantization in train_gpt.py via INTX_BITS_BY_NAME
- CUDA wrapper script:
  - handoffs/claude_5090_friend/run_parameter_golf_cuda_5090.sh

Evidence file:
- handoffs/claude_5090_friend/mixed_precision_quant_fullval.json

What I want you to do:

1. Verify the environment.
   - If .venv is missing, create it and install requirements.
   - Verify CUDA is available from Python/PyTorch.
   - If the full sp1024 dataset/tokenizer are missing, download them with:
     .venv/bin/python data/cached_challenge_fineweb.py --variant sp1024
   - If the human asks for the safest possible start, stop here and report:
     - whether CUDA works
     - whether the dataset/tokenizer are ready
     - the exact command you would run next for the primary job

2. Run the primary job:
   - RUN_MODE=primary handoffs/claude_5090_friend/run_parameter_golf_cuda_5090.sh

Primary job intent:
- this is the only run that should happen by default
- do not ask the human to tune knobs or babysit loop commands
- 10 layers
- MLP 3x
- model_dim 512
- heads 8 / kv heads 4
- mixed-bit export profile (mlp6_attn6):
  - *.mlp.fc.weight:6
  - *.mlp.proj.weight:6
  - *.attn.c_q.weight:6
  - *.attn.c_k.weight:6
  - *.attn.c_v.weight:6
  - *.attn.proj.weight:6
- contiguous exact roundtrip is the main gating metric

3. Only run the secondary job if the primary result is genuinely encouraging.

Encouraging means:
- exact roundtrip val_bpb beats 2.35570158
  OR
- exact roundtrip val_bpb is close enough that the result is clearly the best architecture-side progress in this branch

If encouraging, run:
- RUN_MODE=secondary handoffs/claude_5090_friend/run_parameter_golf_cuda_5090.sh

Secondary job intent:
- same recipe, but 11 layers instead of 10
- keep the same mlp6_attn6 profile unless there is a concrete reason to try the smaller mlp5_attn7 backup

4. After runs finish, report:
- run ids
- exact roundtrip val_bpb
- compressed artifact bytes
- whether either run beat 2.35570158
- whether either run looks good enough to archive and promote in this repo

Constraints:
- Do not reopen bigram-hash or soft-floor branches.
- Do not reopen sacred-tensor carrier sweeps or fp32 sidecar branches on the borrowed GPU.
- Do not refactor unrelated code.
- Do not use archive_parameter_golf_run.py to invent archived reports from mutable logs state.
- Keep the human out of run loops unless you hit a real blocker.

If you need a safe first check before the full run:
- inspect handoffs/claude_5090_friend/run_parameter_golf_cuda_5090.sh
- verify train_gpt.py can see CUDA
- then run the primary job directly
```

## What the human should do

1. Send your friend this branch name:
   - `codex/public-parity-gap`
2. Send your friend this folder:
   - `handoffs/claude_5090_friend/`
3. Tell your friend:
   - open `handoffs/claude_5090_friend/START_HERE.md`
   - if they feel nervous, they can ask Claude to do only the preflight first

Low-pressure fallback:

- a successful preflight-only pass is still useful next week
- the friend does not need to babysit loops or pick hyperparameters by hand

That is the cleanest handoff with the least translation burden.
