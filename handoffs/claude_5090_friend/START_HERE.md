# Start Here

This handoff is meant to be low-pressure.

You do not need to understand the repo before starting.
You do not need to babysit shell loops.
If anything feels off, it is fine to stop after the preflight check.

## What To Do

1. Open this repo:
   - `/Users/aaronday/dev/parameter-golf`
2. Check out this branch:
   - `codex/public-parity-gap`
3. Open:
   - `handoffs/claude_5090_friend/PASTE_TO_CLAUDE.md`
4. Paste that file into Claude from inside the repo.
5. If you want the safest possible start, add one sentence:
   - `Do the preflight only and stop before training.`

## What Claude Should Handle

- environment check
- CUDA check
- dataset/tokenizer check
- the primary run if preflight is clean
- an optional second run only if the first one is genuinely promising
- a short final summary

## What You Are Not Being Asked To Do

- no manual loop babysitting
- no hyperparameter decisions by hand
- no mutable-log archiving
- no unrelated repo cleanup
