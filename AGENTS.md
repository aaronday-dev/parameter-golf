# Parameter Golf Agent Contract

This file is the repo-local contract for any agent working in
`/Users/aaronday/dev/parameter-golf`.

## Read Order

1. Read this file.
2. Read `state/current.yaml`.
3. Treat `state/current.yaml` as the single mutable source of current repo state.
4. Use `docs/`, `results/`, and git history as evidence and history, not as the
   mutable source of frontier facts when they disagree with `state/current.yaml`.

## Source Of Truth

- Update `state/current.yaml` first whenever any of these change:
  - current leader
  - smoke baseline
  - active queue
  - dead families
  - mixed-bit default
  - active compute lane
  - branch intent for the next real run
- Do not hand-copy those facts into multiple prompts or runbooks just to answer
  the current task.
- If any prompt, doc, worksheet, or runbook disagrees with
  `state/current.yaml`, report the drift explicitly and follow the YAML unless
  the user overrides it.
- Use explicit dates in `YYYY-MM-DD` format when time matters.

## Execution Defaults

- Smallest reversible slice first.
- Prefer one task, one patch, one check.
- Do not reopen dead families listed in `state/current.yaml` without a fresh
  reason.
- Do not invent archived reports from mutable logs state.
- Treat external compute as a bounded lane with explicit gates from
  `state/current.yaml`.
- `./bin/pg` is the canonical CLI noun for repo state and normalized reports.

## Verification

- Run the narrowest meaningful check for the change.
- For state/doc-only changes, run `git diff --check`.
- If a repo-state verifier exists, run it when changing state or generated docs.
- If verification is limited, say exactly what was not verified.

## Change Control

- Prefer generators, templates, and scripts over duplicated instructions.
- Keep research narrative in `docs/parameter-golf-research-log.md`.
- Keep operational receipts out of narrative docs when a dedicated ops surface
  exists.
- Do not delete data, rewrite history, or make destructive cleanup changes
  unless the user explicitly asks.

## Output Expectations

For non-trivial work, return:

- Intent
- Evidence used
- Changes
- Verification
- Rollback
- Unknowns

## Canonical Commands

- help: `./bin/pg --help`
- leaderboard: `./bin/pg leaderboard --json`
- queue: `./bin/pg queue --json`
- state: `./bin/pg state --json`
- report: `./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json`
