# Parameter Golf MoE-Compress Handover Prompt

Use this prompt in the main Parameter Golf thread if you want that thread to
decide whether the `moe-compress` exploration is worth adopting locally.

```text
We explored this external repo on 2026-03-22:

https://github.com/0xSero/moe-compress

Treat this as a decision memo, not as a mandate to implement anything.

Ground truth from inspection:

- `moe-compress` is not a new compression algorithm repo.
- It is a very small orchestration wrapper around a MoE workflow:
  - pipeline runner
  - calibration-bundle builder
  - normalized report renderer
- The actual observe / prune / quantize / benchmark mechanics are NOT in that repo.
- The example pipeline shells out to missing vendor commands like:
  - `./vendor/run_observations.sh`
  - `./vendor/run_pruning.sh`
  - `./vendor/run_quantization.sh`
  - `./vendor/run_benchmarks.sh`
- So there is little direct algorithmic leverage for Parameter Golf from that repo itself.

Current local Parameter Golf reality:

- Our actual artifact-compression path already lives in:
  - `train_gpt.py`
  - `train_gpt_mlx.py`
- That path already does the real work that matters here:
  - int8 quantization
  - `zlib` / `lzma` artifact compression
  - decompression
  - exact roundtrip validation
- In other words:
  the external repo does not replace our core compression path.

The one part that looked worth stealing:

- the idea of a normalized, auditable, single-run report boundary
- not for MoE pruning
- for our existing Parameter Golf runs

I implemented a local prototype of exactly that useful slice and nothing more:

- new script:
  `scripts/render_parameter_golf_run_report.py`
- README hook:
  `README.md`

What the local prototype does:

- parses one archived Parameter Golf run log
- extracts:
  - run id
  - model topology
  - shared-core / schedule summary
  - train settings
  - compressed artifact facts
  - pre-roundtrip vs exact roundtrip validation
  - 16,000,000-byte budget margin
- emits:
  - `report.json`
  - `report.md`
  - `index.html`

What was verified locally:

- the script compiles
- it works on the current best `lzma` full-eval run:
  `results/mlx_full_seq_mlp4x_200_realval_vb524k.txt`
- it also works on an older `zlib` full-eval run:
  `results/mlx_full_seq_mlp3x_200_realval_vb524k.txt`
- it correctly prefers a real artifact file when present and falls back to the logged artifact size when not

Example generated outputs from the local check:

- `/private/tmp/parameter-golf-report-best/report.md`
- `/private/tmp/parameter-golf-report-best/index.html`
- `/private/tmp/parameter-golf-report-zlib/report.md`

Current recommendation:

- do NOT spend time porting the full `moe-compress` workflow into Parameter Golf
- there is no evidence that its orchestration shell would buy us score
- if we take anything, take only the reporting boundary

Decision question for this thread:

Should we keep and extend the new local report path, or should we stop here and avoid adding more workflow surface area?

If you decide to keep it, the next high-EV move is:

1. wire report generation into `scripts/archive_parameter_golf_run.py`
2. emit a normalized report automatically whenever a run is archived
3. keep the scope tight to existing PG logs and metrics
4. do not add MoE-style vendor abstraction or fake generality

Constraints for answering:

1. Stay inside `/Users/aaronday/dev/parameter-golf`
2. Treat this as an operability / legibility decision, not a model-quality hypothesis
3. Prefer the smallest reversible choice
4. If recommending implementation, name the exact next patch in this repo
5. If rejecting implementation, say why the extra workflow surface is not worth it

Start by:

1. reading `docs/parameter-golf-moe-compress-handover-prompt.md`
2. reading `scripts/render_parameter_golf_run_report.py`
3. reading `scripts/archive_parameter_golf_run.py`
4. deciding whether automatic report generation belongs in the archive flow
```
