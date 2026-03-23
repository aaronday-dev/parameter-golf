#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


ARTIFACT_BUDGET_BYTES = 16_000_000
RUN_ID_RE = re.compile(r"^run_id:([A-Za-z0-9_.-]+)$", re.MULTILINE)
VAL_LOADER_RE = re.compile(r"^val_loader:shards pattern=(.+) tokens:(\d+)$", re.MULTILINE)
TRAIN_LOADER_RE = re.compile(r"^train_loader:shards pattern=(.+)$", re.MULTILINE)
TOKENIZER_RE = re.compile(r"^tokenizer_path:(.+)$", re.MULTILINE)
MODEL_RE = re.compile(
    r"^model_params:(\d+) vocab_size:(\d+) layers:(\d+) dim:(\d+) heads:(\d+) kv_heads:(\d+) seq_len:(\d+) tie_embeddings:(True|False)$",
    re.MULTILINE,
)
SHARED_CORE_RE = re.compile(r"^shared_core:(.+)$", re.MULTILINE)
ITERATION_RE = re.compile(r"^iterations:(.+)$", re.MULTILINE)
OPTIMIZER_RE = re.compile(r"^optimizer:(.+)$", re.MULTILINE)
COMPUTE_RE = re.compile(r"^compute_dtype:(.+)$", re.MULTILINE)
DTYPES_RE = re.compile(r"^dtypes (.+)$", re.MULTILINE)
SAVED_MODEL_RE = re.compile(r"^saved_model:(.+) bytes:(\d+)$", re.MULTILINE)
SERIALIZED_RE = re.compile(
    r"^serialized_model_int8_(zlib|lzma):(\d+) bytes \(payload:(\d+) raw_[a-z]+:(\d+) payload_ratio:([0-9.]+)x\)$",
    re.MULTILINE,
)
STEP_VAL_RE = re.compile(
    r"^step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:(\d+)ms(?: step_avg:([0-9.]+)ms)?$",
    re.MULTILINE,
)
ROUNDTRIP_RE = re.compile(
    r"^final_int8_(zlib|lzma)_roundtrip val_loss:([0-9.]+) val_bpb:([0-9.]+) eval_time:(\d+)ms$",
    re.MULTILINE,
)
ROUNDTRIP_EXACT_RE = re.compile(
    r"^final_int8_(zlib|lzma)_roundtrip_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one Parameter Golf run log into normalized JSON, Markdown, and HTML."
    )
    parser.add_argument("--log", required=True, help="Path to a run log under logs/ or results/.")
    parser.add_argument("--output-dir", required=True, help="Directory for report outputs.")
    parser.add_argument(
        "--artifact-policy",
        choices=("prefer-local", "log-only"),
        default="prefer-local",
        help="Whether to prefer current local artifact files or rely only on facts recorded in the log.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2) + "\n")


def last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1]


def parse_compact_fields(compact: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+):([^\s]+)", compact):
        fields[key] = parse_scalar(value)
    return fields


def parse_scalar(value: str) -> Any:
    if value in {"True", "False"}:
        return value == "True"
    if value in {"-", "inf"}:
        return value
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def find_artifact_file(log_path: Path, run_id: str) -> tuple[Path | None, str | None]:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = []
    for base_dir in (log_path.parent, repo_root / "logs"):
        for compressor, ext in (("lzma", ".ptx"), ("zlib", ".ptz")):
            candidates.append((base_dir / f"{run_id}_mlx_model.int8{ext}", compressor))
    for candidate, compressor in candidates:
        if candidate.is_file():
            return candidate, compressor
    return None, None


def resolve_storage_artifact(
    log_path: Path,
    run_id: str,
    logged_storage_compressor: str,
    logged_compressed_model_bytes: int,
    artifact_policy: str,
) -> tuple[Path | None, str, int, str]:
    if artifact_policy == "log-only":
        return None, logged_storage_compressor, logged_compressed_model_bytes, "log"
    if artifact_policy != "prefer-local":
        raise ValueError(f"Unsupported artifact_policy={artifact_policy!r}")

    artifact_path, artifact_path_compressor = find_artifact_file(log_path, run_id)
    if artifact_path is not None:
        return (
            artifact_path,
            artifact_path_compressor or logged_storage_compressor,
            artifact_path.stat().st_size,
            "file",
        )
    return None, logged_storage_compressor, logged_compressed_model_bytes, "log"


def resolve_existing_path(log_path: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(raw_path)
    search_roots = [Path.cwd(), log_path.parent, repo_root]
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
    return None


def build_headlines(report: dict[str, Any]) -> list[str]:
    model = report["model"]
    compression = report["compression"]
    results = report["results"]
    derived = report["derived"]
    shared_core = report["architecture"]["shared_core"]
    evaluation = report["evaluation"]

    findings = [
        f"Exact post-roundtrip val_bpb is {results['exact_val_bpb']:.8f} on a "
        f"{evaluation['roundtrip_compressor']} roundtrip."
    ]
    if compression["storage_source"] == "file":
        findings.append(
            f"Current local stored artifact is {compression['storage_compressor']} "
            f"at {compression['compressed_model_bytes']:,} bytes."
        )
    else:
        findings.append(
            f"The archived log reports a {compression['storage_compressor']} artifact "
            f"at {compression['compressed_model_bytes']:,} bytes."
        )
    if derived["artifact_budget_margin_bytes"] is not None:
        if derived["artifact_budget_margin_bytes"] >= 0:
            findings.append(
                f"The compressed artifact is under the decimal 16,000,000-byte cap by "
                f"{derived['artifact_budget_margin_bytes']:,} bytes."
            )
        else:
            findings.append(
                f"The compressed artifact is over the decimal 16,000,000-byte cap by "
                f"{abs(derived['artifact_budget_margin_bytes']):,} bytes."
            )
    if report["results"]["roundtrip_penalty_bpb"] is not None:
        findings.append(
            f"Quantized roundtrip changed val_bpb by {report['results']['roundtrip_penalty_bpb']:+.8f} "
            f"relative to the final in-run validation."
        )
    arch_label = "shared-core" if shared_core.get("enabled") else "sequential"
    findings.append(
        f"The run is a {arch_label} architecture with {model['layers']} layers, dim {model['dim']}, "
        f"and schedule `{shared_core.get('schedule', 'unknown')}`."
    )
    if derived["is_full_eval"]:
        findings.append(
            f"This is a full-eval result over {report['evaluation']['val_tokens']:,} validation tokens."
        )
    else:
        findings.append(
            f"This is a smoke-scale result over {report['evaluation']['val_tokens']:,} validation tokens."
        )
    return findings


def build_report(log_path: Path, artifact_policy: str = "prefer-local") -> dict[str, Any]:
    text = read_text(log_path)

    run_match = last_match(RUN_ID_RE, text)
    if run_match is None:
        raise ValueError(f"Missing runtime run_id in {log_path}")
    run_id = run_match.group(1)

    val_loader_match = last_match(VAL_LOADER_RE, text)
    if val_loader_match is None:
        raise ValueError(f"Missing validation loader line in {log_path}")

    model_match = last_match(MODEL_RE, text)
    if model_match is None:
        raise ValueError(f"Missing model summary line in {log_path}")

    shared_core_match = last_match(SHARED_CORE_RE, text)
    if shared_core_match is None:
        raise ValueError(f"Missing shared_core summary line in {log_path}")

    iteration_match = last_match(ITERATION_RE, text)
    if iteration_match is None:
        raise ValueError(f"Missing iteration summary line in {log_path}")

    serialized_match = last_match(SERIALIZED_RE, text)
    if serialized_match is None:
        raise ValueError(f"Missing compressed artifact line in {log_path}")

    roundtrip_match = last_match(ROUNDTRIP_RE, text)
    if roundtrip_match is None:
        raise ValueError(f"Missing roundtrip evaluation line in {log_path}")

    roundtrip_exact_match = last_match(ROUNDTRIP_EXACT_RE, text)
    if roundtrip_exact_match is None:
        raise ValueError(f"Missing exact roundtrip evaluation line in {log_path}")

    step_val_match = last_match(STEP_VAL_RE, text)
    optimizer_match = last_match(OPTIMIZER_RE, text)
    compute_match = last_match(COMPUTE_RE, text)
    dtypes_match = last_match(DTYPES_RE, text)
    tokenizer_match = last_match(TOKENIZER_RE, text)
    train_loader_match = last_match(TRAIN_LOADER_RE, text)
    saved_model_match = last_match(SAVED_MODEL_RE, text)

    model = {
        "params": int(model_match.group(1)),
        "vocab_size": int(model_match.group(2)),
        "layers": int(model_match.group(3)),
        "dim": int(model_match.group(4)),
        "heads": int(model_match.group(5)),
        "kv_heads": int(model_match.group(6)),
        "seq_len": int(model_match.group(7)),
        "tie_embeddings": model_match.group(8) == "True",
    }
    shared_core_fields = parse_compact_fields(shared_core_match.group(1))
    training_fields = parse_compact_fields(f"iterations:{iteration_match.group(1)}")

    logged_storage_compressor = serialized_match.group(1)
    logged_compressed_model_bytes = int(serialized_match.group(2))
    artifact_path, storage_compressor, compressed_model_bytes, storage_source = resolve_storage_artifact(
        log_path,
        run_id,
        logged_storage_compressor,
        logged_compressed_model_bytes,
        artifact_policy,
    )

    payload_bytes = int(serialized_match.group(3))
    raw_serialized_bytes = int(serialized_match.group(4))
    payload_ratio = float(serialized_match.group(5))
    artifact_margin = ARTIFACT_BUDGET_BYTES - compressed_model_bytes

    prequant_val_loss = float(step_val_match.group(3)) if step_val_match else None
    prequant_val_bpb = float(step_val_match.group(4)) if step_val_match else None
    train_time_ms = int(step_val_match.group(5)) if step_val_match else None

    roundtrip_compressor = roundtrip_match.group(1)
    exact_roundtrip_compressor = roundtrip_exact_match.group(1)
    if roundtrip_compressor != exact_roundtrip_compressor:
        raise ValueError(
            f"Roundtrip compressor mismatch in {log_path}: "
            f"{roundtrip_compressor!r} vs {exact_roundtrip_compressor!r}"
        )
    roundtrip_val_loss = float(roundtrip_match.group(2))
    roundtrip_val_bpb = float(roundtrip_match.group(3))
    roundtrip_eval_ms = int(roundtrip_match.group(4))
    exact_val_loss = float(roundtrip_exact_match.group(2))
    exact_val_bpb = float(roundtrip_exact_match.group(3))
    roundtrip_penalty = None
    if prequant_val_bpb is not None:
        roundtrip_penalty = exact_val_bpb - prequant_val_bpb

    saved_model_path = None if saved_model_match is None else saved_model_match.group(1)
    saved_model_file = (
        resolve_existing_path(log_path, saved_model_path)
        if artifact_policy == "prefer-local"
        else None
    )

    report = {
        "schema_version": 1,
        "title": f"Parameter Golf Report: {run_id}",
        "subtitle": "Normalized single-run view for architecture, compression, and post-roundtrip evaluation.",
        "run": {
            "run_id": run_id,
            "source_log": str(log_path),
            "train_loader_pattern": None if train_loader_match is None else train_loader_match.group(1),
            "val_loader_pattern": val_loader_match.group(1),
            "tokenizer_path": None if tokenizer_match is None else tokenizer_match.group(1),
        },
        "model": model,
        "architecture": {
            "family": "shared_core" if bool(shared_core_fields.get("enabled")) else "sequential",
            "shared_core": shared_core_fields,
            "raw_shared_core_line": shared_core_match.group(0),
        },
        "training": {
            "iterations": training_fields.get("iterations"),
            "train_batch_tokens": training_fields.get("train_batch_tokens"),
            "grad_accum_steps": training_fields.get("grad_accum_steps"),
            "microbatch_tokens": training_fields.get("microbatch_tokens"),
            "microbatch_batch_size": training_fields.get("microbatch_batch_size"),
            "val_max_batch_tokens": training_fields.get("val_max_batch_tokens", training_fields.get("val_batch_size")),
            "verify_quantized_roundtrip": training_fields.get("verify_quantized_roundtrip"),
            "warmup_steps": training_fields.get("warmup_steps"),
            "max_wallclock_seconds": training_fields.get("max_wallclock_seconds"),
            "optimizer_summary": None if optimizer_match is None else optimizer_match.group(1),
            "compute_dtype_summary": None if compute_match is None else compute_match.group(1),
            "tensor_dtype_summary": None if dtypes_match is None else dtypes_match.group(1),
            "train_time_ms": train_time_ms,
        },
        "compression": {
            "storage_compressor": storage_compressor,
            "logged_storage_compressor": logged_storage_compressor,
            "compressed_model_bytes": compressed_model_bytes,
            "logged_compressed_model_bytes": logged_compressed_model_bytes,
            "payload_bytes": payload_bytes,
            "raw_serialized_bytes": raw_serialized_bytes,
            "payload_ratio": payload_ratio,
            "artifact_budget_bytes": ARTIFACT_BUDGET_BYTES,
            "artifact_budget_margin_bytes": artifact_margin,
            "artifact_budget_status": "within_budget" if artifact_margin >= 0 else "over_budget",
            "storage_source": storage_source,
            "artifact_path": None if artifact_path is None else str(artifact_path),
            "saved_model_path": None if saved_model_file is None else str(saved_model_file),
            "saved_model_bytes": None if saved_model_match is None else int(saved_model_match.group(2)),
        },
        "evaluation": {
            "val_tokens": int(val_loader_match.group(2)),
            "is_full_eval": int(val_loader_match.group(2)) >= 10_000_000,
            "roundtrip_compressor": roundtrip_compressor,
            "prequant_val_loss": prequant_val_loss,
            "prequant_val_bpb": prequant_val_bpb,
            "roundtrip_val_loss": roundtrip_val_loss,
            "roundtrip_val_bpb": roundtrip_val_bpb,
            "exact_val_loss": exact_val_loss,
            "exact_val_bpb": exact_val_bpb,
            "roundtrip_eval_ms": roundtrip_eval_ms,
        },
    }
    report["results"] = {
        "exact_val_bpb": exact_val_bpb,
        "exact_val_loss": exact_val_loss,
        "roundtrip_penalty_bpb": roundtrip_penalty,
    }
    report["derived"] = {
        "compressed_bytes_per_param": compressed_model_bytes / model["params"],
        "artifact_budget_margin_bytes": artifact_margin,
        "artifact_budget_margin_pct": artifact_margin / ARTIFACT_BUDGET_BYTES,
        "payload_savings_vs_saved_model": None
        if report["compression"]["saved_model_bytes"] is None
        else 1.0 - (compressed_model_bytes / report["compression"]["saved_model_bytes"]),
        "is_full_eval": report["evaluation"]["is_full_eval"],
    }
    report["headline_findings"] = build_headlines(report)
    report["source_artifacts"] = [str(log_path)]
    if artifact_path is not None:
        report["source_artifacts"].append(str(artifact_path))
    if report["compression"]["saved_model_path"] is not None:
        report["source_artifacts"].append(report["compression"]["saved_model_path"])
    return report


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    model = report["model"]
    architecture = report["architecture"]
    training = report["training"]
    compression = report["compression"]
    evaluation = report["evaluation"]
    derived = report["derived"]
    compression_lines = [
        f"- Storage compressor: `{compression['storage_compressor']}`",
        f"- Compressed model bytes: `{fmt_num(compression['compressed_model_bytes'])}`",
        f"- Payload bytes: `{fmt_num(compression['payload_bytes'])}`",
        f"- Raw serialized bytes: `{fmt_num(compression['raw_serialized_bytes'])}`",
        f"- Payload ratio: `{fmt_num(compression['payload_ratio'])}`x",
        f"- Budget status: `{compression['artifact_budget_status']}`",
        f"- Budget margin: `{fmt_num(compression['artifact_budget_margin_bytes'])}` bytes",
        f"- Storage source: `{compression['storage_source']}`",
    ]
    if (
        compression["storage_source"] != "log"
        or compression["storage_compressor"] != compression["logged_storage_compressor"]
        or compression["compressed_model_bytes"] != compression["logged_compressed_model_bytes"]
    ):
        compression_lines.append(
            f"- Logged artifact: `{compression['logged_storage_compressor']}` at "
            f"`{fmt_num(compression['logged_compressed_model_bytes'])}` bytes"
        )

    lines = [
        f"# {report['title']}",
        "",
        report["subtitle"],
        "",
        "## Summary",
        *[f"- {item}" for item in report["headline_findings"]],
        "",
        "## Run",
        f"- Run id: `{run['run_id']}`",
        f"- Source log: `{run['source_log']}`",
        f"- Validation tokens: `{fmt_num(evaluation['val_tokens'])}`",
        f"- Eval scope: `{'full' if evaluation['is_full_eval'] else 'smoke'}`",
        "",
        "## Model",
        f"- Params: `{fmt_num(model['params'])}`",
        f"- Topology: `{model['layers']}` layers, dim `{model['dim']}`, heads `{model['heads']}`, kv_heads `{model['kv_heads']}`",
        f"- Sequence length: `{model['seq_len']}`",
        f"- Tie embeddings: `{model['tie_embeddings']}`",
        f"- Architecture family: `{architecture['family']}`",
        f"- Shared-core schedule: `{architecture['shared_core'].get('schedule', 'n/a')}`",
        f"- Unique blocks: `{architecture['shared_core'].get('unique_blocks', 'n/a')}`",
        "",
        "## Training",
        f"- Iterations: `{training['iterations']}`",
        f"- Train batch tokens: `{fmt_num(training['train_batch_tokens'])}`",
        f"- Grad accum steps: `{training['grad_accum_steps']}`",
        f"- Microbatch tokens: `{fmt_num(training['microbatch_tokens'])}`",
        f"- Validation batch tokens: `{fmt_num(training['val_max_batch_tokens'])}`",
        f"- Warmup steps: `{training['warmup_steps']}`",
        f"- Max wallclock seconds: `{fmt_num(training['max_wallclock_seconds'], 3)}`",
        f"- Final train time: `{fmt_num(training['train_time_ms'])}` ms",
        "",
        "## Compression",
        *compression_lines,
        "",
        "## Evaluation",
        f"- Roundtrip compressor: `{evaluation['roundtrip_compressor']}`",
        f"- Final in-run val_bpb: `{fmt_num(evaluation['prequant_val_bpb'], 8)}`",
        f"- Exact roundtrip val_bpb: `{fmt_num(evaluation['exact_val_bpb'], 8)}`",
        f"- Roundtrip penalty: `{fmt_num(report['results']['roundtrip_penalty_bpb'], 8)}`",
        f"- Roundtrip eval time: `{fmt_num(evaluation['roundtrip_eval_ms'])}` ms",
        "",
        "## Derived",
        f"- Compressed bytes per parameter: `{fmt_num(derived['compressed_bytes_per_param'], 6)}`",
        f"- Budget margin percent: `{fmt_num(derived['artifact_budget_margin_pct'] * 100.0, 3)}`%",
        f"- Savings vs saved model: `{fmt_num(None if derived['payload_savings_vs_saved_model'] is None else derived['payload_savings_vs_saved_model'] * 100.0, 3)}`%",
        "",
        "## Source Artifacts",
        *[f"- `{item}`" for item in report["source_artifacts"]],
    ]
    return "\n".join(lines) + "\n"


def render_html(report: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    def metric(label: str, value: str) -> str:
        return (
            '<div class="metric">'
            f'<div class="metric-label">{esc(label)}</div>'
            f'<div class="metric-value">{esc(value)}</div>'
            "</div>"
        )

    def bullets(items: list[str]) -> str:
        return "".join(f"<li>{esc(item)}</li>" for item in items)

    run = report["run"]
    model = report["model"]
    architecture = report["architecture"]
    compression = report["compression"]
    evaluation = report["evaluation"]
    derived = report["derived"]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{esc(report['title'])}</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --surface: rgba(255, 252, 247, 0.82);
      --surface-strong: #fffaf3;
      --border: rgba(49, 39, 24, 0.14);
      --text: #2d241a;
      --text-dim: #6b5e4f;
      --accent: #9f3c12;
      --accent-soft: rgba(159, 60, 18, 0.12);
      --blue: #20405f;
      --good: #2b6d46;
      --shadow: 0 18px 48px rgba(49, 39, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font: 16px/1.55 "IBM Plex Sans", "Avenir Next", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(159, 60, 18, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(32, 64, 95, 0.10), transparent 26%),
        linear-gradient(180deg, #faf5ee 0%, #f1e8dc 100%);
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 20px 72px;
    }}
    .hero, .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      margin-bottom: 20px;
    }}
    .hero {{
      padding: 30px;
    }}
    .hero-grid, .card-grid {{
      display: grid;
      gap: 18px;
    }}
    .hero-grid {{
      grid-template-columns: 1.2fr 0.8fr;
    }}
    .card-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      border-radius: 18px;
      padding: 16px;
      background: var(--surface-strong);
      border: 1px solid var(--border);
    }}
    .metric-label {{
      color: var(--text-dim);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .metric-value {{
      color: var(--blue);
      font: 700 28px/1.05 "IBM Plex Mono", "SFMono-Regular", monospace;
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    h1, h2 {{
      margin: 0 0 10px;
      font-family: "Bricolage Grotesque", "Avenir Next", sans-serif;
    }}
    h1 {{
      font-size: 42px;
      line-height: 1.02;
      max-width: 16ch;
    }}
    h2 {{
      font-size: 26px;
    }}
    p, li {{
      color: var(--text-dim);
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      margin-right: 8px;
      margin-bottom: 8px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .card {{
      padding: 22px;
    }}
    code {{
      color: var(--blue);
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      word-break: break-word;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    @media (max-width: 920px) {{
      .hero-grid, .card-grid, .kpis {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        font-size: 34px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">Parameter Golf Run Report</div>
      <div class="hero-grid">
        <div>
          <h1>{esc(report['title'])}</h1>
          <p>{esc(report['subtitle'])}</p>
          <p>This is the reusable part worth stealing from the MoE workflow: one run, one normalized view of architecture, compression, and post-roundtrip truth.</p>
          <div>
            <span class="pill">{esc(run['run_id'])}</span>
            <span class="pill">{esc(architecture['family'])}</span>
            <span class="pill">{esc(f"storage:{compression['storage_compressor']}")}</span>
            <span class="pill">{esc(f"roundtrip:{evaluation['roundtrip_compressor']}")}</span>
          </div>
        </div>
        <div class="kpis">
          {metric("Exact val_bpb", fmt_num(evaluation["exact_val_bpb"], 8))}
          {metric("Artifact bytes", fmt_num(compression["compressed_model_bytes"]))}
          {metric("Budget margin", fmt_num(compression["artifact_budget_margin_bytes"]))}
          {metric("Validation tokens", fmt_num(evaluation["val_tokens"]))}
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Headline findings</h2>
      <ul>{bullets(report["headline_findings"])}</ul>
    </section>

    <section class="card">
      <h2>Model and architecture</h2>
      <div class="card-grid">
        <div>
          <p><strong>Params:</strong> <code>{esc(fmt_num(model['params']))}</code></p>
          <p><strong>Topology:</strong> <code>{esc(f"{model['layers']}L / dim {model['dim']} / {model['heads']} heads / {model['kv_heads']} kv")}</code></p>
          <p><strong>Seq len:</strong> <code>{esc(model['seq_len'])}</code></p>
        </div>
        <div>
          <p><strong>Family:</strong> <code>{esc(architecture['family'])}</code></p>
          <p><strong>Schedule:</strong> <code>{esc(architecture['shared_core'].get('schedule', 'n/a'))}</code></p>
          <p><strong>Unique blocks:</strong> <code>{esc(architecture['shared_core'].get('unique_blocks', 'n/a'))}</code></p>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Compression and evaluation</h2>
      <div class="card-grid">
        <div>
          <p><strong>Storage compressor:</strong> <code>{esc(compression['storage_compressor'])}</code></p>
          <p><strong>Compressed model:</strong> <code>{esc(fmt_num(compression['compressed_model_bytes']))}</code> bytes</p>
          <p><strong>Payload bytes:</strong> <code>{esc(fmt_num(compression['payload_bytes']))}</code></p>
          <p><strong>Payload ratio:</strong> <code>{esc(fmt_num(compression['payload_ratio']))}x</code></p>
          <p><strong>Budget status:</strong> <code>{esc(compression['artifact_budget_status'])}</code></p>
          <p><strong>Storage source:</strong> <code>{esc(compression['storage_source'])}</code></p>
        </div>
        <div>
          <p><strong>Roundtrip compressor:</strong> <code>{esc(evaluation['roundtrip_compressor'])}</code></p>
          <p><strong>Final in-run val_bpb:</strong> <code>{esc(fmt_num(evaluation['prequant_val_bpb'], 8))}</code></p>
          <p><strong>Exact roundtrip val_bpb:</strong> <code>{esc(fmt_num(evaluation['exact_val_bpb'], 8))}</code></p>
          <p><strong>Roundtrip penalty:</strong> <code>{esc(fmt_num(report['results']['roundtrip_penalty_bpb'], 8))}</code></p>
          <p><strong>Roundtrip eval time:</strong> <code>{esc(fmt_num(evaluation['roundtrip_eval_ms']))}</code> ms</p>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Source artifacts</h2>
      <ul>{bullets(report["source_artifacts"])}</ul>
    </section>
  </main>
</body>
</html>
"""


def write_report_bundle(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    html_path = output_dir / "index.html"
    write_json(json_path, report)
    write_text(markdown_path, render_markdown(report))
    write_text(html_path, render_html(report))
    return json_path, markdown_path, html_path


def main() -> None:
    args = parse_args()
    log_path = Path(args.log).resolve()
    output_dir = Path(args.output_dir).resolve()
    report = build_report(log_path, artifact_policy=args.artifact_policy)
    json_path, markdown_path, html_path = write_report_bundle(report, output_dir)

    print(json_path)
    print(markdown_path)
    print(html_path)


if __name__ == "__main__":
    main()
