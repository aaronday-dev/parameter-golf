#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from datetime import date
from pathlib import Path
from typing import Any

from render_parameter_golf_run_report import ARTIFACT_BUDGET_BYTES, build_report, write_report_bundle


DEFAULT_CONTROL_LOG = Path("results/mlx_full_seq_mlp4x_200_realval_vb524k.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive one offline Parameter Golf artifact-eval result into results/ and results/reports/."
    )
    parser.add_argument("run_id", help="Archived offline result id to create under results/.")
    parser.add_argument("--result-json", type=Path, required=True, help="Offline evaluation JSON result.")
    parser.add_argument("--source-log", type=Path, required=True, help="Training run log that produced the source float artifact.")
    parser.add_argument("--source-float-artifact", type=Path, default=None, help="Source float artifact used for the offline derivation.")
    parser.add_argument("--control-log", type=Path, default=DEFAULT_CONTROL_LOG, help="Current capped control log for comparison.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--carrier-summary", required=True, help="Human-readable description of the offline carrier.")
    parser.add_argument("--target-tensor", required=True, help="Tensor that received the offline carrier.")
    parser.add_argument("--rank", type=int, required=True, help="Carrier rank or equivalent size parameter.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_note_text(
    *,
    run_id: str,
    result_json_path: Path,
    source_log: Path,
    source_float_artifact: Path | None,
    carrier_summary: str,
    target_tensor: str,
    rank: int,
    exact_val_bpb: float,
    artifact_bytes: int,
    val_tokens: int,
    source_exact_val_bpb: float,
    control_run_id: str,
    control_exact_val_bpb: float,
) -> str:
    lines = [
        f"run_id:{run_id}",
        f"result_type:offline_artifact_eval",
        f"archived_on:{date.today().isoformat()}",
        f"source_log:{source_log}",
        f"source_result_json:{result_json_path}",
    ]
    if source_float_artifact is not None:
        lines.append(f"source_float_artifact:{source_float_artifact}")
    lines.extend(
        [
            f"carrier_summary:{carrier_summary}",
            f"target_tensor:{target_tensor}",
            f"rank:{rank}",
            f"exact_val_bpb:{exact_val_bpb:.8f}",
            f"artifact_bytes:{artifact_bytes}",
            f"artifact_budget_margin_bytes:{ARTIFACT_BUDGET_BYTES - artifact_bytes}",
            f"val_tokens:{val_tokens}",
            f"source_exact_val_bpb:{source_exact_val_bpb:.8f}",
            f"control_run:{control_run_id}",
            f"control_exact_val_bpb:{control_exact_val_bpb:.8f}",
            "",
            "This is an offline derived artifact evaluation, not a new training run.",
            "It preserves the source model topology and training provenance, then swaps in a derived carrier at artifact time.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    result_json_path = args.result_json.resolve()
    source_log = args.source_log.resolve()
    control_log = (repo_root / args.control_log).resolve() if not args.control_log.is_absolute() else args.control_log.resolve()
    source_float_artifact = None if args.source_float_artifact is None else args.source_float_artifact.resolve()

    payload = read_json(result_json_path)
    rows = payload.get("rows", [])
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {result_json_path}, got {len(rows)}")
    row = rows[0]

    source_report = build_report(source_log, artifact_policy="log-only")
    control_report = build_report(control_log, artifact_policy="log-only")

    run_id = args.run_id
    exact_val_bpb = float(row["val_bpb"])
    artifact_bytes = int(row["artifact_bytes"])
    artifact_margin = ARTIFACT_BUDGET_BYTES - artifact_bytes
    val_tokens = int(source_report["evaluation"]["val_tokens"])
    source_exact_val_bpb = float(source_report["evaluation"]["exact_val_bpb"])
    control_exact_val_bpb = float(control_report["evaluation"]["exact_val_bpb"])
    control_run_id = str(control_report["run"]["run_id"])

    note_path = repo_root / "results" / f"{run_id}.txt"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        build_note_text(
            run_id=run_id,
            result_json_path=result_json_path,
            source_log=source_log,
            source_float_artifact=source_float_artifact,
            carrier_summary=args.carrier_summary,
            target_tensor=args.target_tensor,
            rank=args.rank,
            exact_val_bpb=exact_val_bpb,
            artifact_bytes=artifact_bytes,
            val_tokens=val_tokens,
            source_exact_val_bpb=source_exact_val_bpb,
            control_run_id=control_run_id,
            control_exact_val_bpb=control_exact_val_bpb,
        ),
        encoding="utf-8",
    )

    report = copy.deepcopy(source_report)
    report["title"] = f"Parameter Golf Report: {run_id}"
    report["subtitle"] = "Normalized offline artifact-eval view for a derived capped result."
    report["run"]["run_id"] = run_id
    report["run"]["source_log"] = str(note_path)
    report["run"]["source_kind"] = "offline_artifact_eval"
    report["run"]["source_training_log"] = str(source_log)
    report["run"]["source_result_json"] = str(result_json_path)
    if source_float_artifact is not None:
        report["run"]["source_float_artifact"] = str(source_float_artifact)

    report["compression"]["storage_compressor"] = "lzma"
    report["compression"]["logged_storage_compressor"] = "lzma"
    report["compression"]["compressed_model_bytes"] = artifact_bytes
    report["compression"]["logged_compressed_model_bytes"] = artifact_bytes
    report["compression"]["payload_bytes"] = int(row.get("payload_bytes") or 0)
    report["compression"]["raw_serialized_bytes"] = None
    report["compression"]["payload_ratio"] = None
    report["compression"]["artifact_budget_bytes"] = ARTIFACT_BUDGET_BYTES
    report["compression"]["artifact_budget_margin_bytes"] = artifact_margin
    report["compression"]["artifact_budget_status"] = "within_budget" if artifact_margin >= 0 else "over_budget"
    report["compression"]["storage_source"] = "derived_json"
    report["compression"]["artifact_path"] = None

    report["evaluation"]["roundtrip_compressor"] = "lzma"
    report["evaluation"]["prequant_val_loss"] = None
    report["evaluation"]["prequant_val_bpb"] = None
    report["evaluation"]["roundtrip_val_loss"] = None
    report["evaluation"]["roundtrip_val_bpb"] = None
    report["evaluation"]["exact_val_loss"] = None
    report["evaluation"]["exact_val_bpb"] = exact_val_bpb
    report["evaluation"]["roundtrip_eval_ms"] = None

    report["results"]["exact_val_bpb"] = exact_val_bpb
    report["results"]["exact_val_loss"] = None
    report["results"]["roundtrip_penalty_bpb"] = None

    report["derived"]["compressed_bytes_per_param"] = artifact_bytes / report["model"]["params"]
    report["derived"]["artifact_budget_margin_bytes"] = artifact_margin
    report["derived"]["artifact_budget_margin_pct"] = artifact_margin / ARTIFACT_BUDGET_BYTES
    report["derived"]["is_full_eval"] = report["evaluation"]["is_full_eval"]

    gain_vs_keepf = exact_val_bpb - source_exact_val_bpb
    gain_vs_control = exact_val_bpb - control_exact_val_bpb
    report["headline_findings"] = [
        f"Exact offline post-roundtrip val_bpb is {exact_val_bpb:.8f} on a derived lzma artifact.",
        f"The derived artifact is under the decimal 16,000,000-byte cap by {artifact_margin:,} bytes.",
        f"This result was derived offline from `{source_report['run']['run_id']}` using {args.carrier_summary} on `{args.target_tensor}`.",
        f"Relative to the over-budget keep-float source result, it gives back {gain_vs_keepf:+.8f} bpb.",
        f"Relative to the previous capped local leader `{control_run_id}`, it changes exact val_bpb by {gain_vs_control:+.8f}.",
        f"This is a full-eval result over {val_tokens:,} validation tokens."
        if report["evaluation"]["is_full_eval"]
        else f"This is a smoke-scale result over {val_tokens:,} validation tokens.",
    ]

    report["experiment"] = {
        "result_type": "offline_artifact_eval",
        "carrier_summary": args.carrier_summary,
        "target_tensor": args.target_tensor,
        "rank": args.rank,
        "source_training_run_id": source_report["run"]["run_id"],
        "control_run_id": control_run_id,
    }
    report["source_artifacts"] = [
        str(note_path),
        str(result_json_path),
        str(source_log),
    ]
    if source_float_artifact is not None:
        report["source_artifacts"].append(str(source_float_artifact))

    report_dir = repo_root / "results" / "reports" / run_id
    json_path, markdown_path, html_path = write_report_bundle(report, report_dir)
    print(note_path)
    print(json_path)
    print(markdown_path)
    print(html_path)


if __name__ == "__main__":
    main()
