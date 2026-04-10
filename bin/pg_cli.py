#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "state" / "current.yaml"
REPORT_SCRIPT = REPO_ROOT / "scripts" / "render_parameter_golf_run_report.py"
APP_VERSION = "0.0.1"
DEFAULT_REPORT_LOG = REPO_ROOT / "results" / "mlx_full_seq_mlp4x_200_realval_vb524k.txt"


class CliError(Exception):
    def __init__(self, message: str, *, error_type: str = "CliError", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


def _emit_envelope(command: str, *, ok: bool, result: dict[str, Any] | None = None, error: CliError | None = None) -> str:
    payload: dict[str, Any] = {"ok": ok, "command": command}
    if ok:
        payload["result"] = result or {}
    else:
        assert error is not None
        payload["error"] = {
            "type": error.error_type,
            "message": str(error),
            "details": error.details,
        }
    return json.dumps(payload, sort_keys=True, default=str)


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        raise CliError("missing state/current.yaml", error_type="StateError", details={"state_path": str(STATE_PATH)})
    payload = yaml.safe_load(STATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CliError(
            "state/current.yaml must parse to an object",
            error_type="StateError",
            details={"state_path": str(STATE_PATH)},
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./bin/pg",
        description="pg is the canonical CLI noun for Parameter Golf frontier state and normalized reports.",
        epilog=(
            "First success:\n"
            "  ./bin/pg leaderboard --json\n\n"
            "Machine path:\n"
            "  ./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"pg {APP_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("state", "Emit the full parsed repo state from state/current.yaml."),
        ("leaderboard", "Emit the current frontier summary from state/current.yaml."),
        ("queue", "Emit the active lane and drift summary from state/current.yaml."),
    ):
        sub = subparsers.add_parser(name, prog=f"./bin/pg {name}", help=help_text)
        sub.add_argument("--json", action="store_true", help="Emit the canonical pg JSON envelope.")

    report = subparsers.add_parser("report", prog="./bin/pg report", help="Render a normalized report for one archived run log.")
    report.add_argument("--log", required=True, help="Path to a run log under results/ or logs/.")
    report.add_argument("--output-dir", required=True, help="Directory for rendered report outputs.")
    report.add_argument(
        "--artifact-policy",
        choices=("prefer-local", "log-only"),
        default="prefer-local",
        help="Whether to prefer current local artifacts or log-recorded facts only.",
    )
    report.add_argument("--json", action="store_true", help="Emit the canonical pg JSON envelope.")
    return parser


def _state_result() -> dict[str, Any]:
    payload = _load_state()
    return {
        "state_path": str(STATE_PATH),
        "last_updated": str(payload.get("last_updated") or ""),
        "state": payload,
    }


def _leaderboard_result() -> dict[str, Any]:
    payload = _load_state()
    frontier = payload.get("frontier") if isinstance(payload.get("frontier"), dict) else {}
    focus = payload.get("current_focus") if isinstance(payload.get("current_focus"), dict) else {}
    return {
        "state_path": str(STATE_PATH),
        "last_updated": str(payload.get("last_updated") or ""),
        "primary_lane": str(focus.get("primary_lane") or ""),
        "repo_leader": frontier.get("repo_leader"),
        "architecture_baseline": frontier.get("architecture_baseline"),
        "over_budget_best": frontier.get("over_budget_best"),
        "smoke_baseline": frontier.get("smoke_baseline"),
        "latest_nearby_miss": frontier.get("latest_nearby_miss"),
    }


def _queue_result() -> dict[str, Any]:
    payload = _load_state()
    return {
        "state_path": str(STATE_PATH),
        "last_updated": str(payload.get("last_updated") or ""),
        "current_focus": payload.get("current_focus"),
        "lanes": payload.get("lanes"),
        "automation": payload.get("automation"),
        "known_drift": payload.get("known_drift"),
        "dead_families": payload.get("dead_families"),
    }


def _report_result(log_path: str, output_dir: str, artifact_policy: str) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--log",
            str(Path(log_path).expanduser()),
            "--output-dir",
            str(output_root),
            "--artifact-policy",
            str(artifact_policy),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CliError(
            "report render failed",
            error_type="ReportRenderFailed",
            details={
                "exit_code": proc.returncode,
                "stdout": str(proc.stdout or "").strip(),
                "stderr": str(proc.stderr or "").strip(),
            },
        )
    report_json = output_root / "report.json"
    report_md = output_root / "report.md"
    report_html = output_root / "index.html"
    if not report_json.is_file():
        raise CliError(
            "report render did not produce report.json",
            error_type="ReportRenderFailed",
            details={"output_dir": str(output_root)},
        )
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    return {
        "log": str(Path(log_path).expanduser().resolve()),
        "output_dir": str(output_root),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "report_html": str(report_html),
        "report": report_payload,
    }


def _print_leaderboard(result: dict[str, Any]) -> None:
    leader = result.get("repo_leader") if isinstance(result.get("repo_leader"), dict) else {}
    print("pg leaderboard")
    print(f"state_path: {result['state_path']}")
    print(f"primary_lane: {result['primary_lane']}")
    print(f"run_id: {leader.get('run_id', '')}")
    print(f"exact_val_bpb: {leader.get('exact_val_bpb', '')}")
    print(f"artifact_bytes: {leader.get('artifact_bytes', '')}")


def _print_queue(result: dict[str, Any]) -> None:
    focus = result.get("current_focus") if isinstance(result.get("current_focus"), dict) else {}
    lanes = result.get("lanes") if isinstance(result.get("lanes"), dict) else {}
    print("pg queue")
    print(f"primary_lane: {focus.get('primary_lane', '')}")
    print(f"secondary_lane: {focus.get('secondary_lane', '')}")
    print(f"known_drift: {len(result.get('known_drift') or [])}")
    for lane_id, lane_payload in sorted(lanes.items()):
        status = lane_payload.get("status") if isinstance(lane_payload, dict) else ""
        print(f"{lane_id}: {status}")


def _print_state(result: dict[str, Any]) -> None:
    print(f"state_path: {result['state_path']}")
    print(f"last_updated: {result['last_updated']}")
    print(json.dumps(result["state"], indent=2, sort_keys=True, default=str))


def _print_report(result: dict[str, Any]) -> None:
    report = result.get("report") if isinstance(result.get("report"), dict) else {}
    print("pg report")
    print(f"log: {result['log']}")
    print(f"report_json: {result['report_json']}")
    print(f"report_md: {result['report_md']}")
    print(f"report_html: {result['report_html']}")
    print(f"title: {report.get('title', '')}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    command = str(ns.command)
    try:
        if command == "state":
            result = _state_result()
            if ns.json:
                print(_emit_envelope("state", ok=True, result=result))
            else:
                _print_state(result)
            return 0
        if command == "leaderboard":
            result = _leaderboard_result()
            if ns.json:
                print(_emit_envelope("leaderboard", ok=True, result=result))
            else:
                _print_leaderboard(result)
            return 0
        if command == "queue":
            result = _queue_result()
            if ns.json:
                print(_emit_envelope("queue", ok=True, result=result))
            else:
                _print_queue(result)
            return 0
        if command == "report":
            result = _report_result(str(ns.log), str(ns.output_dir), str(ns.artifact_policy))
            if ns.json:
                print(_emit_envelope("report", ok=True, result=result))
            else:
                _print_report(result)
            return 0
        raise CliError(f"unsupported command: {command}", error_type="CliUsageError")
    except CliError as exc:
        if getattr(ns, "json", False):
            print(_emit_envelope(command, ok=False, error=exc))
        else:
            print(f"pg error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
