#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from render_parameter_golf_run_report import build_report, write_report_bundle


EXACT_BPB_RE = re.compile(
    r"final_int8_(?:zlib|lzma)_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"
)
ARTIFACT_RE = re.compile(r"serialized_model_int8_(zlib|lzma):(\d+) bytes")
VAL_TOKENS_RE = re.compile(r"val_loader:.* tokens:(\d+)")
README_BEST_RE = re.compile(
    r"(## Current Best Verified Local Result\n\n"
    r"Best exact real-data result currently in this repo:\n\n)"
    r"- run: `([^`]+)`\n"
    r"- exact `val_bpb = ([0-9.]+)`\n"
    r"- compressed artifact size: ([^\n]+)\n"
    r"- hardware: ([^\n]+)\n",
    re.MULTILINE,
)
RESEARCH_START = "The current best promoted architectural result is now the MLX real-data promotion:\n\n"
RESEARCH_END = "\nThat displaced the earlier local leaders:\n"


@dataclass
class RunInfo:
    run_id: str
    log_path: Path
    exact_val_bpb: float
    artifact_bytes: int | None
    artifact_compressor: str | None
    val_tokens: int

    @property
    def is_full_eval(self) -> bool:
        return self.val_tokens >= 10_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive a Parameter Golf run and optionally update the baseline docs.")
    parser.add_argument("run_id", help="Run id without .txt")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_run_info(repo_root: Path, run_id: str) -> RunInfo:
    log_path = repo_root / "logs" / f"{run_id}.txt"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log: {log_path}")
    text = log_path.read_text(encoding="utf-8")
    exact_match = EXACT_BPB_RE.search(text)
    if exact_match is None:
        raise ValueError(f"Missing exact roundtrip val_bpb in {log_path}")
    val_tokens_match = VAL_TOKENS_RE.search(text)
    if val_tokens_match is None:
        raise ValueError(f"Missing validation token count in {log_path}")
    artifact_match = ARTIFACT_RE.search(text)
    artifact_bytes: int | None = None
    artifact_compressor: str | None = None
    if artifact_match is not None:
        artifact_compressor = artifact_match.group(1)
        artifact_bytes = int(artifact_match.group(2))

    # Prefer the actual local artifact file if it exists. This keeps the docs
    # aligned with post-run rescues such as the lzma re-encode of an older zlib log.
    for compressor, ext in (("lzma", ".ptx"), ("zlib", ".ptz")):
        artifact_path = repo_root / "logs" / f"{run_id}_mlx_model.int8{ext}"
        if artifact_path.is_file():
            artifact_compressor = compressor
            artifact_bytes = artifact_path.stat().st_size
            break

    return RunInfo(
        run_id=run_id,
        log_path=log_path,
        exact_val_bpb=float(exact_match.group(1)),
        artifact_bytes=artifact_bytes,
        artifact_compressor=artifact_compressor,
        val_tokens=int(val_tokens_match.group(1)),
    )


def archive_log(run: RunInfo, repo_root: Path, dry_run: bool) -> str:
    dest = repo_root / "results" / run.log_path.name
    if dry_run:
        return f"would archive {run.log_path.name} -> {dest.relative_to(repo_root)}"
    shutil.copy2(run.log_path, dest)
    return f"archived {run.log_path.name} -> {dest.relative_to(repo_root)}"


def render_archived_report(run: RunInfo, repo_root: Path, dry_run: bool) -> str:
    archived_log_path = repo_root / "results" / run.log_path.name
    report_dir = repo_root / "results" / "reports" / run.run_id
    if dry_run:
        return (
            f"would render archived report bundle -> {report_dir.relative_to(repo_root)} "
            f"(artifact_policy=log-only)"
        )
    if not archived_log_path.is_file():
        raise FileNotFoundError(f"Missing archived log for report generation: {archived_log_path}")
    report = build_report(archived_log_path, artifact_policy="log-only")
    write_report_bundle(report, report_dir)
    return f"rendered archived report bundle -> {report_dir.relative_to(repo_root)}"


def update_readme(run: RunInfo, repo_root: Path, dry_run: bool) -> str:
    if not run.is_full_eval:
        return "skipped README update (smoke run)"
    path = repo_root / "README.md"
    text = path.read_text(encoding="utf-8")
    match = README_BEST_RE.search(text)
    if match is None:
        raise ValueError("Could not find current-best section in README.md")
    current_val = float(match.group(3))
    if run.exact_val_bpb >= current_val:
        return f"skipped README update ({run.exact_val_bpb:.8f} >= current best {current_val:.8f})"
    artifact_phrase = "unknown"
    if run.artifact_bytes is not None and run.artifact_compressor is not None:
        artifact_phrase = f"{run.artifact_bytes:,} bytes via `{run.artifact_compressor}`"
    replacement = (
        f"{match.group(1)}"
        f"- run: `{run.run_id}`\n"
        f"- exact `val_bpb = {run.exact_val_bpb:.8f}`\n"
        f"- compressed artifact size: `{artifact_phrase}`\n"
        f"- hardware: {match.group(5)}\n"
    )
    if not dry_run:
        path.write_text(text[: match.start()] + replacement + text[match.end() :], encoding="utf-8")
    return f"updated README current-best section -> {run.run_id}"


def update_research_log(run: RunInfo, repo_root: Path, dry_run: bool) -> str:
    if not run.is_full_eval:
        return "skipped research-log baseline update (smoke run)"
    path = repo_root / "docs" / "parameter-golf-research-log.md"
    text = path.read_text(encoding="utf-8")
    exact_line_match = re.search(r"- exact `val_bpb = ([0-9.]+)`", text)
    if exact_line_match is None:
        raise ValueError("Could not find current-best val_bpb in research log")
    current_val = float(exact_line_match.group(1))
    if run.exact_val_bpb >= current_val:
        return f"skipped research-log baseline update ({run.exact_val_bpb:.8f} >= current best {current_val:.8f})"
    start = text.find(RESEARCH_START)
    end = text.find(RESEARCH_END)
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find current-best block in research log")
    block = [
        RESEARCH_START.rstrip(),
        "",
        f"- `{run.run_id}`",
        f"- exact `val_bpb = {run.exact_val_bpb:.8f}`",
    ]
    if run.artifact_bytes is not None and run.artifact_compressor is not None:
        block.append(f"- compressed artifact size: `{run.artifact_bytes:,}` bytes via `{run.artifact_compressor}`")
    new_text = (
        text[:start]
        + "\n".join(block)
        + text[end:]
    )
    new_text = re.sub(r"Last updated: \d{4}-\d{2}-\d{2}", f"Last updated: {date.today().isoformat()}", new_text, count=1)
    new_text = re.sub(
        r"As of `\d{4}-\d{2}-\d{2}`, the work has split into two distinct tracks:",
        f"As of `{date.today().isoformat()}`, the work has split into two distinct tracks:",
        new_text,
        count=1,
    )
    if not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return f"updated research-log current-best block -> {run.run_id}"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run = read_run_info(repo_root, args.run_id)
    actions = [
        archive_log(run, repo_root, args.dry_run),
        render_archived_report(run, repo_root, args.dry_run),
        update_readme(run, repo_root, args.dry_run),
        update_research_log(run, repo_root, args.dry_run),
    ]
    print(f"run_id={run.run_id}")
    print(f"exact_val_bpb={run.exact_val_bpb:.8f}")
    print(f"artifact_bytes={run.artifact_bytes}")
    print(f"artifact_compressor={run.artifact_compressor}")
    print(f"val_tokens={run.val_tokens}")
    for action in actions:
        print(action)


if __name__ == "__main__":
    main()
