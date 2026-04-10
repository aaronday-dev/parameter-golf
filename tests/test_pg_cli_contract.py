from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_PATH = REPO_ROOT / "bin" / "pg"
DEFAULT_LOG = REPO_ROOT / "results" / "mlx_full_seq_mlp4x_200_realval_vb524k.txt"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BIN_PATH), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_pg_help_and_version_are_safe() -> None:
    help_proc = _run("--help")
    assert help_proc.returncode == 0, help_proc.stderr
    assert "canonical CLI noun" in help_proc.stdout
    assert "leaderboard" in help_proc.stdout
    assert "queue" in help_proc.stdout
    assert "report" in help_proc.stdout
    assert help_proc.stdout.startswith("usage: ./bin/pg ")

    leaderboard_help = _run("leaderboard", "--help")
    assert leaderboard_help.returncode == 0, leaderboard_help.stderr
    assert "--json" in leaderboard_help.stdout
    assert leaderboard_help.stdout.startswith("usage: ./bin/pg leaderboard")

    version_proc = _run("--version")
    assert version_proc.returncode == 0, version_proc.stderr
    assert version_proc.stdout.strip() == "pg 0.0.1"


def test_pg_leaderboard_and_queue_json() -> None:
    leaderboard = _run("leaderboard", "--json")
    assert leaderboard.returncode == 0, leaderboard.stderr
    leaderboard_payload = json.loads(leaderboard.stdout)
    assert leaderboard_payload["ok"] is True
    assert leaderboard_payload["command"] == "leaderboard"
    assert leaderboard_payload["result"]["repo_leader"]["run_id"] == "mlx_full_seq_mlp4x_resid64_block0proj_offline_realval_v1"

    queue = _run("queue", "--json")
    assert queue.returncode == 0, queue.stderr
    queue_payload = json.loads(queue.stdout)
    assert queue_payload["ok"] is True
    assert queue_payload["command"] == "queue"
    assert queue_payload["result"]["current_focus"]["primary_lane"] == "study_closed"
    assert "known_drift" in queue_payload["result"]


def test_pg_report_json_success() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = _run(
            "report",
            "--log",
            str(DEFAULT_LOG),
            "--output-dir",
            str(Path(tmpdir) / "report"),
            "--json",
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["ok"] is True
        assert payload["command"] == "report"
        assert payload["result"]["report"]["run"]["run_id"] == "mlx_full_seq_mlp4x_200_realval_vb524k"
        assert Path(payload["result"]["report_json"]).is_file()


def test_readme_and_agents_lock_canonical_pg_surface() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "./bin/pg --help" in readme
    assert "./bin/pg leaderboard --json" in readme
    assert "./bin/pg report --log results/mlx_full_seq_mlp4x_200_realval_vb524k.txt --output-dir /tmp/pg-report --json" in readme
    assert '{"ok":true,"command":"leaderboard","result":{...}}' in readme
    assert "state/current.yaml" in readme
    assert "python -m pip install pyyaml" in readme
    assert "bin/pg" in agents
