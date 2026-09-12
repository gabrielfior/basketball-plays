"""Tests for scripts/extract_trajectories.py's command line surface."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from basketball_plays.schema import FrameRecord

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_trajectories.py"


def load_script():
    spec = importlib.util.spec_from_file_location("extract_trajectories", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cli = load_script()


def test_team_map_resolves_a_named_team_and_gives_the_other_cluster_the_other_team():
    assert cli.resolve_team_map("auto", "Duke", "Michigan") is None
    assert cli.resolve_team_map("0=Duke", "Duke", "Michigan") == {0: "Duke", 1: "Michigan"}
    assert cli.resolve_team_map("1=Duke", "Duke", "Michigan") == {1: "Duke", 0: "Michigan"}
    assert cli.resolve_team_map("0=Michigan", "Duke", "Michigan") == {0: "Michigan", 1: "Duke"}
    # any game's teams, not just the reference one, and spacing/case do not matter
    assert cli.resolve_team_map("0 = north carolina", "North Carolina", "Duke") == {
        0: "North Carolina", 1: "Duke"}


def test_team_map_rejects_a_team_that_is_not_playing_in_this_game():
    with pytest.raises(SystemExit) as excinfo:
        cli.resolve_team_map("0=Kansas", "Duke", "Michigan")
    assert "Kansas" in str(excinfo.value) and "Duke" in str(excinfo.value)


def test_team_map_rejects_a_malformed_spec():
    for spec in ("Duke", "2=Duke", "0=", "=Duke"):
        with pytest.raises(SystemExit) as excinfo:
            cli.resolve_team_map(spec, "Duke", "Michigan")
        assert "--team-map" in str(excinfo.value)


def a_raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rec = FrameRecord(frame_idx=0, t=0.0, keypoints=[[0.0, 0.0, 0.0]] * 33, detections=[],
                      ball=None, numbers=[])
    (raw / "frames_00.jsonl").write_text(rec.to_json() + "\n")
    return raw


def run_cli(tmp_path, extra):
    raw = a_raw_dir(tmp_path)
    timeline = tmp_path / "scoreboard_raw.jsonl"
    timeline.write_text("")  # a timeline with no reads at all -> no period spans
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "video.mp4"), "--skip-gpu",
         "--raw-dir", str(raw), "--out", str(tmp_path / "trajectories.jsonl"),
         "--periods-from", str(timeline), *extra],
        capture_output=True, text=True, cwd=ROOT, check=False)


def test_extraction_fails_loudly_when_the_period_spans_are_empty(tmp_path):
    # Falling back silently would learn one offense map for a whole multi-period game.
    proc = run_cli(tmp_path, [])

    assert proc.returncode != 0
    assert "no period spans detected" in proc.stdout
    assert "scoreboard_raw.jsonl" in proc.stdout
    assert "--allow-no-periods" in proc.stderr


def test_allow_no_periods_falls_back_instead_of_failing(tmp_path):
    proc = run_cli(tmp_path, ["--allow-no-periods"])

    assert proc.returncode == 0, proc.stderr
    assert "falling back to the time-windowed offense rule" in proc.stdout
