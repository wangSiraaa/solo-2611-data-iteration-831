import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, *args):
    cmd = [
        sys.executable, "-m", "prepquant.cli",
        "--ledger", str(tmp_path / "l.db"),
        "--vault", str(tmp_path / "v.db"),
        *args,
    ]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


LOAD_V1 = {
    "operations": [{"op": "sample", "container": "RAW", "volume": "2", "sample": "s1"}],
    "densities": {"RAW": "1"},
    "solvent_density": "1",
    "fills": {"RAW": {"bottle": {"name": "tall", "chambers": [{"height": "8", "area": "1/2"}]}}},
}

LOAD_V2 = {
    "operations": [{"op": "sample", "container": "RAW", "volume": "2", "sample": "s1"}],
    "densities": {"RAW": "1"},
    "solvent_density": "1",
    "fills": {"RAW": {"bottle": {"name": "wide", "chambers": [{"height": "8", "area": "1"}]}}},
}

RACK = {
    "name": "R",
    "holes": [{"name": "h", "x": "0", "height": "0", "allowed_weights": ["2"]}],
    "manifest": "manifest.json",
    "positions": {"RAW": {"x": "0", "height": "0"}},
    "weights": [{"name": "w1", "mass": "2"}],
    "conditions": [{"name": "c1", "k_lo": "0", "k_hi": "1", "a": "-1", "b": "1"}],
}


def test_cli_bottle_load_and_manifest_revision(tmp_path):
    spec = tmp_path / "v1.json"
    spec.write_text(json.dumps(LOAD_V1), encoding="utf-8")
    out1 = tmp_path / "m1.json"
    r = run(tmp_path, "bottle-load", str(spec), "--out", str(out1))
    assert r.returncode == 0, r.stderr
    m1 = json.loads(out1.read_text())
    assert m1["version"] == 1
    assert m1["bottles"]["RAW"]["load"]["total_moment"] == "4"

    spec2 = tmp_path / "v2.json"
    spec2.write_text(json.dumps(LOAD_V2), encoding="utf-8")
    out2 = tmp_path / "m2.json"
    r = run(tmp_path, "bottle-load", str(spec2), "--out", str(out2), "--previous", str(out1),
            "--note", "corrected")
    assert r.returncode == 0, r.stderr
    m2 = json.loads(out2.read_text())
    assert m2["version"] == 2
    assert m2["previous"]["version"] == 1
    assert m2["previous"]["manifest_hash"] == m1["manifest_hash"]
    assert m2["bottles"]["RAW"]["load"]["total_moment"] == "2"

    # manifest-show reads the stored manifest independently
    r = run(tmp_path, "manifest-show", str(out2))
    assert r.returncode == 0, r.stderr
    view = json.loads(r.stdout)
    assert view["version"] == 2
    assert view["bottles"]["RAW"]["total_moment"] == "2"


def test_cli_rack_solve_uses_manifest(tmp_path):
    spec = tmp_path / "v1.json"
    spec.write_text(json.dumps(LOAD_V1), encoding="utf-8")
    out = tmp_path / "manifest.json"
    assert run(tmp_path, "bottle-load", str(spec), "--out", str(out)).returncode == 0
    rack = tmp_path / "rack.json"
    rack.write_text(json.dumps(RACK), encoding="utf-8")
    r = run(tmp_path, "rack-solve", str(rack))
    assert r.returncode == 0, r.stderr
    sol = json.loads(r.stdout)
    assert sol["feasible"] is True
    assert sol["added_mass"] == "2"
    assert sol["placements"][0]["hole"] == "h"
    assert sol["landing"]["c1"]["landing_at_k_hi"] == "1"

    # corrected (wide) manifest: no weight needed
    spec2 = tmp_path / "v2.json"
    spec2.write_text(json.dumps(LOAD_V2), encoding="utf-8")
    out2 = tmp_path / "manifest2.json"
    assert run(tmp_path, "bottle-load", str(spec2), "--out", str(out2),
               "--previous", str(out)).returncode == 0
    rack_b = dict(RACK)
    rack_b["manifest"] = "manifest2.json"
    rack2 = tmp_path / "rack2.json"
    rack2.write_text(json.dumps(rack_b), encoding="utf-8")
    r = run(tmp_path, "rack-solve", str(rack2))
    assert r.returncode == 0, r.stderr
    sol = json.loads(r.stdout)
    assert sol["feasible"] is True and sol["added_mass"] == "0"


def test_cli_infeasible_rack_reported(tmp_path):
    spec = tmp_path / "v1.json"
    spec.write_text(json.dumps(LOAD_V1), encoding="utf-8")
    out = tmp_path / "manifest.json"
    assert run(tmp_path, "bottle-load", str(spec), "--out", str(out)).returncode == 0
    rack = dict(RACK)
    rack["conditions"] = [{"name": "c1", "k_lo": "0", "k_hi": "1", "a": "5", "b": "6"}]
    rack_path = tmp_path / "rack.json"
    rack_path.write_text(json.dumps(rack), encoding="utf-8")
    r = run(tmp_path, "rack-solve", str(rack_path))
    assert r.returncode == 0, r.stderr
    sol = json.loads(r.stdout)
    assert sol["feasible"] is False
    assert sol["added_mass"] is None
    assert sol["reason"]


def test_offline_demo_runs():
    r = subprocess.run(
        [sys.executable, "-m", "prepquant.demo_offline"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "offline demo OK" in r.stdout
    # every hand-checked assertion inside the demo passed
    assert r.stdout.count("  - ") >= 7

