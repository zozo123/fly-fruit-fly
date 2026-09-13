"""Verify published flight evidence using only the Python standard library."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("assessment", ROOT / "src/fly_fruit_fly/assessment.py")
assessment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessment)

def check(condition, message):
    if not condition:
        raise ValueError(message)

for name, expected in (("expert", True), ("superfly", False)):
    folder = ROOT / "results" / ("2026-09-12-" + name)
    provenance = json.loads((folder / "provenance.json").read_text())
    for filename, digest in provenance["files"].items():
        check(hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest,
              f"{name}/{filename}: checksum mismatch")
    metrics = json.loads((folder / "metrics.json").read_text())
    recorded = json.loads((folder / "assessment.json").read_text())
    actual = assessment.assess(metrics)
    check(actual == recorded, f"{name}: assessment differs from raw metrics")
    check(actual["task_gate_passed"] is expected, f"{name}: unexpected gate outcome")
    print(f"{name}: {actual['passing_episodes']}/{actual['episodes']} pass; evidence verified")

media = json.loads((ROOT / "media/provenance.json").read_text())
for filename, digest in media["files"].items():
    check(hashlib.sha256((ROOT / "media" / filename).read_bytes()).hexdigest() == digest,
          f"media/{filename}: checksum mismatch")
print("All published media checksums verified.")

