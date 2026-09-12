"""Check recorded evidence and recompute the README numbers using stdlib only."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/2026-09-12-smoke'

def main():
    provenance = json.loads((RESULTS / 'provenance.json').read_text())
    for name, digest in provenance['sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f'Hash mismatch: {name}')
    for name in ['baseline', 'ppo']:
        report = json.loads((RESULTS / f'{name}.json').read_text())
        episodes = report['episodes']
        assert [e['seed'] for e in episodes] == [10000, 10001, 10002]
        assert len(episodes) == 3 and all(e['failed'] for e in episodes)
        assert not any(e['completed_reference'] for e in episodes)
        assert abs(report['mean_return'] - mean(e['return'] for e in episodes)) < 1e-8
        print(name, json.dumps({
            'mean_duration_ms': 1000 * mean(e['duration_s'] for e in episodes),
            'mean_return': mean(e['return'] for e in episodes),
            'mean_episode_tracking_error_cm': mean(e['mean_tracking_error_cm'] for e in episodes),
            'completed': 0, 'failed': 3,
        }))
    training = json.loads((RESULTS / 'training.json').read_text())
    assert training['actual_steps'] == 8192 and training['seed'] == 0
    print(f"Verified {len(provenance['sha256'])} file hashes and recorded metrics.")

if __name__ == '__main__':
    main()
