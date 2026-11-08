"""Chess numbers from results/c2[456]-*/rows.json and results/c24-data/result.json; prints one JSON object.

First-move accuracy on the 50,000 test puzzles (one position each, ply 0), in points, for every arm and data fraction
(all data, 10%, 1%), relative to 8-layer attention at the same data (mean over its seeds); first-move accuracy on the
test puzzles tagged pin, skewer, xRayAttack or discoveredAttack (by_theme '_line'); parameter counts.
    python3 analyze_chess.py
"""
import json
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent
ARMS = dict(sa8='attention, 8 layers', cub8='joins replacing attention in blocks 3 and 6',
            hyb8c='joins beside attention in blocks 3 and 6, connected output', hyb8r='joins beside attention, plain output',
            hyb8z='joins beside attention, plain output, centred keys')
FRACTIONS = {'full': 1.0, 'p10': 0.1, 'p1': 0.01}


def rows():
    """[(arm, fraction, row)] for every saved run."""
    out = []
    for d in sorted((HERE / 'results').glob('c2[456]-*')):
        m = re.match(r'c2\d-(?:(full|p1|p10)-)?([a-z0-9]+?)(?:-s\d)?$', d.name)
        if not (d / 'rows.json').exists() or not m or m.group(2) not in ARMS:
            continue
        for r in json.loads((d / 'rows.json').read_text()):
            out.append((m.group(2), FRACTIONS[m.group(1) or 'full'], dict(r, run=d.name)))
    return out


def main():
    R = rows()
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    first = {}
    for arm, f, r in R:
        first.setdefault(arm, {}).setdefault(f, []).append(100 * r['test']['first'])
    base = {f: mean(v) for f, v in first['sa8'].items()}
    rel = {a: {f: mean(v) - base[f] for f, v in fs.items()} for a, fs in first.items() if a != 'sa8'}
    joins = [x for a, fs in rel.items() for x in fs.values()]
    line = {}
    for arm, f, r in R:
        if f == 1.0:
            acc, n = r['test']['by_theme']['_line']
            line.setdefault(arm, []).append(100 * acc)
            line_n = n
    data = json.loads((HERE / 'results' / 'c24-data' / 'result.json').read_text())['result']
    out = dict(
        arms=ARMS,
        first_move={a: {str(f): v for f, v in sorted(fs.items(), reverse=True)} for a, fs in first.items()},
        relative_to_sa8={a: {str(f): v for f, v in sorted(fs.items(), reverse=True)} for a, fs in rel.items()},
        joins_relative_range=[min(joins), max(joins)],
        sa8_seed_spread_full=max(first['sa8'][1.0]) - min(first['sa8'][1.0]),
        line_theme_full=dict(puzzles=line_n, by_arm=line, sa8_mean=mean(line['sa8']),
                             joins_beside=[min(x for a in ('hyb8c', 'hyb8r', 'hyb8z') for x in line[a]),
                                           max(x for a in ('hyb8c', 'hyb8r', 'hyb8z') for x in line[a])]),
        params={a: sorted({r['params'] for arm, _, r in R if arm == a}) for a in ARMS},
        updates={a: {str(f): sorted({r['steps'] for arm, f2, r in R if arm == a and f2 == f}) for f in FRACTIONS.values()
                     if any(arm == a and f2 == f for arm, f2, _ in R)} for a in ARMS},
        train_examples={str(f): sorted({r['train_examples'] for _, f2, r in R if f2 == f}) for f in FRACTIONS.values()},
        data={k: data[k] for k in ('puzzles', 'train_puzzles', 'val_puzzles', 'test_puzzles', 'train_examples', 'val_examples', 'test_examples')},
        runs=sorted({r['run'] for _, _, r in R}))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
