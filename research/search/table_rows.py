"""Table 1 rows for the looped layers (S8, S9, S11, S12) and their eight-layer control. python table_rows.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
TOP = {49: 24, 75: 37, 100: 49, 125: 62, 150: 74, 200: 99, 250: 124}


def rows(arm):
    out = []
    for s in (0, 1, 2):
        f = HERE / 'results' / f'{arm}-v49-s{s}' / 'rows.json'
        if f.exists():
            out.append(json.loads(f.read_text())[0])
    return out


def fmt(v):
    return f"{min(v) * 100:.1f}–{max(v) * 100:.1f}%" if len(v) > 1 else f"{v[0] * 100:.1f}%"


for arm, K in (('s8-fixed8', '8'), ('s8-loop', '8'), ('s9-looptri', '8'), ('s9-looptri', '12'), ('s12-looptrimix', '8'),
               ('s12-looptrimix', '12'), ('s11-looptri256', '8'), ('s11-looptri256', '12')):
    rr = rows(arm)
    if not rr:
        continue
    conv = [r['converged_at'] for r in rr]
    print(arm, 'K', K, 'converged', sum(c is not None for c in conv), 'of', len(rr), 'at', [c and round(c / 1e6, 2) for c in conv])
    for V in sorted(int(v) for v in rr[0]['sizes']):
        top = TOP[V]
        worst = [min(r['sizes'][str(V)][K][str(L)] for L in range(1, top + 1)) for r in rr if K in r['sizes'][str(V)]]
        rand = [r['sizes'][str(V)][K]['None'] for r in rr if K in r['sizes'][str(V)]]
        mean = [sum(r['sizes'][str(V)][K][str(L)] for L in range(1, top + 1)) / top for r in rr if K in r['sizes'][str(V)]]
        print(f'   V={V}: worst {fmt(worst)}  mean {fmt(mean)}  random {fmt(rand)}')
