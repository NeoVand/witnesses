"""Score S15 runs as they arrive. python s15_score.py"""
import json
from pathlib import Path
TOP = {49: 24, 125: 62, 150: 74, 200: 99, 250: 124}
for arm in ('tight256', 'tightmix256'):
    for s in (0, 1, 2):
        f = Path(f'results/s15-{arm}-v49-s{s}/rows.json')
        if not f.exists():
            continue
        r = json.loads(f.read_text())[0]
        print(arm, s, 'conv', r['converged_at'])
        for V in (49, 125, 150, 200, 250):
            d = r['sizes'][str(V)]; top = TOP[V]
            means = {K: sum(d[str(K)][str(L)] for L in range(1, top + 1)) / top for K in range(1, 17)}
            okK = [K for K in range(1, 17) if min(d[str(K)][str(L)] for L in range(1, top + 1)) >= .99]
            reach = []
            for K in range(1, 9):
                l = 0
                for L in range(1, top + 1):
                    if d[str(K)][str(L)] < .95:
                        break
                    l = L
                reach.append(l)
            b = max(means, key=means.get)
            print(f'   V={V}: best mean {means[b]:.3f} (K={b}) K all>=.99 {okK[:1]}..{okK[-1:]} ({len(okK)}) reach95 K1-8 {reach} random K12 {d["12"]["None"]}')
