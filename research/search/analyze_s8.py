"""Score S8 from results/s8-*/rows.json. python analyze_s8.py"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
TOP = {49: 24, 75: 37, 100: 49, 125: 62}


def lstar(acc, top, bar=.99):
    l = 0
    for L in range(1, top + 1):
        if acc[str(L)] < bar:
            break
        l = L
    return l


def mean(acc, lo, hi):
    v = [acc[str(L)] for L in range(lo, hi + 1)]
    return sum(v) / len(v)


def load(arm):
    out = {}
    for sd in (0, 1, 2):
        f = HERE / 'results' / f's8-{arm}-v49-s{sd}' / 'rows.json'
        if f.exists():
            out[sd] = json.loads(f.read_text())[0]
    return out


loop, fixed = load('loop'), load('fixed8')
print('== loop')
for sd, r in loop.items():
    print(f'seed {sd}: converged_at {r["converged_at"]}, {r["seconds"]:.0f} s, {r["params"]} params')
    for V, top in TOP.items():
        by = r['sizes'][str(V)]
        ls = {int(K): lstar(a, top) for K, a in by.items()}
        ms = {int(K): round(mean(a, 1, top), 3) for K, a in by.items()}
        bestK = max(ms, key=ms.get)
        print(f'  V={V}: l*(K) {[ls[K] for K in sorted(ls)]}')
        print(f'         mean  {[ms[K] for K in sorted(ms)]}  best K={bestK} mean {ms[bestK]} random {by[str(bestK)]["None"]:.3f}')
print('== fixed8')
for sd, r in fixed.items():
    print(f'seed {sd}: converged_at {r["converged_at"]}, {r["seconds"]:.0f} s, {r["params"]} params')
    for V, top in TOP.items():
        a = r['sizes'][str(V)]['8']
        print(f'  V={V}: l* {lstar(a, top)}, mean {mean(a, 1, top):.3f}, beyond 24 {mean(a, 25, top) if top > 24 else float("nan"):.3f}, random {a["None"]:.3f}')

print('== predictions')
i = [sd for sd, r in loop.items() if r['converged_at'] and min(r['sizes']['49']['8'][str(L)] for L in range(1, 25)) >= .99]
print('(i) converged and >= .99 at every lookahead at 49 with K=8:', i, 'PASS' if len(i) >= 2 else 'FAIL')
ok, strong = [], []
for sd, r in loop.items():
    def best(V, f):
        return max(f(a) for a in r['sizes'][str(V)].values())
    if all(best(V, lambda a, V=V: mean(a, 1, TOP[V])) >= .95 for V in (100, 125)):
        ok.append(sd)
    if all(any(lstar(a, TOP[V]) == TOP[V] for a in r['sizes'][str(V)].values()) for V in (100, 125)):
        strong.append(sd)
print('(ii) pass seeds', ok, 'strong seeds', strong, 'PASS' if len(ok) >= 2 else 'FAIL', 'STRONG' if len(strong) >= 2 else '')
fx = max(mean(r['sizes']['125']['8'], 25, 62) for r in fixed.values()) if fixed else float('nan')
grow, beat = [], []
for sd, r in loop.items():
    by = r['sizes']['125']
    ls = [lstar(by[str(K)], 62) for K in range(1, 17)]
    grow.append(ls)
    beat.append(round(max(mean(a, 25, 62) for a in by.values()) - fx, 3))
print('(iii) l*(K) at 125:', grow, '; best mean 25-62 minus best fixed8:', beat)
