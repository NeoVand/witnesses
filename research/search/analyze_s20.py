"""S20 scoring: per run, convergence, accuracy at the scored K = ceil(log2 n) + 2, best K, reach.
python3 analyze_s20.py [run ids]; reads results/<id>/rows.json and sizes_s<seed>.json; prints one line per run and size
and writes results/s20_summary.json."""
import json, math, sys
from statistics import mean
from pathlib import Path
R = Path(__file__).resolve().parent / 'results'
RUNS = sys.argv[1:] or [f's20-loop{a}-v49-s{s}' for a in ('newton', 'pmean') for s in (0, 1, 2)]


def reach(row, thr=.99):
    """Largest lookahead L* such that every lookahead 1..L* is answered at least thr."""
    looks = sorted(int(k) for k in row if k != 'None')
    best = 0
    for L in looks:
        if row[str(L)] < thr:
            break
        best = L
    return best


out = {}
for run in RUNS:
    d = R / run
    if not (d / 'rows.json').exists():
        print(run, 'missing')
        continue
    rows = json.loads((d / 'rows.json').read_text())
    r = rows[0]
    sd = r['seed']
    sizes = json.loads((d / f'sizes_s{sd}.json').read_text())
    res = dict(converged_at=r['converged_at'], params=r['params'], train_seconds=round(r['seconds']), sizes={})
    for V, byK in sizes.items():
        Kreg = math.ceil(math.log2(int(V))) + 2
        row = byK[str(Kreg)]
        looks = [k for k in row if k != 'None']
        means = {K: mean(byK[K][k] for k in looks) for K in byK}
        Kbest = max(means, key=means.get)
        res['sizes'][V] = dict(K=Kreg, mean=mean(row[k] for k in looks), worst=min(row[k] for k in looks),
                               random=row.get('None'), best_K=int(Kbest), best_mean=means[Kbest],
                               reach99_by_K=[reach(byK[str(K)]) for K in range(1, len(byK) + 1)])
        s = res['sizes'][V]
        print(f"{run} n={V}: K={Kreg} mean {s['mean']:.4f} worst {s['worst']:.3f} random {s['random']:.3f} | best K {s['best_K']} "
              f"mean {s['best_mean']:.4f} | reach(K=1..8) {s['reach99_by_K'][:8]}")
    print(f"{run}: converged_at {res['converged_at']} params {res['params']} train {res['train_seconds']} s")
    out[run] = res
(R / f"{RUNS[0].split('-')[0]}_summary.json").write_text(json.dumps(out, indent=1))  # s20_summary.json, s21_summary.json, ...
