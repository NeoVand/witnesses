"""S22 scoring: per run, convergence and the mean over lookaheads at every size (no K to choose);
the pilot's rule. python3 analyze_s22.py [run ids]; reads results/<id>/rows.json, sizes_s<seed>.json, progress.jsonl;
prints one line per run and size and writes results/s22_summary.json."""
import json, sys
from statistics import mean
from pathlib import Path
R = Path(__file__).resolve().parent / 'results'
RUNS = sys.argv[1:] or sorted(p.name for p in R.glob('s22-*') if (p / 'rows.json').exists())


def reach(row, thr=.99):
    """Largest L* such that every lookahead 1..L* is answered at least thr."""
    best = 0
    for L in sorted(int(k) for k in row if k != 'None'):
        if row[str(L)] < thr:
            break
        best = L
    return best


out = {}
for run in RUNS:
    d = R / run
    r = json.loads((d / 'rows.json').read_text())[0]
    prog = [json.loads(l) for l in (d / 'progress.jsonl').read_text().splitlines()]
    acc = [p['train_acc'] for p in prog]
    first90 = next((prog[i]['examples'] for i in range(1, len(acc)) if acc[i] >= .9 and acc[i - 1] >= .9), None)
    sizes = json.loads((d / f"sizes_s{r['seed']}.json").read_text())
    res = dict(converged_at=r['converged_at'], first_two_checks_at_0p9=first90, params=r['params'], train_seconds=round(r['seconds']),
               final_train_acc=acc[-1], sizes={})
    for V, byL in sizes.items():
        row = byL[str(r['cfg']['layers'])]
        looks = [k for k in row if k != 'None']
        res['sizes'][V] = dict(mean=mean(row[k] for k in looks), worst=min(row[k] for k in looks),
                               worst_lookahead=int(min(looks, key=lambda k: row[k])), random=row.get('None'), reach99=reach(row))
        s = res['sizes'][V]
        print(f"{run} n={V}: mean {s['mean']:.4f} worst {s['worst']:.3f} (L={s['worst_lookahead']}) random {s['random']:.3f} reach99 {s['reach99']}")
    print(f"{run}: converged_at {res['converged_at']} two checks >= 0.9 at {first90} final train acc {acc[-1]:.3f} "
          f"params {res['params']} train {res['train_seconds']} s")
    out[run] = res
(R / 's22_summary.json').write_text(json.dumps(out, indent=1))
