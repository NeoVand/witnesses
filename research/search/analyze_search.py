"""One line per run of the search benchmark, from results/<id>/rows.json (Supplementary Table 1's columns: convergence,
examples, parameters, accuracy). python analyze_search.py <run id or prefix>... ; with --fetch first, the runs are
downloaded from the witnesses-search Modal volume (search/<id>) before they are read."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent


def fetch(id_):
    d = HERE / 'results' / id_
    d.mkdir(parents=True, exist_ok=True)
    for f in ('result.json', 'rows.json', 'progress.jsonl'):
        subprocess.run(['modal', 'volume', 'get', '--force', 'witnesses-search', f'search/{id_}/{f}', str(d / f)], capture_output=True)
    return d


args = sys.argv[1:]
get = bool(args) and args[0] == '--fetch'
args = args[1:] if get else args
ids = [a for a in args] if get else sorted({p.name for a in args for p in (HERE / 'results').glob(a + '*') if p.is_dir()})
for id_ in ids:
    d = fetch(id_) if get else HERE / 'results' / id_
    if True:
        if not (d / 'rows.json').exists():
            res = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else {}
            print(f'{id_:28s} not done', res.get('error', '')[-300:])
            continue
        for r in json.loads((d / 'rows.json').read_text()):
            last = r['curve'][-1] if r['curve'] else {}
            print(f"{id_:28s} s{r['seed']} V={r['vertices']} params={r['params']} GF/ex={r['flops'] / 1e9:.3f} ex={r['examples'] / 1e6:.2f}M conv={r['converged_at']} "
                  f"sec={r['seconds']:.0f} ex/s={r['examples'] / max(r['seconds'], 1):.0f} loss={last.get('loss', 0):.3f} train={last.get('train_acc', 0):.3f} test_mean={r['test_mean_lookahead']:.3f}")
