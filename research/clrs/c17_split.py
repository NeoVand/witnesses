"""The SCC split by K on the official test file and on the fresh draw of seed 20,261,109, from the
saved predictions. Per seed and K: overall accuracy, accuracy where the representative is the component's smallest index
and where DFS order fixes it, and wrong pointers inside the true component. The official numbers repeat diagnose_c3.py's;
the fresh draw is regenerated with the library (dm-clrs 2.0.3) as score_fresh.py does and sha256-checked.
results/c17_split_c13.json: python -I c17_split.py results c13 8,16 results/c17_split_c13.json
results/c17_split_c17.json: python -I c17_split.py results c17 8,34 results/c17_split_c17.json
python -I c17_split.py <results dir> <run prefix> <K list> [out json]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_data

TASK, SEED = 'strongly_connected_components', 20261109


def split(t, pr, n):
    m = np.arange(t.shape[1])[None] < n[:, None]
    mins = np.array([[np.nonzero(t[g] == t[g, v])[0].min() if m[g, v] else -1 for v in range(t.shape[1])] for g in range(len(t))])
    ok, w = (pr == t) & m, (pr != t) & m
    inside = np.take_along_axis(t, np.clip(pr, 0, None), 1) == t
    a, b = m & (t == mins), m & (t != mins)
    return dict(acc=100 * ok.sum() / m.sum(), min_index=100 * ok[a].mean(), dfs_fixed=100 * ok[b].mean(),
                n_min=int(a.sum()), n_dfs=int(b.sum()), wrong=int(w.sum()), wrong_inside=int((w & inside).sum()))


def main(results, prefix, Ks, out=None):
    results, Ks = Path(results), [int(k) for k in Ks.split(',')]
    test = np.load(HERE / 'data_local' / TASK / 'test.npz')
    fb = gen_data.sample(TASK, 128, 64, SEED)
    fresh = gen_data.pack(fb, 64)
    man = json.loads((results / 'data' / TASK / f'fresh_L{SEED}_L128.json').read_text())
    assert man['sha256'] == gen_data.digest(fresh), 'fresh graphs differ from the Modal manifest'
    res = {}
    for sd in (0, 1, 2):
        pz = np.load(results / f'{prefix}-{TASK}-s{sd}' / f'preds_s{sd}.npz')
        for name, d, key in (('test', test, 'test/scc_id'), ('fresh', fresh, f'fresh{SEED}/scc_id')):
            P = pz[key].astype(int)
            for K in Ks:
                r = split(d['scc_id'].astype(int), P[K - 1][:, :d['scc_id'].shape[1]], d['n'].astype(int))
                res[f'{name}/s{sd}/K{K}'] = r
                print(prefix, name, sd, K, ' '.join(f'{k} {v:.2f}' if isinstance(v, float) else f'{k} {v}' for k, v in r.items()), flush=True)
    for name in ('test', 'fresh'):
        for K in Ks:
            rs = [res[f'{name}/s{sd}/K{K}'] for sd in (0, 1, 2)]
            print(prefix, name, 'mean', K, ' '.join(f'{k} {np.mean([r[k] for r in rs]):.2f}' for k in ('acc', 'min_index', 'dfs_fixed')),
                  'wrong_inside', sum(r['wrong_inside'] for r in rs), '/', sum(r['wrong'] for r in rs), flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
