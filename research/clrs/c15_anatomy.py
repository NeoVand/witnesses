"""Anatomy of mst_prim pointer errors from saved predictions, official 64-node test file, K = 8, pooled over seeds 0
to 2 (CPU). Orientation error: node v points to u with pi[u] = v (its own child: the right tree
edge, the wrong way). Depth: hops from s along the true pointers (s at 0). Error rates are over all nodes, as the
library scores them (the root included). Also depth bins over non-root nodes (0-7, 8-15, 16-23, 24+; key c14_bins).
python3 c15_anatomy.py <run prefix> [K]  (c13, c15 or c16; reads results/<prefix>-mst_prim-s<seed>/preds_s<seed>.npz and
data_local/mst_prim/test.npz; writes results/c15_anatomy_<prefix>.json)"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


def depth(pi, n):
    d = np.zeros(n, int)
    for v in range(n):
        u, k = v, 0
        while pi[u] != u:
            u, k = pi[u], k + 1
        d[v] = k
    return d


def main(prefix, K=8):
    K = int(K)
    d = {k: v for k, v in np.load(HERE / 'data_local' / 'mst_prim' / 'test.npz').items()}
    out = dict(prefix=prefix, K=K, seeds={})
    pool = dict(err=0, nodes=0, orient=0, nonroot_err=0, le4=[0, 0], ge8=[0, 0], c14_bins={b: [0, 0] for b in ('0-7', '8-15', '16-23', '24+')})
    for sd in (0, 1, 2):
        pr = np.load(HERE / 'results' / f'{prefix}-mst_prim-s{sd}' / f'preds_s{sd}.npz')['test/pi'][K - 1].astype(int)
        e = o = 0
        for g in range(len(d['n'])):
            n = int(d['n'][g]); t = d['pi'][g, :n].astype(int); p = pr[g, :n]
            dep = depth(t, n)
            for v in range(n):
                wrong = p[v] != t[v]
                e += wrong
                o += bool(wrong and p[v] < n and t[p[v]] == v and p[v] != v)
                pool['nodes'] += 1
                if dep[v] <= 4:
                    pool['le4'][0] += wrong; pool['le4'][1] += 1
                if dep[v] >= 8:
                    pool['ge8'][0] += wrong; pool['ge8'][1] += 1
                if t[v] != v:
                    pool['nonroot_err'] += wrong
                    b = '0-7' if dep[v] <= 7 else '8-15' if dep[v] <= 15 else '16-23' if dep[v] <= 23 else '24+'
                    pool['c14_bins'][b][0] += wrong; pool['c14_bins'][b][1] += 1
        out['seeds'][sd] = dict(errors=int(e), orientation=int(o), score=round(100 * (1 - e / (len(d['n']) * 64)), 3))
        pool['err'] += int(e); pool['orient'] += int(o)
    pool['rate_le4'] = round(pool['le4'][0] / pool['le4'][1], 4)
    pool['rate_ge8'] = round(pool['ge8'][0] / pool['ge8'][1], 4)
    pool['c14_rates'] = {b: (round(v[0] / v[1], 4) if v[1] else None) for b, v in pool['c14_bins'].items()}
    out['pooled'] = {k: (v if not isinstance(v, list) else [int(x) for x in v]) for k, v in pool.items()}
    print(json.dumps(out, default=int))
    (HERE / 'results' / f'c15_anatomy_{prefix}.json').write_text(json.dumps(out, indent=1, default=int))


if __name__ == '__main__':
    main(*sys.argv[1:])
