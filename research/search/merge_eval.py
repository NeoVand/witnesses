"""Random-graph errors by merging paths and by step : pair state with the sum over the middle vertex
(s2-pairn-v49) against triangle attention (s4-pairt-v49), both eight layers over names at 49 vertices.
python merge_eval.py <their train.py> <out.json>"""
import json, sys, random
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st, l2s_eval

V = 49
n, pad, vocab = 3 * V + 5, V + 3, V + 5
ARMS = {'sum': ('s2-pairn-v49-s{}', dict(kind='pairn', emb=32, D=16, layers=8, r=8, bf16=False, remat=False)),
        'tri': ('s4-pairt-v49-s{}', dict(kind='pairn', emb=32, D=16, layers=8, r=8, tri='attn', bf16=False, remat=False))}


def restore(cfg, path):
    p = st.init(jax.random.PRNGKey(0), cfg, vocab, n)
    P = dict(np.load(path))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    return jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])


def features(row):
    r = [int(t) for t in row if t != pad]
    edges = [(r[i + 1], r[i + 2]) for i in range(len(r)) if r[i] == V + 2]
    parents = {}
    for u, v in edges:
        parents[v] = parents.get(v, 0) + 1
    merges = sum(1 for c in parents.values() if c >= 2)
    step = len(r) - r.index(V + 1) - 1
    return merges, step


def main(train_py, out):
    ns = l2s_eval.load(train_py)
    random.seed(4321); np.random.seed(4321)
    x, y = ns['generate_eval_data'](n, min_path_length=2, distance_from_start=-1, distance_from_end=-1, lookahead_steps=None,
                                    num_paths_at_fork=None, num_samples=3000, max_prefix_vertices=None)
    feats = np.array([features(row) for row in x])
    res = dict(n=len(x), merges_hist={int(k): int(v) for k, v in zip(*np.unique(np.minimum(feats[:, 0], 6), return_counts=True))})
    for arm, (pat, cfg) in ARMS.items():
        fwd = jax.jit(lambda p, xx, cfg=cfg: st.forward(p, cfg, xx, pad)[1])
        for s in (0, 1, 2):
            path = Path(__file__).resolve().parent / 'results' / pat.format(s) / f'params_s{s}.npz'
            p = restore(cfg, path)
            pred = np.concatenate([np.asarray(fwd(p, j.array(x[i:i + 100]))).argmax(-1) for i in range(0, len(x), 100)])
            ok = pred == y
            m = np.minimum(feats[:, 0], 6)
            first = feats[:, 1] == 1
            row = dict(acc=float(ok.mean()),
                       by_merges={int(k): [float(ok[m == k].mean()), int((m == k).sum())] for k in np.unique(m)},
                       first=[float(ok[first].mean()), int(first.sum())], later=[float(ok[~first].mean()), int((~first).sum())],
                       first_no_merge=float(ok[first & (m == 0)].mean()), first_merge3=float(ok[first & (m >= 3)].mean()),
                       later_no_merge=float(ok[~first & (m == 0)].mean()), later_merge3=float(ok[~first & (m >= 3)].mean()))
            res[f'{arm}-s{s}'] = row
            print(arm, s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'by_merges'},
                  {k: round(v[0], 2) for k, v in row['by_merges'].items()}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
