"""S1b: where does a converged pair model fail on random graphs?

Rebuilds each test example's graph from its tokens (`E u v` edges, `Q start goal`, `P` and the path so far), computes
its features with Saparov et al.'s own `lookahead_depth`, runs the saved model on CPU, and prints accuracy by feature
for the random-graph set (lookahead=None) and the balanced sets pooled.
python gap_analysis.py <params.npz> <their train.py> <vertices> <n per set>
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st, l2s_eval

CFG = dict(kind='pair', emb=32, D=16, layers=8, r=8, bf16=False, remat=False)


def load_params(path, cfg, vocab, n):
    p = st.init(jax.random.PRNGKey(0), cfg, vocab, n)
    P = dict(np.load(path))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    return jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])


def features(ns, row, V, answer):
    E, Q, P = V + 2, V + 4, V + 1
    toks = [int(t) for t in row if t != V + 3]
    q = toks.index(Q)
    edges = [(toks[i + 1], toks[i + 2]) for i in range(0, q, 3) if toks[i] == E]
    start, goal = toks[q + 1], toks[q + 2]
    path = toks[q + 4:]
    nodes = {}
    node = lambda i: nodes.setdefault(i, ns['Node'](i))
    for u, v in edges:
        node(u).children.append(node(v)); node(v).parents.append(node(u))
    cur = node(path[-1])
    look = ns['lookahead_depth'](cur, node(answer), node(goal))

    def reach(a, b, seen=None):
        seen = seen if seen is not None else set()
        if a == b:
            return True
        seen.add(a.id)
        return any(reach(c, b, seen) for c in a.children if c.id not in seen)
    useful = sum(reach(c, node(goal)) for c in cur.children)
    merges = sum(len(n_.parents) >= 2 for n_ in nodes.values())
    return dict(lookahead=min(look, 9), children=min(len(cur.children), 5), useful=useful, vertices=len(nodes) // 10 * 10,
                step=min(len(path), 6), merges=min(merges, 6), edges=len(edges) // 10 * 10)


def main(params, train_py, V, per):
    n, pad, vocab = 3 * V + 5, V + 3, V + 5
    ns = l2s_eval.load(train_py)
    sets = l2s_eval.test_sets(ns, n, per, rng_seed=12345)
    p = load_params(params, CFG, vocab, n)
    said_fn = jax.jit(lambda p, x: st.answer_logits(p, CFG, x, pad, vocab).argmax(-1))
    out = {}
    for name, keys in (('random', [None]), ('balanced', [k for k in sets if k is not None])):
        xs = np.concatenate([sets[k][0] for k in keys]); ys = np.concatenate([sets[k][1] for k in keys])
        if name == 'balanced':
            pick = np.random.default_rng(0).permutation(len(xs))[:per]
            xs, ys = xs[pick], ys[pick]
        said = np.concatenate([np.asarray(said_fn(p, j.array(xs[i:i + 64]))) for i in range(0, len(xs), 64)])
        ok = said == ys
        groups = defaultdict(lambda: defaultdict(list))
        for row, y, good in zip(xs, ys, ok):
            for f, v in features(ns, row, V, int(y)).items():
                groups[f][v].append(good)
        out[name] = dict(accuracy=float(ok.mean()), n=int(len(ok)),
                         by={f: {str(v): [round(float(np.mean(g)), 3), len(g)] for v, g in sorted(d.items())} for f, d in groups.items()})
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
