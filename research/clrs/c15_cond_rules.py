"""The hand-built read rules (c15_hand.py) applied to each head's learned conductances on the official file at K = 8:
tree-blind rule, whether the true parent has the lower potential, and how many of the model's errors sit where the best
head misorients. CPU. Needs the runs' params_s<seed>.npz and preds_s<seed>.npz.
Writes results/c15_cond_rules.json (c15) or results/c16_cond_rules.json (c16).
JAX_PLATFORMS=cpu python -I c15_cond_rules.py [c15 | c16]"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax, jax.numpy as j
import clrs_train as ct, c15_cond as cc
H = cc.HERE
d = ct.load(H / 'data_local' / 'mst_prim' / 'test.npz')
G, N = len(d['n']), 64
pi, s = d['pi'].astype(int), d['s'].astype(int)
f = jax.jit(cc.conductances)
out = {}
PRE = sys.argv[1] if len(sys.argv) > 1 else 'c15'
for sd in (0, 1, 2):
    W = dict(np.load(H / 'results' / f'{PRE}-mst_prim-s{sd}' / f'params_s{sd}.npz'))
    p0 = ct.init(jax.random.PRNGKey(sd), dict(cc.CFG, klogn=PRE == 'c16'), ct.OUTS['mst_prim'])
    flat, tree = jax.tree_util.tree_flatten_with_path(p0)
    p = jax.tree_util.tree_unflatten(tree, [j.array(W[jax.tree_util.keystr(k)]) for k, _ in flat])
    pred = np.load(H / 'results' / f'{PRE}-mst_prim-s{sd}' / f'preds_s{sd}.npz')['test/pi'][7].astype(int)
    xs = [ct.to_x(d, np.arange(i, min(i + 8, G))) for i in range(0, G, 8)]
    cs = np.concatenate([np.asarray(f(p, {k: j.array(v) for k, v in x.items()})) for x in xs])
    x = ct.to_x(d, np.arange(G))
    phi, R = ct.kirchhoff(j.array(cs), j.array(x['s1']), j.array(x['m']), 1e-4)
    phi, R = np.asarray(phi, np.float64), np.asarray(R, np.float64)
    res = {}
    for h in range(cs.shape[1]):
        o = np.clip((phi[:, h, :, None] - phi[:, h, None, :]) / (R[:, h] + 1e-6), -1, 1)
        sc = np.where(np.eye(N, dtype=bool)[None], 0., o + 1 / (1 + R[:, h]))
        blind = sc.argmax(-1) == pi
        # tree rule: the true parent has smaller potential than u, and every true child larger
        tr = np.ones((G, N), bool)
        for g in range(G):
            for u in range(N):
                if pi[g, u] != u:
                    tr[g, u] &= phi[g, h, pi[g, u]] < phi[g, h, u]
        res[h] = dict(blind=int(blind.sum()), parent_lower=int(tr.sum()))
    wrong = pred != pi
    best = max(res, key=lambda h: res[h]['parent_lower'])
    o_b = phi[:, best]
    mis = np.array([[pi[g, u] != u and o_b[g, pi[g, u]] >= o_b[g, u] for u in range(N)] for g in range(G)])
    out[sd] = dict(heads=res, model_errors=int(wrong.sum()), best_head=best, errors_where_best_head_misorients=int((wrong & mis).sum()), misoriented_nodes_best_head=int(mis.sum()))
    print(sd, json.dumps(out[sd]), flush=True)
(H / 'results' / ('c15_cond_rules.json' if PRE == 'c15' else f'{PRE}_cond_rules.json')).write_text(json.dumps(out, indent=1))
