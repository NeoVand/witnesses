"""The learned conductances of the Kirchhoff read-out, from the final parameters, at K = 8,
on the official 64-node test file and on val (16 nodes): per head, quantiles of c on true-tree edges and on off-tree input
edges, and the per-node sum of off-tree conductance. CPU. Needs the runs' params_s<seed>.npz (shipped under results/) and data_local/mst_prim/{test,val}.npz (gen_data.py). Writes results/c15_cond_<prefix>.json.
JAX_PLATFORMS=cpu python -I c15_cond.py [run dir template, seeds] (default 'c15-mst_prim-s{sd}' 0,1,2; c16: 'c16-mst_prim-s{sd}')"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax, jax.numpy as j
import clrs_train as ct

CFG = dict(D=128, heads=4, mlp=312, wmlp=32, agg='mix', logn=True, kirch=4, keps=1e-4)


def conductances(p, x, K=8):
    z = ct.embed(p, x)
    for _ in range(K):
        z = ct.agg_block(p['blk'], z, x['m'], 'mix')
    t = ct.cnorm(z)
    A, m = x['A'], x['m']
    N = A.shape[-1]
    off = (1 - j.eye(N)) * m[:, :, None] * m[:, None, :]
    E = (((A != 0) | (A.swapaxes(1, 2) != 0)).astype(A.dtype) * off)[:, None]
    l = j.einsum('bcij,ch->bhij', t, p['kir']['w']) + p['kir']['b'][None, :, None, None]
    if 'kap' in p['kir']:  # klogn
        l = l - p['kir']['kap'][None, :, None, None] * j.log(j.maximum(m.sum(1), 1.))[:, None, None, None]
    return jax.nn.sigmoid((l + l.swapaxes(-1, -2)) / 2) * E


def q(a):
    a = np.asarray(a)
    return {k: float(f'{np.quantile(a, v):.3g}') for k, v in (('p01', .01), ('p10', .1), ('p50', .5), ('p90', .9), ('p99', .99))} if a.size else {}


def main(run='c15-mst_prim-s{sd}', seeds='0,1,2'):
    out = {}
    f = jax.jit(conductances)
    for sd in map(int, seeds.split(',')):
        W = dict(np.load(HERE / 'results' / run.format(sd=sd) / f'params_s{sd}.npz'))
        p0 = ct.init(jax.random.PRNGKey(sd), dict(CFG, klogn=run.startswith('c16')), ct.OUTS['mst_prim'])
        flat, tree = jax.tree_util.tree_flatten_with_path(p0)
        p = jax.tree_util.tree_unflatten(tree, [j.array(W[jax.tree_util.keystr(k)]) for k, _ in flat])
        for split in ('test', 'val'):
            d = ct.load(HERE / 'data_local' / 'mst_prim' / f'{split}.npz')
            G = len(d['n'])
            cs = np.concatenate([np.asarray(f(p, {k: j.array(v) for k, v in ct.to_x(d, np.arange(i, min(i + 8, G))).items()})) for i in range(0, G, 8)])
            A, n, pi = d['A'], d['n'].astype(int), d['pi'].astype(int)
            N = A.shape[-1]
            res = {}
            for h in range(cs.shape[1]):
                tr, ot, leak = [], [], []
                for g in range(G):
                    nn = n[g]
                    T = np.zeros((nn, nn), bool)
                    for v in range(nn):
                        if pi[g, v] != v:
                            T[v, pi[g, v]] = T[pi[g, v], v] = True
                    has = (A[g, :nn, :nn] != 0) & ~np.eye(nn, dtype=bool)
                    c = cs[g, h, :nn, :nn]
                    iu = np.triu_indices(nn, 1)
                    tr += list(c[iu][T[iu]]); ot += list(c[iu][(has & ~T)[iu]])
                    leak += list((c * (has & ~T)).sum(1))
                res[f'head{h}'] = dict(tree=q(tr), offtree=q(ot), offtree_sum_per_node=q(leak),
                                       tree_below_half=round(float(np.mean(np.array(tr) < .5)), 4),
                                       offtree_above_1em4=round(float(np.mean(np.array(ot) > 1e-4)), 4),
                                       offtree_above_1em3=round(float(np.mean(np.array(ot) > 1e-3)), 4))
            out[f's{sd}/{split}'] = res
            print(sd, split, json.dumps(res), flush=True)
    (HERE / 'results' / f"c15_cond_{run.split('-')[0]}.json").write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
