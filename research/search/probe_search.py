"""What does each layer of a trained name-level pair state know? (paper Fig. 4)

For a saved `pairn` model: run the benchmark's test questions, keep the pair state after every layer, and
(i) fit a linear probe (ridge regression on the 16 normalised channels) for "v is reachable from u within k steps",
    k = 1, 2, 4, ..., trained on half the questions and scored by AUC on the other half;
(ii) apply the final read-out to every layer ("what it would answer if it stopped here"), accuracy by lookahead.
python probe_search.py <params.npz> <their train.py> <vertices> <per lookahead> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st, l2s_eval


def states(p, cfg, x, pad):
    """The pairn forward pass of search_train.py, keeping the state after the start and after every layer."""
    B, n = x.shape
    h0 = p['tok'][x] + p['pos'][None, :n]
    Nn = pad - 2
    hn = st.norm(h0)
    sh = lambda t, o: j.roll(t, -o, axis=1) * ((j.arange(n) + o >= 0) & (j.arange(n) + o < n))[None, :, None]
    ctx = st.norm(hn + sh(hn, -1) @ p['C'][0] + sh(hn, 1) @ p['C'][1])
    OH = jax.nn.one_hot(x, Nn, dtype=hn.dtype)
    cnt = OH.sum(1)
    g = j.einsum('bin,bie->bne', OH, ctx) * jax.lax.rsqrt(j.maximum(cnt, 1.))[..., None]
    r = p['U'].shape[-1]
    u, w = j.einsum('bne,edr->bdnr', g, p['U']), j.einsum('bne,edr->bdnr', g, p['V'])
    z = j.einsum('bdur,bdvr->bduv', u, w) / r ** .5
    for oi, o in enumerate(st.NEAR):
        M = (ctx @ p['A'][oi]) * (sh(ctx, o) @ p['Bm'][oi]) + p['c'][oi]
        z = z + j.einsum('biu,bid,biv->bduv', OH, M, sh(OH, o))
    present = (cnt > 0).astype(z.dtype)
    nv = present.sum(1)[:, None, None, None]
    out = [z]
    for b in p['blocks']:
        z = st.pair_stack(dict(blocks=[b]), dict(cfg, remat=False), z, present, nv)
        out.append(z)
    cur = jax.nn.one_hot(x[:, -1], Nn, dtype=z.dtype)
    reads = []
    for zl in out:
        row = j.einsum('bduv,bu->bvd', zl, cur)
        reads.append(j.where(present > 0, st.norm(row) @ p['w'], -1e9))
    return out, reads, present


def graph_of(row, V):
    E, Q = V + 2, V + 4
    toks = [int(t) for t in row if t != V + 3]
    q = toks.index(Q)
    A = np.zeros((V + 1, V + 1), bool)
    for i in range(0, q, 3):
        if toks[i] == E:
            A[toks[i + 1], toks[i + 2]] = True
    return A


def reach_within(A, k):
    R = A | np.eye(len(A), dtype=bool)
    out, P = R.copy(), R.copy()
    for _ in range(k - 1):
        P = (P.astype(np.int32) @ R.astype(np.int32)) > 0
    return P


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = y.sum(); neg = len(y) - pos
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))


def main(params, train_py, V, per, out_path):
    cfg = dict(kind='pairn', emb=32, D=16, layers=8, r=8, bf16=False, remat=False)
    n, pad, vocab = 3 * V + 5, V + 3, V + 5
    p = st.init(jax.random.PRNGKey(0), cfg, vocab, n)
    P = dict(np.load(params))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    p = jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])
    ns = l2s_eval.load(train_py)
    sets = l2s_eval.test_sets(ns, n, per, rng_seed=12345)
    looks = [L for L in sets if L is not None]
    xs = np.concatenate([sets[L][0] for L in looks]); ys = np.concatenate([sets[L][1] for L in looks])
    lk = np.concatenate([[L] * len(sets[L][0]) for L in looks])
    fwd = jax.jit(lambda p, x: states(p, cfg, x, pad))
    check = jax.jit(lambda p, x: st.forward(p, cfg, x, pad)[1])
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= V]
    rng = np.random.default_rng(0)
    feats = {l: [] for l in range(9)}
    labels = {k: [] for k in ks}
    gfeats = {l: [] for l in range(9)}
    glabels = {k: [] for k in ks}
    gsplit = []
    split = []
    lens_ok = {l: [] for l in range(9)}
    maxdiff = 0.
    for i in range(0, len(xs), 32):
        xb = j.array(xs[i:i + 32])
        zs, reads, present = fwd(p, xb)
        ref = np.asarray(check(p, xb))[:, :pad - 2]
        maxdiff = max(maxdiff, float(np.abs(np.where(ref > -1e8, ref - np.asarray(reads[-1]), 0)).max()))
        for l, rd in enumerate(reads):
            lens_ok[l].extend((np.asarray(rd).argmax(-1) == ys[i:i + 32]).tolist())
        zs = [np.asarray(z) for z in zs]
        pres = np.asarray(present)
        for b in range(xb.shape[0]):
            A = graph_of(xs[i + b], V)
            names = np.flatnonzero(pres[b][:V + 1])
            uu, vv = rng.choice(names, 120), rng.choice(names, 120)
            for l in range(9):
                zb = zs[l][b][:, uu, vv].T
                feats[l].append((zb - zb.mean(1, keepdims=True)) / (zb.std(1, keepdims=True) + 1e-6))
            for k in ks:
                labels[k].append(reach_within(A, k)[uu, vv])
            split.append(np.full(120, (i + b) % 2 == 0))
            toks = [int(t) for t in xs[i + b] if t != pad]
            goal = toks[toks.index(V + 4) + 2]
            for l in range(9):  # goal-directed: the pair (v, goal), for every name v in the graph
                zb = zs[l][b][:, names, goal].T
                gfeats[l].append((zb - zb.mean(1, keepdims=True)) / (zb.std(1, keepdims=True) + 1e-6))
            for k in ks:
                glabels[k].append(reach_within(A, k)[names, goal])
            gsplit.append(np.full(len(names), (i + b) % 2 == 0))
    split = np.concatenate(split)
    res = dict(vertices=V, ks=ks, examples=len(xs), forward_check_maxdiff=maxdiff, probe_auc={}, goal_probe_auc={}, readout_by_lookahead={})
    gsplit = np.concatenate(gsplit)
    for l in range(9):
        X = np.concatenate(feats[l]); X = np.concatenate([X, np.ones((len(X), 1))], 1)
        row = {}
        for k in ks:
            y = np.concatenate(labels[k])
            tr = split
            W = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ (y[tr] * 2. - 1))
            row[str(k)] = auc(X[~tr] @ W, y[~tr])
        res['probe_auc'][str(l)] = row
        X = np.concatenate(gfeats[l]); X = np.concatenate([X, np.ones((len(X), 1))], 1)
        grow = {}
        for k in ks:
            y = np.concatenate(glabels[k]); tr = gsplit
            W = np.linalg.solve(X[tr].T @ X[tr] + 1e-2 * np.eye(X.shape[1]), X[tr].T @ (y[tr] * 2. - 1))
            grow[str(k)] = auc(X[~tr] @ W, y[~tr]) if 0 < y[~tr].mean() < 1 else None
        res['goal_probe_auc'][str(l)] = grow
        ok = np.array(lens_ok[l])
        res['readout_by_lookahead'][str(l)] = {str(L): float(ok[lk == L].mean()) for L in looks}
    Path(out_path).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'readout_by_lookahead'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
