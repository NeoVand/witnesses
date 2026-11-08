"""What one looped triangle-attention layer computes (S9 weights, 49 vertices; CPU).

For test questions, run the layer K = 1..8 times and record, after each run:
- probe: ridge regression on the 16 normalised channels of z_uv, all ordered pairs of present names, label
  "v is reachable from u within h steps" for h = 1, 2, 4, ..., 32; AUC on held-out questions (even/odd split);
- witness: for pairs with dist(u, v) <= 2^K, the attention mass, summed over heads / 4, on middle vertices k with
  dist(u, k) + dist(k, v) = dist(u, v) (k on a shortest path), against the share of such k among present names.
python mech_loop.py <params.npz> <their train.py> <out.json> [questions per lookahead]
"""
import json, sys, random
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st, l2s_eval

V, P = 49, 128
n, pad = 3 * V + 5, V + 3
STACK = len(sys.argv) > 5 and sys.argv[5] == 'stack'   # the eight-layer S2 model (sum over k, positions, own names)
CFG = (dict(kind='pairn', emb=32, D=16, layers=8, r=8, bf16=False, remat=False) if STACK else
       dict(kind='pairn', emb=32, D=16, layers=1, r=8, loop=True, nopos=True, tri='attn', bf16=False, remat=False))
HS = (1, 2, 4, 8, 16, 32)
KMAX = 8


def restore(path):
    p = st.init(jax.random.PRNGKey(0), CFG, V + 5 if STACK else P + 5, n)
    W = dict(np.load(path))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    return jax.tree_util.tree_unflatten(tree, [j.array(W[jax.tree_util.keystr(k)]) for k, _ in flat])


def run(p, x, e):
    """States after each run and attention over k: lists of [D, N, N] and [H, k, u, v]."""
    h0 = p['tok'][x[None]] + p['pos'][None, :n] if STACK else p['tok'][e[None]]
    z, present, nv, Nn = st.pairn_start(p, x[None], pad, h0)
    lin = lambda t, w: j.einsum('bcij,cd->bdij', t, w)
    cnorm = lambda t: (t - t.mean(1, keepdims=True)) * jax.lax.rsqrt(t.var(1, keepdims=True) + 1e-6)
    m = present
    H, D = 4, z.shape[1]
    dh = D // H
    zs, atts = [], []
    for it in range(KMAX):
        b = p['blocks'][it] if STACK else p['blocks'][0]
        t = cnorm(z)
        Bb, _, N, _ = t.shape
        if STACK:
            a_, c_ = lin(t, b['a']) * m[:, None, None, :], lin(t, b['b']) * m[:, None, :, None]
            z = z + jax.nn.sigmoid(lin(t, b['g'])) * j.matmul(a_, c_) / j.sqrt(nv)
            z = z + lin(jax.nn.gelu(lin(cnorm(z), b['f1'])), b['f2'])
            zs.append(np.asarray(cnorm(z)[0]))
            atts.append(np.zeros((H, N, N, N), np.float32))
            continue
        q = lin(t, b['a']).reshape(Bb, H, dh, N, N)
        k_ = lin(t, b['b']).reshape(Bb, H, dh, N, N)
        sc = j.einsum('bhcuk,bhckv->bhkuv', q, k_) / dh ** .5
        sc = j.where(m[:, None, :, None, None] > 0, sc, -1e9)
        att = jax.nn.softmax(sc, axis=2)
        v1 = lin(t, b['v1']).reshape(Bb, H, dh, N, N)
        v2 = lin(t, b['v2']).reshape(Bb, H, dh, N, N)
        tri = j.einsum('bhkuv,bhcuk,bhckv->bhcuv', att, v1, v2).reshape(Bb, D, N, N)
        z = z + jax.nn.sigmoid(lin(t, b['g'])) * tri
        z = z + lin(jax.nn.gelu(lin(cnorm(z), b['f1'])), b['f2'])
        zs.append(np.asarray(cnorm(z)[0]))
        atts.append(np.asarray(att[0]))
    return zs, atts, np.asarray(present[0])


def distances(row):
    r = [int(t) for t in row if t != pad]
    adj = {}
    for i in range(len(r)):
        if r[i] == V + 2:
            adj.setdefault(r[i + 1], []).append(r[i + 2])
    names = sorted({t for t in r if 1 <= t <= V})
    INF = 10 ** 6
    d = np.full((V + 1, V + 1), INF)
    for s in names:
        d[s, s] = 0
        frontier, seen = [s], {s}
        step = 0
        while frontier:
            step += 1
            nxt = []
            for u in frontier:
                for w in adj.get(u, []):
                    if w not in seen:
                        seen.add(w)
                        d[s, w] = step
                        nxt.append(w)
            frontier = nxt
    return d, names


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float('nan')
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main(params, train_py, out, per=20):
    p = restore(params)
    ns = l2s_eval.load(train_py)
    random.seed(99); np.random.seed(99)
    tests = l2s_eval.test_sets(ns, n, per, rng_seed=777)
    feats = {K: [] for K in range(1, KMAX + 1)}
    labels = {K: {h: [] for h in HS} for K in range(1, KMAX + 1)}
    glabels = {K: {h: [] for h in HS} for K in range(1, KMAX + 1)}  # v reaches the goal within h (any row u)
    qid = {K: [] for K in range(1, KMAX + 1)}
    wit = {K: [[], []] for K in range(1, KMAX + 1)}
    qi = 0
    for L, (xs, ys) in tests.items():
        if L is None:
            continue
        for row in xs:
            t = np.arange(V + 5); t[V + 1:] = P + np.arange(1, 5)
            zs, atts, present = run(p, j.array(row), j.array(t[row]))
            d, names = distances(row)
            rr = [int(tk) for tk in row if tk != pad]
            g = rr[rr.index(V + 4) + 2]
            idx = np.array(names)
            uu, vv = np.meshgrid(idx, idx, indexing='ij')
            off = uu != vv
            for K in range(1, KMAX + 1):
                f = zs[K - 1][:, uu, vv].transpose(1, 2, 0)[off]
                feats[K].append(f)
                qid[K].append(np.full(len(f), qi))
                for h in HS:
                    labels[K][h].append((d[uu, vv] <= h)[off])
                    glabels[K][h].append((d[vv, g] <= h)[off])
                # witness mass: pairs reachable within 2^K, middle vertices on a shortest path
                A = atts[K - 1].mean(0)[np.ix_(idx, idx, idx)]  # [k, u, v]
                Dm = d[np.ix_(idx, idx)]
                onpath = (Dm.T[:, :, None] + Dm[:, None, :]) == Dm[None, :, :]
                sel = (Dm <= 2 ** K) & ~np.eye(len(idx), dtype=bool)
                wit[K][0].extend((A * onpath).sum(0)[sel].tolist())
                wit[K][1].extend(onpath.mean(0)[sel].tolist())
            qi += 1
    res = dict(questions=qi, probe={}, witness={})
    for K in range(1, KMAX + 1):
        F = np.concatenate(feats[K]); Q = np.concatenate(qid[K])
        tr, te = Q % 2 == 0, Q % 2 == 1
        Xtr = np.c_[F[tr], np.ones(tr.sum())]
        res['probe'][K] = {}
        for h in HS:
            y = np.concatenate(labels[K][h]).astype(float)
            w = np.linalg.solve(Xtr.T @ Xtr + 0.01 * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
            res['probe'][K][h] = auc(np.c_[F[te], np.ones(te.sum())] @ w, y[te] > 0.5)
        res.setdefault('goal_probe', {})[K] = {}
        for h in HS:
            y = np.concatenate(glabels[K][h]).astype(float)
            w = np.linalg.solve(Xtr.T @ Xtr + 0.01 * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
            res['goal_probe'][K][h] = auc(np.c_[F[te], np.ones(te.sum())] @ w, y[te] > 0.5)
        res['witness'][K] = dict(mass=float(np.mean(wit[K][0])), share=float(np.mean(wit[K][1])), pairs=len(wit[K][0]))
        print(K, {h: round(v, 3) for h, v in res['probe'][K].items()}, 'witness', round(res['witness'][K]['mass'], 3),
              'vs share', round(res['witness'][K]['share'], 3), 'goal', {h: round(v, 3) for h, v in res['goal_probe'][K].items()}, flush=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20)
