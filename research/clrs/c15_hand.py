"""The Kirchhoff read-out built by hand (CPU). On the official 64-node mst_prim
test file: conductance 1 on the true MST edges, 0 (or `off`) on the other input-graph edges, grounded at s, eps I added,
solved in float32 by clrs_train.kirchhoff (the code the trained runs use). Reads each node's parent two ways:
(a) the tree neighbour with smaller potential; (b) tree-blind, from the read-out's two features alone:
argmax_j o_uj + q_uj with the diagonal at 0 (the root has no positive candidate and keeps itself).
Reports exactness, max |phi - depth|, and the condition number of the grounded matrix (float64, numpy).
Writes results/c15_hand_eps<eps>.json.
JAX_PLATFORMS=cpu python -I c15_hand.py [eps, default 1e-4] [split file ..., default data_local/mst_prim/test.npz]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax, jax.numpy as j
import clrs_train as ct


def run(d, off, eps):
    A, n, s, pi = d['A'], d['n'].astype(int), d['s'].astype(int), d['pi'].astype(int)
    G, N = A.shape[0], A.shape[-1]
    m = (np.arange(N)[None] < n[:, None]).astype(np.float32)
    s1 = np.zeros((G, N), np.float32); s1[np.arange(G), s] = 1
    has = ((A != 0) | (A.swapaxes(1, 2) != 0)) & ~np.eye(N, dtype=bool)[None] & (m[:, :, None] * m[:, None, :] > 0)
    tree = np.zeros((G, N, N), bool)
    for g in range(G):
        for v in range(n[g]):
            if pi[g, v] != v:
                tree[g, v, pi[g, v]] = tree[g, pi[g, v], v] = True
    assert (tree <= has).all()
    c = np.where(tree, 1., np.where(has, off, 0.)).astype(np.float32)[:, None]
    phi, R = ct.kirchhoff(j.array(c), j.array(s1), j.array(m), eps)
    phi, R = np.asarray(phi)[:, 0].astype(np.float64), np.asarray(R)[:, 0].astype(np.float64)
    o = np.clip((phi[:, :, None] - phi[:, None, :]) / (R + 1e-6), -1, 1)
    score = np.where(np.eye(N, dtype=bool)[None], 0., o + 1 / (1 + R))
    score = np.where(m[:, None, :] > 0, score, -np.inf)
    dep = ct.tree_depth(np.where(m > 0, pi, -1))
    ok_a = ok_b = tot = 0
    conds, deperr = [], []
    for g in range(G):
        k = [u for u in range(n[g]) if u != s[g]]
        Lg = np.diag(c[g, 0].sum(1)) - c[g, 0]
        Lg = Lg[np.ix_(k, k)].astype(np.float64) + eps * np.eye(len(k))
        conds.append(np.linalg.cond(Lg))
        for u in range(n[g]):
            tot += 1
            nb = np.nonzero(tree[g, u])[0]
            low = [v for v in nb if phi[g, v] < phi[g, u]]
            pa = u if u == s[g] else (low[0] if len(low) == 1 else -1)
            ok_a += pa == pi[g, u]
            ok_b += int(score[g, u].argmax()) == pi[g, u]
            deperr.append(abs(phi[g, u] - dep[g, u]))
    return dict(nodes=tot, exact_tree_rule=int(ok_a), exact_blind_rule=int(ok_b), max_abs_phi_minus_depth=round(float(max(deperr)), 4),
                max_depth=int(dep.max()), cond_median=float(f'{np.median(conds):.4g}'), cond_max=float(f'{max(conds):.4g}'))


def main(eps='1e-4', *files):
    eps = float(eps)
    files = files or [str(HERE / 'data_local' / 'mst_prim' / 'test.npz')]
    out = {}
    for f in files:
        d = {k: v for k, v in np.load(f).items()}
        for off in (0., 1e-3):
            r = run(d, off, eps)
            out[f'{Path(f).name} off={off} eps={eps}'] = r
            print(Path(f).name, 'off', off, 'eps', eps, json.dumps(r), flush=True)
    (HERE / 'results' / f'c15_hand_eps{eps:g}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
