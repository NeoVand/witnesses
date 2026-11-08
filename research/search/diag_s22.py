"""S22 diagnostic (CPU): what a trained propagator layer learned, on the benchmark's test questions at several sizes.
Per channel h: t0_h; the gates G on reversed edges (k <- v, the orientation that makes X_cv = walks from v to the goal),
forward edges and non-edges; the source S on the goal column and elsewhere; the spectral radius of G (the pole is at 1);
the per-hop weight on edges; and X at the answer against the best other child, by lookahead.
python3 -I diag_s22.py <run id> <their train.py> [sizes ...]; writes results/<run>/diag.json."""
import json, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as j
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search_train as st, l2s_eval


def load(run):
    d = HERE / 'results' / run
    row = json.loads((d / 'rows.json').read_text())[0]
    cfg, P, V = row['cfg'], row['name_pool'], row['vertices']
    tmpl = st.init(jax.random.PRNGKey(0), cfg, P + 5, 3 * V + 5)
    W = dict(np.load(d / f"params_s{row['seed']}.npz"))
    flat, tree = jax.tree_util.tree_flatten_with_path(tmpl)
    return jax.tree_util.tree_unflatten(tree, [j.array(W[jax.tree_util.keystr(k)]) for k, _ in flat]), cfg, P


def main(run, train_py, sizes):
    p, cfg, P = load(run)
    b = p['blocks'][0]
    t0 = np.asarray(jax.nn.sigmoid(b['pt']))
    ns = l2s_eval.load(train_py)
    out = dict(t0=t0.tolist())
    for V in sizes:
        sets = l2s_eval.test_sets(ns, 3 * V + 5, 2 if V > 100 else 4, rng_seed=12345)
        pad = V + 3
        stats = {h: dict(rev=[], fwd=[], non=[], sgoal=[], sother=[]) for h in range(len(t0))}
        rho, marg = [], {}
        for L, (xs, ys) in sets.items():
            for x, y in zip(xs, ys):
                x = np.asarray(x)[None]
                e = np.arange(V + 5)
                e[V + 1:] = P + np.arange(1, 5)
                xx = j.array(x)
                z, present, nv, Nn = st.pairn_start(p, xx, pad, p['tok'][j.array(e[x])])
                t = (z - z.mean(1, keepdims=True)) * jax.lax.rsqrt(z.var(1, keepdims=True) + 1e-6)
                proj = lambda w, c: j.einsum('bcij,ch->bhij', t, w) + c[None, :, None, None]
                mm = np.asarray(present[:, None, :, None] * present[:, None, None, :])[0]
                G = np.asarray(jax.nn.sigmoid(b['pt'])[None, :, None, None] * jax.nn.sigmoid(proj(b['pg'], b['pgb'])))[0] * mm
                S = np.asarray(jax.nn.sigmoid(proj(b['ps'], b['psb'])))[0] * mm
                X = np.asarray(st.propagator(t, b, present, cfg))[0]
                toks = [int(v) for v in x[0] if v != pad]
                q = toks.index(V + 4)
                A = np.zeros((V + 1, V + 1), bool)
                for i in range(0, q, 3):
                    if toks[i] == V + 2:
                        A[toks[i + 1], toks[i + 2]] = True
                goal, cur = toks[q + 2], toks[-1]
                pres = mm[0] > 0
                non = pres & ~A & ~A.T
                gm = np.zeros_like(pres)
                gm[:, goal] = True
                for h in range(len(t0)):
                    stats[h]['rev'] += list(G[h][A.T]); stats[h]['fwd'] += list(G[h][A]); stats[h]['non'] += list(G[h][non][::7])
                    stats[h]['sgoal'] += list(S[h][pres & gm]); stats[h]['sother'] += list(S[h][pres & ~gm][::7])
                if len(rho) < 6:
                    rho.append(np.abs(np.linalg.eigvals(G)).max(-1).tolist())
                ch = [v for v in np.flatnonzero(A[cur]) if v != int(y)]
                if ch:
                    lx = np.log(X[:, cur] + 1e-20)
                    marg.setdefault(str(L), []).append((lx[:, int(y)] - lx[:, ch].max(1)).tolist())
        q = lambda a, s: float(np.quantile(a, s)) if len(a) else None
        out[str(V)] = dict(
            channels={h: dict(t0=float(t0[h]), gate_rev_edge_median=q(s['rev'], .5), gate_rev_edge_min=q(s['rev'], 0.),
                              gate_fwd_edge_median=q(s['fwd'], .5), gate_nonedge_median=q(s['non'], .5), gate_nonedge_p99=q(s['non'], .99),
                              src_goal_median=q(s['sgoal'], .5), src_other_median=q(s['sother'], .5), src_other_p99=q(s['sother'], .99))
                      for h, s in stats.items()},
            spectral_radius=rho,
            logX_answer_minus_best_other_child={L: np.median(np.array(v), 0).round(2).tolist() for L, v in marg.items()})
        print(V, json.dumps(out[str(V)])[:3000], flush=True)
    (HERE / 'results' / run / 'diag.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]] or [49])
