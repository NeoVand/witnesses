"""Export a trained name-level pair state for the browser demo: weights, benchmark questions, reference outputs.

Writes export/pairn_v{V}.json (every parameter: name, shape, flat values), export/graphs_v{V}.json (tokenised test
questions by lookahead with answers and parsed edges), export/reference_v{V}.json (for the first 12 questions: final
logits over names, the answer read from every layer, and every layer's map m[u][v] = w . norm(z_uv)).
python export_demo.py <params.npz> <their train.py> <vertices> <per lookahead>
"""
import json, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st, l2s_eval, probe_search as ps
OUT = Path(__file__).resolve().parent / 'export'


def main(params, train_py, V, per):
    cfg = dict(kind='pairn', emb=32, D=16, layers=8, r=8, bf16=False, remat=False)
    n, pad, vocab = 3 * V + 5, V + 3, V + 5
    p0 = st.init(jax.random.PRNGKey(0), cfg, vocab, n)
    P = dict(np.load(params))
    flat, tree = jax.tree_util.tree_flatten_with_path(p0)
    p = jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])
    weights = [dict(name=jax.tree_util.keystr(k), shape=list(np.asarray(P[jax.tree_util.keystr(k)]).shape),
                    data=[round(float(v), 7) for v in np.asarray(P[jax.tree_util.keystr(k)]).ravel()]) for k, _ in flat]
    meta = dict(vertices=V, tokens=n, pad=pad, vocab=vocab, names=pad - 2, markers=dict(PATH=V + 1, EDGE=V + 2, PAD=V + 3, QUERY=V + 4),
                cfg=cfg, near_offsets=list(st.NEAR), note='forward pass: search_train.forward (kind pairn) and pair_stack; layer norm eps 1e-6')
    (OUT / f'pairn_v{V}.json').write_text(json.dumps(dict(meta=meta, weights=weights)))
    ns = l2s_eval.load(train_py)
    sets = l2s_eval.test_sets(ns, n, per, rng_seed=777)
    qs = []
    for L, (xs, ys) in sets.items():
        for x, y in zip(xs, ys):
            toks = [int(t) for t in x if t != pad]
            q = toks.index(V + 4)
            edges = [[toks[i + 1], toks[i + 2]] for i in range(0, q, 3) if toks[i] == V + 2]
            qs.append(dict(lookahead=L, tokens=[int(t) for t in x], answer=int(y), edges=edges, start=toks[q + 1], goal=toks[q + 2], path=toks[q + 4:]))
    (OUT / f'graphs_v{V}.json').write_text(json.dumps(dict(meta=meta, questions=qs)))
    x = j.array(np.array([q['tokens'] for q in qs[:12]]))
    zs, reads, present = ps.states(p, cfg, x, pad)
    logits = np.asarray(st.forward(p, cfg, x, pad)[1])[:, :pad - 2]
    maps = []
    for zl in zs:
        zt = jnp_t = np.asarray(zl).transpose(0, 2, 3, 1)  # [B, u, v, D]
        nz = (zt - zt.mean(-1, keepdims=True)) / np.sqrt(zt.var(-1, keepdims=True) + 1e-6)
        maps.append(np.einsum('buvd,d->buv', nz, np.asarray(p['w'])))
    ref = dict(meta=meta, questions=list(range(12)), final_logits=np.round(logits, 5).tolist(),
               layer_readouts=[np.round(np.asarray(r), 5).tolist() for r in reads],
               layer_maps=[np.round(m, 4).tolist() for m in maps], present=np.asarray(present).tolist(),
               accuracy=float(np.mean(np.asarray(st.forward(p, cfg, j.array(np.array([q['tokens'] for q in qs])), pad)[1]).argmax(-1) == np.array([q['answer'] for q in qs]))))
    (OUT / f'reference_v{V}.json').write_text(json.dumps(ref))
    print('exported', len(weights), 'arrays,', len(qs), 'questions; accuracy on them', ref['accuracy'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
