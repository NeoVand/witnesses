"""Export the trained propagator (run s22c-cap99-v49-s0, the row-capped layer) for the browser demo: weights, the
250-vertex question and its map, and the parity reference the browser's forward pass is tested against.

Checkpoint: results/s22c-cap99-v49-s0/params_s0.npz, the run whose gates Fig. 2a-b draws. One layer, four channels,
t0 = 0.99 fixed, rows capped, names from a pool of 256, no positions; 28,472 parameters. CPU only (jax, numpy):

    nice -n 19 python -I export_prop.py weights      static/model/prop_v49.json
    nice -n 19 python -I export_prop.py q250         static/model/graphs_v250.json, static/maps250/0.json
    nice -n 19 python -I export_prop.py reference    test-data/reference_prop.json

Paths are relative to the repository root. The 250-vertex question is paper/figures/data/q250.json (test question 1
at lookahead 31 of the benchmark's own 250-vertex test set, the set the s22c runs were scored on) and its map
paper/figures/data/map250.json (both written by search250.py), copied unchanged.
`reference` runs search_train's own forward pass (pairn_start, pair_stack with tri 'prop', pairn_read; float32 on CPU,
as scored) on fixed questions at 49, 125 and 250 vertices and writes, per question, the logits over the names present,
and per channel the current row of X = S (I - G)^{-1} and of the source S; for the 250-vertex question also the gates
of every road in both directions. It also writes the training code's answer to every exported question (300 at 49, 37
at 125, 1 at 250), so the browser's pass can be checked to answer exactly as the training code does.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TT = HERE.parents[1]                                  # repository root
PAPER = TT / 'paper' / 'figures' / 'data'
RUN = 's22c-cap99-v49-s0'
CKPT = HERE / 'results' / RUN / 'params_s0.npz'
JOB = HERE / 'results' / RUN / 'started.json'
P = 256  # the name pool
NEAR = (-4, -3, -2, -1, 1, 2, 3, 4)


def meta_for(V):
    return dict(vertices=V, tokens=3 * V + 5, pad=V + 3, vocab=V + 5, names=V + 1,
                markers=dict(PATH=V + 1, EDGE=V + 2, PAD=V + 3, QUERY=V + 4))


def weights():
    job = json.loads(JOB.read_text())
    cfg = job['model']
    assert cfg['tri'] == 'prop' and cfg['rowcap'] and not cfg['learn_t0'] and cfg['layers'] == 1
    W = np.load(CKPT)
    arrays = [dict(name=k, shape=list(W[k].shape), data=[round(float(v), 7) for v in W[k].ravel()]) for k in sorted(W.files)]
    params = sum(int(np.prod(a['shape'])) for a in arrays)
    meta = dict(meta_for(job['vertices']), name_pool=P, embed_rows=P + 5, cfg=cfg, near_offsets=list(NEAR),
                seed=job['seeds'][0], params=params,
                note='forward pass: search_train.pairn_start, pair_stack (tri prop: the propagator, one layer, four '
                     'channels; gates G = t0 sigma(pg . t + pgb), rows scaled so that no row of G sums past t0; source '
                     'S = sigma(ps . t + psb); X = S (I - G)^-1 by one linear solve; features log(X + 1e-20) and '
                     '1 - exp(-X) mixed by po and gated by sigma(g . t); then the per-pair MLP) and pairn_read; names keep '
                     'their ids, the markers P, E, padding and Q are embedded at rows 257..260 of the 261-row table; '
                     'layer norm eps 1e-6')
    out = TT / 'static' / 'model' / 'prop_v49.json'
    out.write_text(json.dumps(dict(meta=meta, weights=arrays), separators=(',', ':')))
    print('wrote', out, out.stat().st_size, 'bytes,', params, 'parameters')


def q250():
    q = json.loads((PAPER / 'q250.json').read_text())
    m = json.loads((PAPER / 'map250.json').read_text())
    V = q['vertices']
    assert m['start'] == q['start'] and m['goal'] == q['goal'] and sorted(map(tuple, m['edges'])) == sorted(map(tuple, q['edges']))
    toks = [int(t) for t in q['tokens']]
    meta = dict(meta_for(V), kind='test',
                note=f'Saparov et al.\'s 250-vertex test set ({q["source"]}): test question {q["index"]} at lookahead '
                     f'{q["lookahead"]}, the first at that lookahead that names all 250 vertices, from the set the '
                     'propagator runs were scored on (seed 12345).')
    out = dict(meta=meta, questions=[dict(lookahead=q['lookahead'], tokens=toks, answer=int(q['answer']), edges=q['edges'],
                                          start=q['start'], goal=q['goal'], path=q['path'])])
    f = TT / 'static' / 'model' / 'graphs_v250.json'
    f.write_text(json.dumps(out, separators=(',', ':')))
    d = TT / 'static' / 'maps250'
    d.mkdir(exist_ok=True)
    m.pop('question', None)  # its 'source' is a path on the training machine; the map itself is unchanged
    (d / '0.json').write_text(json.dumps(m, separators=(',', ':')))
    print('wrote', f, 'and', d / '0.json')


def reference():
    import jax
    jax.config.update('jax_platforms', 'cpu')
    import jax.numpy as j
    sys.path.insert(0, str(HERE))
    import search_train as st, loop_train as lt
    job = json.loads(JOB.read_text())
    cfg = dict(job['model'], bf16=False, remat=False)
    W = np.load(CKPT)

    def params(dtype):
        tree = st.init(jax.random.PRNGKey(0), dict(job['model']), P + 5, 3 * job['vertices'] + 5)
        flat, td = jax.tree_util.tree_flatten_with_path(tree)
        return jax.tree_util.tree_unflatten(td, [j.asarray(W[jax.tree_util.keystr(k)], dtype) for k, _ in flat])

    def run(xs, Vt, dtype):
        p = params(dtype)
        x = j.asarray(np.array(xs))
        e = j.asarray(lt.marker_ids(np.array(xs), Vt, P))
        pad_t = Vt + 3
        z, present, nv, Nn = st.pairn_start(p, x, pad_t, p['tok'][e], False)
        b = p['blocks'][0]
        t = (z - z.mean(1, keepdims=True)) * jax.lax.rsqrt(z.var(1, keepdims=True) + 1e-6)
        X = st.propagator(t, b, present, cfg)                     # [B, H, N, N]
        mm = present[:, None, :, None] * present[:, None, None, :]
        proj = lambda w, c: j.einsum('bcij,ch->bhij', t, w) + c[None, :, None, None]
        S = jax.nn.sigmoid(proj(b['ps'], b['psb'])) * mm
        t0 = cfg['t0']
        G = t0 * jax.nn.sigmoid(proj(b['pg'], b['pgb'])) * mm
        G = G * (t0 / j.maximum(G.sum(-1, keepdims=True), t0))
        z = st.pair_stack(p, cfg, z, present, nv)
        logits = st.pairn_read(p, z, x, present, Nn, pad_t)[:, :Nn]
        return (np.asarray(logits, np.float64), np.asarray(X, np.float64), np.asarray(S, np.float64),
                np.asarray(G, np.float64), np.asarray(present))

    sets = {}
    for V in (49, 125, 250):
        sets[V] = json.loads((TT / 'static' / 'model' / f'graphs_v{V}.json').read_text())['questions']
    picks = {49: [0, 13, 40, 62, 100, 150, 200, 250, 287, 291, 294, 298],
             125: list(range(0, len(sets[125]), 3))[:12], 250: [0]}
    cases, answers = [], {}
    for V, qs in sets.items():
        said = []
        for i in range(0, len(qs), 16):
            xs = [q['tokens'] for q in qs[i:i + 16]]
            lg, *_ = run(xs, V, j.float32)
            said += [int(np.argmax(r)) for r in lg]
        right = sum(int(s == q['answer']) for s, q in zip(said, qs))
        answers[str(V)] = said
        print(V, 'questions', len(qs), 'answered as the benchmark', right)
        xs = [qs[i]['tokens'] for i in picks[V]]
        l32, X32, S32, G32, pr = run(xs, V, j.float32)
        jax.config.update('jax_enable_x64', True)
        l64, X64, _, _, _ = run(xs, V, j.float64)
        jax.config.update('jax_enable_x64', False)
        for b, i in enumerate(picks[V]):
            q = qs[i]
            names = [int(v) for v in np.flatnonzero(pr[b] > 0)]
            cur = q['tokens'][-1]
            kids = [v for u, v in q['edges'] if u == cur]
            ans = int(np.argmax(l32[b]))
            margin = []
            for h in range(X32.shape[1]):
                lx = np.log(X32[b, h, cur] + 1e-20)
                other = [lx[k] for k in kids if k != ans]
                margin.append(float(lx[ans] - max(other)) if other else float('-inf'))
            c = dict(V=V, question=i, lookahead=q['lookahead'], tokens=q['tokens'], names=names, current=cur,
                     answer=ans, expected=q['answer'],
                     logits=[round(float(l32[b, v]), 6) for v in names],
                     X=[[float(f'{X32[b, h, cur, v]:.7g}') for v in names] for h in range(X32.shape[1])],
                     S=[[float(f'{S32[b, h, cur, v]:.7g}') for v in names] for h in range(S32.shape[1])],
                     working=int(np.argmax(margin)), margins=[round(x, 4) for x in margin],
                     float64_gap=dict(logits=float(np.abs(l32[b][names] - l64[b][names]).max()),
                                      X=float(np.abs(X32[b][:, cur, names] - X64[b][:, cur, names]).max())))
            if V == 250:
                c['gates'] = [dict(edge=[u, v], back=[float(f'{G32[b, h, v, u]:.6g}') for h in range(4)],
                                   along=[float(f'{G32[b, h, u, v]:.6g}') for h in range(4)]) for u, v in q['edges']]
            cases.append(c)
            print(V, i, 'lookahead', q['lookahead'], 'names', len(names), 'answer', ans, 'expected', q['answer'],
                  'working channel', c['working'], 'margins', c['margins'], 'float64 gap', c['float64_gap'])
    out = TT / 'test-data' / 'reference_prop.json'
    out.write_text(json.dumps(dict(
        note='The training code\'s own forward pass (search_train.pairn_start, pair_stack with the propagator, '
             'pairn_read; float32 on CPU, as the runs were scored) for static/model/prop_v49.json. Per case: the logits '
             'over the names present (ascending), and per channel h the current town\'s row of X = S (I - G)^-1 and of '
             'the source S over the same names; working: the channel with the largest margin of log X at the answer '
             'over the current town\'s other children; float64_gap: the largest difference between these float32 values '
             'and the same pass in float64. answers: the training code\'s answer to every exported question.',
        cases=cases, answers=answers), separators=(',', ':')))
    print('wrote', out, out.stat().st_size, 'bytes,', len(cases), 'cases')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'weights':
        weights()
    elif cmd == 'q250':
        q250()
    elif cmd == 'reference':
        reference()
