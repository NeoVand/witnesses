"""Export the looped OR join (run s21-looprate-v49-s1, the rate form J = 1) for the browser demo: weights, questions
at 125 vertices, and the parity reference the browser's forward pass is tested against.

Checkpoint: results/s21-looprate-v49-s1/params_s1.npz. Chosen
on the reported metrics alone (results/s21_summary.json), before any demo question was run: all three seeds score
1.0000 at 125 vertices, K = 9; seed 1 is the only one at 1.000 on every lookahead and on random graphs at every test
size (seed 0: random graphs 0.994 at 125; seed 2: worst lookahead 0.998 at 100).

    nice -n 19 python -I export_orjoin.py weights               static/model/orjoin_v49.json
    nice -n 19 python -I export_orjoin.py standin               static/model/graphs_v125.json (stand-ins, below)
    nice -n 19 python -I export_orjoin.py questions <train.py>  the same file from the benchmark's own test sets
    nice -n 19 python -I export_orjoin.py questions-from results/export125/graphs_v125_raw.json   (modal_export125.py)
    nice -n 19 python -I export_orjoin.py reference             test-data/reference_orjoin.json

Paths are relative to the repository root. Questions at 125 vertices come from Saparov et al.'s train.py
(generate_eval_data, seeded 777 before each lookahead as l2s_eval.test_sets does, the seed export_demo.py used at 49
vertices): the first 4 of each lookahead in LOOKS_125 and of the random graphs. modal_export125.py runs their code in
a Modal image, `questions-from` writes the demo's file, maps_v125.py lays out the maps and `drop-unmapped` leaves out
the 3 of 40 the map engine could not draw (lookahead 1 and 2, a start with 59 to 119 roads). `standin` (graphs of the
benchmark's lookahead shape, marked meta.kind = 'standin') is a placeholder the site does not use.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TT = HERE.parents[1]                                  # repository root
CKPT = HERE / 'results' / 's21-looprate-v49-s1' / 'params_s1.npz'
JOB = HERE / 'results' / 's21-looprate-v49-s1' / 'started.json'
P = 128  # the name pool
LOOKS_125 = [1, 2, 4, 8, 16, 24, 32, 48, 62]  # the page's lookahead choices at 125 vertices
PER_125 = 4
NEAR = (-4, -3, -2, -1, 1, 2, 3, 4)


def meta_for(V, cfg):
    return dict(vertices=V, tokens=3 * V + 5, pad=V + 3, vocab=V + 5, names=V + 1,
                markers=dict(PATH=V + 1, EDGE=V + 2, PAD=V + 3, QUERY=V + 4))


# ------------------------------------------------------------------------------------------------ weights
def weights():
    job = json.loads(JOB.read_text())
    cfg = job['model']
    W = np.load(CKPT)
    arrays = [dict(name=k, shape=list(W[k].shape), data=[round(float(v), 7) for v in W[k].ravel()]) for k in sorted(W.files)]
    meta = dict(meta_for(job['vertices'], cfg), name_pool=P, embed_rows=P + 5, cfg=dict(cfg, iters_trained=job['train_iters']),
                near_offsets=list(NEAR), seed=job['seeds'][0],
                note='forward pass: loop_train.logits (kind pairn, tri newton with J = 1: the rate form of the OR, one block '
                     'run K times, no positions); names keep their ids, the markers P, E, padding and Q are embedded at rows '
                     'P+1..P+4 of the 133-row table; layer norm eps 1e-6')
    out = TT / 'static' / 'model' / 'orjoin_v49.json'
    out.write_text(json.dumps(dict(meta=meta, weights=arrays), separators=(',', ':')))
    print('wrote', out, out.stat().st_size, 'bytes,', sum(int(np.prod(a['shape'])) for a in arrays), 'parameters')


# ------------------------------------------------------------------------------------------------ questions
def lookahead(ch, vertex, nxt, goal):
    """The benchmark's lookahead of stepping from vertex to nxt toward goal (the demo's graph.ts, a port of theirs)."""
    frontier = [(v, v) for v in ch.get(vertex, [])]
    visited = set(ch.get(vertex, []))
    depth = 0
    while frontier:
        if all(b == nxt for _, b in frontier):
            return depth
        depth += 1
        nf = []
        for v, branch in frontier:
            if v == goal:
                return depth
            for c in ch.get(v, []):
                if c not in visited:
                    nf.append([c, branch])
                    visited.add(c)
                elif branch == nxt:
                    for f in nf:
                        if f[0] == c:
                            f[1] = branch
        frontier = [tuple(f) for f in nf]
    return depth


def encode(edges, start, goal, path, V):
    n = 3 * V + 5
    body = []
    for u, v in edges:
        body += [V + 2, u, v]
    body += [V + 4, start, goal, V + 1, *path]
    assert len(body) <= n
    return [V + 3] * (n - len(body)) + body


def standin_one(L, V, rng):
    """A start, b branches (the goal's of L roads, the others of L or L + 1), names from 1..V, edges shuffled."""
    budget = V - 1
    for _ in range(1000):
        total = int(rng.integers(V // 2, V + 1))  # the benchmark's graphs at 125 vertices hold 67 to 125 on average
        b = int(np.clip(round((total - 1) / (L + 0.5)), 2, 40))
        lens = [L] + [L + int(rng.integers(0, 2)) for _ in range(b - 1)]
        if sum(lens) <= budget:
            break
    names = [int(x) for x in rng.permutation(np.arange(1, V + 1))[: 1 + sum(lens)]]
    start, it = names[0], iter(names[1:])
    edges, firsts, goal = [], [], None
    for i, ln in enumerate(lens):
        prev = start
        for s in range(ln):
            v = next(it)
            if s == 0:
                firsts.append(v)
            edges.append((prev, v))
            prev = v
        if i == 0:
            goal = prev
    order = rng.permutation(len(edges))
    edges = [edges[i] for i in order]
    ch = {}
    for u, v in edges:
        ch.setdefault(u, []).append(v)
    assert lookahead(ch, start, firsts[0], goal) == L, (L, lens)
    return dict(lookahead=L, tokens=encode(edges, start, goal, [start], V), answer=firsts[0],
                edges=[list(e) for e in edges], start=start, goal=goal, path=[start])


def standin(V=125):
    rng = np.random.default_rng(20261107)
    qs = [standin_one(L, V, rng) for L in LOOKS_125 for _ in range(PER_125)]
    meta = dict(meta_for(V, None), kind='standin',
                note='Stand-ins: graphs of the benchmark\'s lookahead shape in its input format, made by export_orjoin.py '
                     '(a start with branches of L or L + 1 roads, the goal at the end of one at L roads). Not the '
                     'benchmark\'s test questions; replace them with `export_orjoin.py questions <train.py>`.')
    write_questions(V, meta, qs)


def questions(train_py, V=125):
    """The benchmark's own test questions, from their train.py on this machine (l2s_eval.test_sets, rng_seed 777)."""
    import l2s_eval
    ns = l2s_eval.load(train_py)
    sets = l2s_eval.test_sets(ns, 3 * V + 5, PER_125, rng_seed=777)
    write_test(V, [(L, xs, ys) for L, (xs, ys) in sets.items()])


def questions_from(raw):
    """The same questions from modal_export125.py's output (their train.py run in the Modal image)."""
    d = json.loads(Path(raw).read_text())
    assert d['per'] == PER_125 and d['seed'] == 777
    write_test(d['vertices'], [(s['lookahead'], s['inputs'], s['outputs']) for s in d['sets']])


def write_test(V, sets):
    """Rule: the first PER_125 questions of each lookahead in LOOKS_125 and of the random graphs, in the test set's order."""
    pad = V + 3
    qs = []
    for L, xs, ys in sets:
        if L is not None and L not in LOOKS_125:
            continue
        for x, y in list(zip(xs, ys))[:PER_125]:
            toks = [int(t) for t in x if t != pad]
            q = toks.index(V + 4)
            edges = [[toks[i + 1], toks[i + 2]] for i in range(0, q, 3) if toks[i] == V + 2]
            qs.append(dict(lookahead=L, tokens=[int(t) for t in x], answer=int(y), edges=edges, start=toks[q + 1],
                           goal=toks[q + 2], path=toks[q + 4:]))
    meta = dict(meta_for(V, None), kind='test', note=f'Saparov et al.\'s test sets (generate_eval_data at {3 * V + 5} tokens, '
                f'random and numpy seeded 777 before each lookahead, as for the 49-vertex set): the first {PER_125} questions '
                f'of each lookahead in {LOOKS_125} and of the random graphs, in the test set\'s order.')
    write_questions(V, meta, qs)


def drop_unmapped(V=125):
    """Rule, after maps_v125.py: a question the map engine cannot lay out (its guarantees fail on every attempt) is left
    out of the page's set and listed in meta.dropped; the remaining maps are renumbered with their questions. The maps
    depend only on each question's graph, so renumbering changes no layout. Nothing here looks at the model."""
    f = TT / 'static' / 'model' / f'graphs_v{V}.json'
    d = json.loads(f.read_text())
    maps = TT / 'static' / f'maps{V}'
    keep, dropped = [], []
    for i, q in enumerate(d['questions']):
        if (maps / f'{i}.json').exists():
            keep.append((i, q))
        else:
            hub = sum(1 for u, _ in q['edges'] if u == q['start'])
            dropped.append(dict(question=i, lookahead=q['lookahead'], vertices=len({x for e in q['edges'] for x in e}),
                                start_roads=hub, reason='the map engine found no layout'))
    tmp = [(maps / f'{i}.json', maps / f'_{j}.json') for j, (i, _) in enumerate(keep)]
    for a, b in tmp:
        a.rename(b)
    for j in range(len(keep)):
        (maps / f'_{j}.json').rename(maps / f'{j}.json')
    d['questions'] = [q for _, q in keep]
    d['meta']['dropped'] = dropped
    d['meta']['note'] += f' Questions the map engine cannot lay out are left out (meta.dropped): {len(dropped)}.'
    f.write_text(json.dumps(d, separators=(',', ':')))
    print('kept', len(keep), 'dropped', dropped)


def write_questions(V, meta, qs):
    out = TT / 'static' / 'model' / f'graphs_v{V}.json'
    out.write_text(json.dumps(dict(meta=meta, questions=qs), separators=(',', ':')))
    print('wrote', out, len(qs), 'questions,', out.stat().st_size, 'bytes')


# ------------------------------------------------------------------------------------------------ the reference
def reference(K=10):
    """The training code's own logits after every iteration K = 1..10 for fixed questions at 49 and 125 vertices, in
    float32 (as the model was trained and scored) and float64 (to show the float32 noise floor), plus every pair's
    read-out (the maps the demo draws) for three of them."""
    import jax
    jax.config.update('jax_platforms', 'cpu')
    import jax.numpy as j
    sys.path.insert(0, str(HERE))
    import search_train as st, loop_train as lt
    job = json.loads(JOB.read_text())
    cfg = dict(job['model'], bf16=False, remat=False, iters=1)
    W = np.load(CKPT)

    def run(xs, Vt, dtype):
        tree = st.init(jax.random.PRNGKey(0), dict(job['model']), P + 5, 3 * job['vertices'] + 5)
        flat, td = jax.tree_util.tree_flatten_with_path(tree)
        p = jax.tree_util.tree_unflatten(td, [j.asarray(W[jax.tree_util.keystr(k)], dtype) for k, _ in flat])
        x = j.asarray(np.array(xs))
        e = j.asarray(lt.marker_ids(np.array(xs), Vt, P))
        pad_t = Vt + 3
        z, present, nv, Nn = st.pairn_start(p, x, pad_t, p['tok'][e], False)
        logits, maps = [], []
        w = p['w']
        for _ in range(K):
            z = st.pair_stack(p, cfg, z, present, nv)
            logits.append(np.asarray(st.pairn_read(p, z, x, present, Nn, pad_t))[:, :Nn])
            zt = np.asarray(z).transpose(0, 2, 3, 1)
            nz = (zt - zt.mean(-1, keepdims=True)) / np.sqrt(zt.var(-1, keepdims=True) + 1e-6)
            maps.append(np.einsum('buvd,d->buv', nz, np.asarray(w)))
        return np.stack(logits, 1), np.stack(maps, 1), np.asarray(present)

    g49 = json.loads((TT / 'static' / 'model' / 'graphs_v49.json').read_text())['questions']
    g125 = json.loads((TT / 'static' / 'model' / 'graphs_v125.json').read_text())
    picks49 = [0, 13, 40, 62, 100, 150, 200, 250, 287, 291, 294, 298]  # lookaheads 1 to 24 and three random graphs
    q125 = g125['questions']
    picks125 = list(range(0, len(q125), max(1, len(q125) // 12)))[:12]
    cases = []
    for V, qs, picks in ((49, g49, picks49), (125, q125, picks125)):
        xs = [qs[i]['tokens'] for i in picks]
        l32, m32, pr = run(xs, V, j.float32)
        jax.config.update('jax_enable_x64', True)
        l64, _, _ = run(xs, V, j.float64)
        jax.config.update('jax_enable_x64', False)
        for b, i in enumerate(picks):
            names = [int(v) for v in np.flatnonzero(pr[b] > 0)]
            c = dict(V=V, source=f'graphs_v{V}.json question {i}' + (' (stand-in)' if V == 125 and g125['meta'].get('kind') == 'standin' else ''),
                     lookahead=qs[i]['lookahead'], tokens=xs[b], names=names, answer=qs[i]['answer'],
                     logits=[[round(float(l32[b, k, v]), 6) for v in names] for k in range(K)],
                     float64_gap=float(np.abs(l32[b][:, names] - l64[b][:, names]).max()))
            if b < (2 if V == 49 else 1):
                c['maps'] = {str(k + 1): [[round(float(m32[b, k, u, v]), 5) for v in names] for u in names] for k in (0, 4, K - 1)}
            cases.append(c)
            print(V, i, 'lookahead', qs[i]['lookahead'], 'names', len(names), 'answers by K',
                  [names[int(np.argmax(l32[b, k, names]))] for k in range(K)], 'expected', qs[i]['answer'],
                  'float32 vs float64', f"{c['float64_gap']:.2e}")
    out = TT / 'test-data' / 'reference_orjoin.json'
    out.write_text(json.dumps(dict(
        note='The training code\'s own logits (loop_train / search_train, float32 on CPU) for the looped OR join of '
             'static/model/orjoin_v49.json: after each iteration K = 1..10, over the names present (ascending); maps: '
             'the read-out w . norm(z_uv) of every pair after K = 1, 5 and 10. float64_gap: the largest difference '
             'between these float32 logits and the same pass in float64.',
        iterations=K, cases=cases), separators=(',', ':')))
    print('wrote', out, out.stat().st_size, 'bytes,', len(cases), 'cases')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'weights':
        weights()
    elif cmd == 'standin':
        standin()
    elif cmd == 'questions':
        questions(sys.argv[2])
    elif cmd == 'questions-from':
        questions_from(sys.argv[2])
    elif cmd == 'drop-unmapped':
        drop_unmapped()
    elif cmd == 'reference':
        reference()
