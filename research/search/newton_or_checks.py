"""S20 local checks for the Newton-OR and power-mean pair layers.

1. series:   L_J = sum_{j<=J} P_j / j against the exact -sum_k log(1 - a_uk c_kv), error against J, random inputs.
2. boolean:  entries in {0, 0.999}: the soft OR thresholded at 0.5 equals the Boolean matrix product.
3. same:     with the options off, search_train.py gives a reference copy's parameters and outputs (S8 loop,
             S8 fixed8, S9 triangle loop, S16 ssmax + pool_norm, S17 sann loop), bit for bit.
4. smoke:    Newton-OR and power-mean loops (bf16, as training): finite loss and gradients; loss falls on a fixed batch.
5. theory:   a hand-set OR layer (search_train.newton_or itself) decides reach within 2^L steps on the benchmark's test
             graphs (export/graphs_v49.json, made by the benchmark's own test-set code) and on random DAGs, as
             Theorem 1's summed construction does; gain needed against n; J = 1
             against 8; starting states perturbed by noise, OR against sum.
python3 -I newton_or_checks.py <reference search_train.py> [part ...]; writes results/s20_checks.json.
"""
import importlib.util, json, math, sys, time
from collections import deque
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import search_train as st
OUT = HERE / 'results' / 's20_checks.json'


def series(rng):
    """Error of the truncated series L_J against the exact -sum_k log(1 - a c), and of the soft OR 1 - exp(-L_J)
    against the exact 1 - prod_k (1 - a c), per J; the OR is checked with few middle vertices, where it is not near 1."""
    out = {}
    for regime, lo, hi in (('uniform(0,1)', 0., 1.), ('uniform(0.5,0.999)', .5, .999), ('uniform(0,0.5)', 0., .5)):
        for N in (125, 4):
            B, D = 2, 4
            a = rng.uniform(lo, hi, (B, D, N, N)).astype(np.float32)
            c = rng.uniform(lo, hi, (B, D, N, N)).astype(np.float32)
            ac = a.astype(np.float64)[..., :, :, None] * c.astype(np.float64)[..., None, :, :]  # [B, D, u, k, v]
            exact_L = -np.log1p(-ac).sum(-2)
            exact_O = 1 - np.exp(-exact_L)
            rows = {}
            for J in (1, 2, 4, 8, 12, 16, 32):
                L = np.asarray(st.newton_series(j.array(a), j.array(c), j.ones((J, D)))).astype(np.float64)
                O = np.asarray(st.newton_or(j.array(a), j.array(c), j.ones((J, D)))).astype(np.float64)
                rel = np.abs(L - exact_L) / exact_L
                rows[str(J)] = dict(median_rel_err_L=float(np.median(rel)), max_rel_err_L=float(rel.max()),
                                    max_abs_err_O=float(np.max(np.abs(O - exact_O))))
            out[f'{regime} N={N}'] = rows
    return out


def boolean(rng):
    """Soft OR with entries in {0, 0.999}, thresholded at 0.5, against the Boolean product, over many graphs."""
    bad = tot = 0
    lo_true, hi_false = 1., 0.
    for trial in range(60):
        N = int(rng.choice([5, 17, 49, 125, 250]))
        dens = rng.uniform(.005, .6)
        A = rng.random((1, 2, N, N)) < dens
        C = rng.random((1, 2, N, N)) < dens
        O = np.asarray(st.newton_or(j.array(A * .999, j.float32), j.array(C * .999, j.float32), j.ones((8, 2))))
        truth = (A.astype(np.int64) @ C.astype(np.int64)) > 0
        bad += int(((O > .5) != truth).sum())
        tot += truth.size
        if truth.any():
            lo_true = min(lo_true, float(O[truth].min()))
        if (~truth).any():
            hi_false = max(hi_false, float(O[~truth].max()))
    return dict(entries=tot, disagreements=bad, min_O_where_true=lo_true, max_O_where_false=hi_false, J=8)


def load_orig(path):
    spec = importlib.util.spec_from_file_location('search_train_orig', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(orig_path, rng):
    """Options off: identical parameters and identical read-outs, old against new code."""
    old = load_orig(orig_path)
    PAIRN = dict(kind='pairn', emb=32, D=16, layers=8, r=8)
    SANN = dict(kind='sann', emb=32, D=16, r=8, d=64, mlp=128, H=4, layers=8)
    cfgs = {'s8-loop': dict(PAIRN, layers=1, loop=True, nopos=True), 's8-fixed8': dict(PAIRN, nopos=True),
            's9-looptri': dict(PAIRN, layers=1, loop=True, nopos=True, tri='attn'),
            's16-poolss': dict(PAIRN, layers=1, loop=True, nopos=True, tri='attn', pool_norm=True, ssmax=True),
            's17-sannloop': dict(SANN, layers=1, loop=True, nopos=True)}
    V, P = 9, 16
    n, pad = 3 * V + 5, V + 3
    x = rng.integers(0, V + 5, (4, n))
    x[:, -1] = rng.integers(1, V + 1, 4)
    e = np.where(x <= V, x, x - V - 1 + P + 1)
    out = {}
    for name, cfg in cfgs.items():
        res = {}
        for seed in (0, 1):
            p0 = old.init(jax.random.PRNGKey(seed), cfg, P + 5, n)
            p1 = st.init(jax.random.PRNGKey(seed), cfg, P + 5, n)
            l0, t0 = jax.tree_util.tree_flatten_with_path(p0)
            l1, t1 = jax.tree_util.tree_flatten_with_path(p1)
            same_tree = [jax.tree_util.keystr(k) for k, _ in l0] == [jax.tree_util.keystr(k) for k, _ in l1]
            pdiff = max(float(j.max(j.abs(a - b))) for (_, a), (_, b) in zip(l0, l1)) if same_tree else None
            for bf16 in (True, False):
                c = dict(cfg, iters=6, bf16=bf16, remat=bf16)
                outs = []
                for mod in (old, st):
                    xx, ee = j.array(x), j.array(e)
                    if cfg['kind'] == 'sann':
                        h, zn, present, Nn = mod.sann_start(p0, xx, pad, p0['tok'][ee])
                        outs.append(mod.sann_read(p0, mod.sann_stack(p0, c, h, zn, present), xx, present, Nn, pad))
                    else:
                        z, present, nv, Nn = mod.pairn_start(p0, xx, pad, p0['tok'][ee], cfg.get('pool_norm', False))
                        outs.append(mod.pairn_read(p0, mod.pair_stack(p0, c, z, present, nv), xx, present, Nn, pad))
                res[f'seed{seed}_bf16{int(bf16)}'] = dict(same_param_tree=same_tree, max_param_diff=pdiff,
                                                         max_logit_diff=float(j.max(j.abs(outs[0] - outs[1]))))
        out[name] = res
    # stored S8 / S9 weights through the new code: the read-outs equal the old code's
    for name, run in (('s8-loop', 's8-loop-v49-s0'), ('s9-looptri', 's9-looptri-v49-s0')):
        f = HERE / 'results' / run / 'params_s0.npz'
        if not f.exists():
            continue
        cfg = cfgs[name]
        V, P = 49, 128
        n, pad = 3 * V + 5, V + 3
        tmpl = st.init(jax.random.PRNGKey(0), cfg, P + 5, n)
        W = dict(np.load(f))
        flat, tree = jax.tree_util.tree_flatten_with_path(tmpl)
        p = jax.tree_util.tree_unflatten(tree, [j.array(W[jax.tree_util.keystr(k)]) for k, _ in flat])
        qs = json.loads((HERE / 'export' / 'graphs_v49.json').read_text())['questions'][::25]
        xx = j.array(np.array([q['tokens'] for q in qs]))
        t = np.arange(V + 5)
        t[V + 1:] = P + np.arange(1, 5)
        ee = j.array(t[np.asarray(xx)])
        c = dict(cfg, iters=8, bf16=False, remat=False)
        outs, said = [], []
        for mod in (old, st):
            z, present, nv, Nn = mod.pairn_start(p, xx, pad, p['tok'][ee])
            lg = mod.pairn_read(p, mod.pair_stack(p, c, z, present, nv), xx, present, Nn, pad)
            outs.append(lg)
            said.append(np.asarray(lg.argmax(-1)))
        out[name + '_stored_weights'] = dict(questions=len(qs), max_logit_diff=float(j.max(j.abs(outs[0] - outs[1]))),
                                             answers_right=int((said[1] == np.array([q['answer'] for q in qs])).sum()))
    return out


def smoke(rng):
    """bf16 training steps on one fixed random batch of tokens: finite, and the loss falls (memorisation only)."""
    out = {}
    V, P = 9, 16
    n, pad = 3 * V + 5, V + 3
    x = rng.integers(0, V + 5, (32, n))
    x[:, -1] = rng.integers(1, V + 1, 32)
    e = np.where(x <= V, x, x - V - 1 + P + 1)
    lab = np.array([rng.choice(row[(row >= 1) & (row <= V)]) for row in x])  # a name the question writes
    for tri in ('newton', 'pmean'):
        cfg = dict(kind='pairn', emb=32, D=16, layers=1, r=8, loop=True, nopos=True, tri=tri, iters=6)
        p = st.init(jax.random.PRNGKey(0), cfg, P + 5, n)
        tx = optax.adam(3e-3)
        s = tx.init(p)

        @jax.jit
        def step(p, s):
            def loss(p):
                z, present, nv, Nn = st.pairn_start(p, j.array(x), pad, p['tok'][j.array(e)])
                lp = jax.nn.log_softmax(st.pairn_read(p, st.pair_stack(p, cfg, z, present, nv), j.array(x), present, Nn, pad), -1)
                return -j.take_along_axis(lp, j.array(lab)[:, None], 1).mean()
            v, g = jax.value_and_grad(loss)(p)
            u, s = tx.update(g, s, p)
            gn = optax.global_norm(g)
            return optax.apply_updates(p, u), s, v, gn
        curve = []
        for i in range(150):
            p, s, v, gn = step(p, s)
            if i % 25 == 0 or i == 149:
                curve.append([i, round(float(v), 4), round(float(gn), 4)])
        params = int(sum(w.size for w in jax.tree.leaves(p)))
        out[tri] = dict(params_at_V9=params, curve_step_loss_gradnorm=curve,
                        finite=bool(all(np.isfinite(c[1]) and np.isfinite(c[2]) for c in curve)))
    return out


# ---------------------------------------------------------------- theory: hand-set OR layer against Theorem 1's sum
def dists(adj):
    """d[v, g] = length of a shortest path from v to g (inf if none), by BFS from every vertex."""
    n = len(adj)
    d = np.full((n, n), np.inf)
    nbr = [np.flatnonzero(adj[u]) for u in range(n)]
    for s in range(n):
        d[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for w in nbr[u]:
                if d[s, w] == np.inf:
                    d[s, w] = d[s, u] + 1
                    q.append(w)
    return d


def or_layers(A, g, L, alpha=40., beta=40., J=8, noise=0., rng=None, gate=True):
    """L layers of the OR layer, hand-set; channels R, T, E, H, F and a constant channel 1 (the biases).
    Repeated layer: a = sigma(alpha (x_uk - 1/2)), c = sigma(alpha (y_kv - 1/2)) for (x, y) = (R, R), (T, T), (H, T);
    O = newton_or(a, c) (search_train's own function, omega = 1); x += sigma(beta (1/2 - x_uv)) O, a gate that closes
    once x_uv is set; no MLP. Last layer: the same join, then the MLP F += relu(H + 2 E - 2) (one unit).
    Read-out F_cv - 1/2. Returns F."""
    n = len(A)
    sig = lambda x: 1 / (1 + np.exp(-x))
    R = ((np.eye(n) + A) > 0).astype(np.float64)
    Z = [R, R.T.copy(), A.astype(np.float64), np.zeros((n, n)), np.zeros((n, n))]
    Z[3][:, g] = 1
    if noise:
        Z[:4] = [z + rng.uniform(-noise, noise, z.shape) for z in Z[:4]]
    R, T, E, H, F = Z
    for l in range(L):
        a = np.stack([sig(alpha * (R - .5)), sig(alpha * (T - .5)), sig(alpha * (H - .5))])[None]
        c = np.stack([sig(alpha * (R - .5)), sig(alpha * (T - .5)), sig(alpha * (T - .5))])[None]
        O = np.asarray(st.newton_or(j.array(a, j.float32), j.array(c, j.float32), j.ones((J, 3))), np.float64)[0]
        gt = (lambda x: sig(beta * (.5 - x))) if gate else (lambda x: 1.)
        R, T, H = R + gt(R) * O[0], T + gt(T) * O[1], H + gt(H) * O[2]
        if l == L - 1:
            F = F + np.maximum(H + 2 * E - 2, 0)
    return F


def sum_layers(A, g, L, noise=0., rng=None):
    """Theorem 1's construction (sum over k, ReLU clip, n + 1 bias)."""
    n = len(A)
    R = ((np.eye(n) + A) > 0).astype(np.float64)
    Gm = np.zeros((n, n))
    Gm[:, g] = 1
    Z = [R, R.T.copy(), A.astype(np.float64), Gm]
    if noise:
        Z = [z + rng.uniform(-noise, noise, z.shape) for z in Z]
    R, T, E, H = Z
    F = np.zeros((n, n))
    relu = lambda x: np.maximum(x, 0)
    for l in range(L):
        R2, T2, H2 = R + R @ R, T + T @ T, H + H @ T
        R, T, H = R2 - relu(R2 - 1), T2 - relu(T2 - 1), H2 - relu(H2 - 1)
        if l == L - 1:
            F = F + relu(H2 + (n + 1) * (E - 1))
    return F


def graphs_bench():
    """The benchmark's own test questions at 49 vertices: (adjacency over names 0..V, goal, current vertex, answer)."""
    d = json.loads((HERE / 'export' / 'graphs_v49.json').read_text())
    V = d['meta']['vertices']
    out = []
    for q in d['questions']:
        A = np.zeros((V + 1, V + 1))
        for u, v in q['edges']:
            A[u, v] = 1
        out.append((A, q['goal'], q['path'][-1], q['answer'], q['lookahead']))
    return out


def graphs_random(rng, n, count):
    out = []
    for _ in range(count):
        p = rng.uniform(1.5, 4.) / n  # sparse DAGs with long paths, as the benchmark's
        A = np.triu(rng.random((n, n)) < p, 1).astype(float)
        perm = rng.permutation(n)
        out.append((A[np.ix_(perm, perm)], int(rng.integers(n)), None, None, None))
    return out


def score(graphs, layers_fn, Ls, **kw):
    """Entries (c, v) where read-out > 0 disagrees with 'v is a child of c from which g is reachable in < 2^L steps'."""
    bad = {L: 0 for L in Ls}
    tot = {L: 0 for L in Ls}
    answer_ok = {L: 0 for L in Ls}
    asked = 0
    for A, g, cur, ans, look in graphs:
        D = dists(A)
        for L in Ls:
            F = layers_fn(A, g, L, **kw)
            truth = (A == 1) & (D[:, g][None, :] <= 2 ** L - 1)
            bad[L] += int(((F - .5 > 0) != truth).sum())
            tot[L] += truth.size
            if ans is not None:
                answer_ok[L] += int(F[cur, ans] - .5 > 0)
        asked += ans is not None
    r = dict(entries={str(L): tot[L] for L in Ls}, disagreements={str(L): bad[L] for L in Ls})
    if asked:
        r['benchmark_answer_marked'] = {str(L): f'{answer_ok[L]}/{asked}' for L in Ls}
    return r


def theory(rng):
    out = {}
    bench = graphs_bench()
    Ls = list(range(1, 8))
    t0 = time.time()
    out['bench49_or'] = score(bench, or_layers, Ls)
    out['bench49_or_J1'] = score(bench, or_layers, Ls, J=1)
    out['bench49_sum'] = score(bench, sum_layers, Ls)
    for n in (125, 250, 500):
        G = graphs_random(rng, n, 6 if n < 500 else 3)
        Lr = [1, 2, 3, math.ceil(math.log2(n)) + 1]
        out[f'random{n}_or'] = score(G, or_layers, Lr)
        out[f'random{n}_or_nogate'] = score(G, or_layers, Lr, gate=False)
    # the smallest input gain alpha (gate gain beta = alpha) with no disagreement, against n
    gains = {}
    for n in (49, 125, 250, 500):
        G = bench[::10] if n == 49 else graphs_random(rng, n, 3)
        Lr = [math.ceil(math.log2(n)) + 1]
        for alpha in (6, 8, 10, 12, 14, 16, 18, 20, 24, 30):
            r = score(G, or_layers, Lr, alpha=float(alpha), beta=float(alpha))
            if sum(r['disagreements'].values()) == 0:
                gains[str(n)] = alpha
                break
        else:
            gains[str(n)] = '>30'
    out['min_gain_alpha_exact'] = gains
    # graded starting states: every entry of R, T, E, H moved by uniform noise in [-delta, delta]
    noisy = {}
    for n, G in (('49 (benchmark)', bench[::10]), ('250 (random)', graphs_random(rng, 250, 3))):
        Lr = [math.ceil(math.log2(int(n.split()[0]))) + 1]
        for delta in (.05, .1, .2, .3, .4):
            o = score(G, or_layers, Lr, noise=delta, rng=np.random.default_rng(1))
            s = score(G, sum_layers, Lr, noise=delta, rng=np.random.default_rng(1))
            noisy[f'n={n} delta={delta}'] = dict(or_disagree=sum(o['disagreements'].values()), sum_disagree=sum(s['disagreements'].values()),
                                                  entries=sum(o['entries'].values()))
    out['noisy_start'] = noisy
    out['seconds'] = round(time.time() - t0)
    return out


if __name__ == '__main__':
    orig = sys.argv[1]
    parts = sys.argv[2:] or ['series', 'boolean', 'same', 'smoke', 'theory']
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in parts:
        rng = np.random.default_rng(0)
        t = time.time()
        res[part] = same(orig, rng) if part == 'same' else globals()[part](rng)
        print(part, round(time.time() - t), 's', json.dumps(res[part])[:3000], flush=True)
        OUT.write_text(json.dumps(res, indent=1))
