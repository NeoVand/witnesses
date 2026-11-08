"""S22 local checks for the propagator join.

1. same:       with the options off, search_train.py gives a reference copy's parameters and read-outs bit for
               bit (regression check; needs that copy) (newton_or_checks.same: S8 loop, S8 fixed8, S9, S16, S17, stored S8/S9 weights; plus S20 Newton-OR,
               S20 power mean, S21 rate form and the stored S21 weights).
2. exact:      hand-set gates G = t0 A through search_train.propagator itself (logits +-1e4, so sigmoid is exactly 0 or 1):
               R_uv > 0 iff v is reachable from u; round(log R_uv / log t0) against the BFS distance (unique shortest
               paths, and all); float32 and float64; the benchmark's test graphs at 49, 125, 250 vertices and random DAGs
               up to 1,000. And the task: source S_uk = [k = goal], G = t0 A^T, so X_cv = walks from v to the goal; the
               decision A_cv and X_cv > 0 against "v is a child of c from which the goal is reachable", every (c, v).
3. background: gates eps on every non-edge among present names, G = t0 A + eps (11^T - A). The decision (the answer's
               R_{v*g} above every other child's) fails at eps*, found by bisection (float64 solve) and predicted by the
               Sherman-Morrison formula eps* = R0_{v*g} / (c_g max(r_v' - r_v*) + S R0_{v*g}), R0 = (I - t0 A)^{-1},
               r = R0 1, c = R0^T 1, S = 1^T R0 1 (and eps < 1/S always). Also a source background s on k != goal.
python3 -I propagator_checks.py <reference search_train.py> <their train.py> [part ...]; writes results/s22_checks.json.
"""
import json, math, os, sys, time
from collections import deque
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get('S22_OUT', HERE / 'results' / 's22_checks.json'))  # the GPU copy is written beside the bench job
T0S = (1e-3, .01, .05, .1, .2, .3, .5, .7, .9, .99)


# ---------------------------------------------------------------------------------------------------- graphs
def bench_graphs(train_py, V, per, seed=12345):
    """The benchmark's own test questions (l2s_eval.test_sets, their code): adjacency over names 0..V, present names,
    goal, current vertex, answer, lookahead (None = their random graphs)."""
    import l2s_eval
    ns = l2s_eval.load(train_py)
    sets = l2s_eval.test_sets(ns, 3 * V + 5, per, rng_seed=seed)
    out = []
    for L, (xs, ys) in sets.items():
        for x, y in zip(xs, ys):
            toks = [int(t) for t in x if t != V + 3]
            q = toks.index(V + 4)
            A = np.zeros((V + 1, V + 1))
            for i in range(0, q, 3):
                if toks[i] == V + 2:
                    A[toks[i + 1], toks[i + 2]] = 1
            present = np.zeros(V + 1)
            present[[t for t in toks if 1 <= t <= V]] = 1
            out.append(dict(A=A, present=present, goal=toks[q + 2], cur=toks[-1], answer=int(y), look=L))
    return out


def random_dags(rng, n, count):
    out = []
    for _ in range(count):
        p = rng.uniform(1.5, 4.) / n  # sparse DAGs with long paths, as in the S20 theory check
        A = np.triu(rng.random((n, n)) < p, 1).astype(float)
        perm = rng.permutation(n)
        out.append(dict(A=A[np.ix_(perm, perm)], present=np.ones(n), goal=None, cur=None, answer=None, look=None))
    return out


def bfs_all(A):
    """d[u, v] (inf if unreachable) and the number of shortest paths cnt[u, v], by BFS from every vertex."""
    n = len(A)
    nbr = [np.flatnonzero(A[u]) for u in range(n)]
    d = np.full((n, n), np.inf)
    cnt = np.zeros((n, n))
    for s in range(n):
        d[s, s], cnt[s, s] = 0, 1
        q = deque([s])
        while q:
            u = q.popleft()
            for w in nbr[u]:
                if d[s, w] == np.inf:
                    d[s, w] = d[s, u] + 1
                    q.append(w)
                if d[s, w] == d[s, u] + 1:
                    cnt[s, w] += cnt[s, u]
    return d, cnt


# ---------------------------------------------------------------------------------------------------- 1. same
def same(orig_path):
    import jax, jax.numpy as j
    import search_train as st, newton_or_checks as chk
    out = chk.same(orig_path, np.random.default_rng(0))
    old = chk.load_orig(orig_path)
    PAIRN = dict(kind='pairn', emb=32, D=16, layers=1, r=8, loop=True, nopos=True)
    cfgs = {'s20-loopnewton': dict(PAIRN, tri='newton'), 's20-looppmean': dict(PAIRN, tri='pmean'), 's21-looprate': dict(PAIRN, tri='newton', J=1)}
    rng = np.random.default_rng(0)
    V, P = 9, 16
    n, pad = 3 * V + 5, V + 3
    x = rng.integers(0, V + 5, (4, n))
    x[:, -1] = rng.integers(1, V + 1, 4)
    e = np.where(x <= V, x, x - V - 1 + P + 1)
    for name, cfg in cfgs.items():
        res = {}
        for seed in (0, 1):
            p0, p1 = old.init(jax.random.PRNGKey(seed), cfg, P + 5, n), st.init(jax.random.PRNGKey(seed), cfg, P + 5, n)
            l0, l1 = jax.tree_util.tree_flatten_with_path(p0)[0], jax.tree_util.tree_flatten_with_path(p1)[0]
            same_tree = [jax.tree_util.keystr(k) for k, _ in l0] == [jax.tree_util.keystr(k) for k, _ in l1]
            pdiff = max(float(j.max(j.abs(a - b))) for (_, a), (_, b) in zip(l0, l1)) if same_tree else None
            for bf16 in (True, False):
                c = dict(cfg, iters=6, bf16=bf16, remat=bf16)
                outs = []
                for mod in (old, st):
                    z, present, nv, Nn = mod.pairn_start(p0, j.array(x), pad, p0['tok'][j.array(e)])
                    outs.append(mod.pairn_read(p0, mod.pair_stack(p0, c, z, present, nv), j.array(x), present, Nn, pad))
                res[f'seed{seed}_bf16{int(bf16)}'] = dict(same_param_tree=same_tree, max_param_diff=pdiff, max_logit_diff=float(j.max(j.abs(outs[0] - outs[1]))))
        out[name] = res
    f = HERE / 'results' / 's21-looprate-v49-s0' / 'params_s0.npz'
    if f.exists():
        cfg = cfgs['s21-looprate']
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
        out['s21-looprate_stored_weights'] = dict(questions=len(qs), max_logit_diff=float(j.max(j.abs(outs[0] - outs[1]))),
                                                  answers_right=int((said[1] == np.array([q['answer'] for q in qs])).sum()))
    return out


# ---------------------------------------------------------------------------------------------------- 2. exact
def handset_R(As, present, t0, dtype):
    """R = (I - t0 A)^{-1} through search_train.propagator (psrc off): t has one channel, A; logits 2e4 A - 1e4."""
    import jax.numpy as j
    import search_train as st
    t = j.array(As[:, None], dtype)
    b = dict(pg=j.array([[2e4]], dtype), pgb=j.array([-1e4], dtype), ps=j.zeros((1, 1), dtype), psb=j.zeros((1,), dtype))
    return np.asarray(st.propagator(t, b, j.array(present, dtype), dict(psrc=False, t0=t0)))[:, 0].astype(np.float64)


def handset_task(As, present, goals, t0, dtype):
    """X = S (I - G)^{-1} through search_train.propagator: channel 0 = A^T (gates), channel 1 = goal column (source)."""
    import jax.numpy as j
    import search_train as st
    B, N, _ = As.shape
    gcol = np.zeros((B, N, N))
    gcol[np.arange(B), :, goals] = 1
    t = j.array(np.stack([np.swapaxes(As, 1, 2), gcol], 1), dtype)
    b = dict(pg=j.array([[2e4], [0.]], dtype), pgb=j.array([-1e4], dtype), ps=j.array([[0.], [2e4]], dtype), psb=j.array([-1e4], dtype))
    return np.asarray(st.propagator(t, b, j.array(present, dtype), dict(psrc=True, t0=t0)))[:, 0].astype(np.float64)


def exact(train_py):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as j
    rng = np.random.default_rng(0)
    sets = {'bench49': bench_graphs(train_py, 49, 10), 'bench125': bench_graphs(train_py, 125, 4), 'bench250': bench_graphs(train_py, 250, 2),
            'random125': random_dags(rng, 125, 6), 'random250': random_dags(rng, 250, 6), 'random500': random_dags(rng, 500, 4),
            'random1000': random_dags(rng, 1000, 2)}
    tiny = {'float32': (float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).smallest_subnormal)),
            'float64': (float(np.finfo(np.float64).tiny), float(np.finfo(np.float64).smallest_subnormal))}
    ftz = {k: float(np.asarray(j.array(v[0], k) * j.array(.25, k))) for k, v in tiny.items()}  # 0.0 if XLA flushes subnormals
    out = dict(tiny_normal_subnormal=tiny, xla_cpu_quarter_of_tiny=ftz, sets={})
    for name, G in sets.items():
        t_s = time.time()
        truth = [bfs_all(g['A']) for g in G]
        dmax = int(max(d[np.isfinite(d)].max() for d, _ in truth))
        indeg = int(max(g['A'].sum(0).max() for g in G))
        outdeg = int(max(g['A'].sum(1).max() for g in G))
        rows = dict(graphs=len(G), vertices=len(G[0]['A']) - (name.startswith('bench')), max_distance=dmax, max_in_degree=indeg, max_out_degree=outdeg, t0={})
        As = np.stack([g['A'] for g in G])
        pres = np.stack([g['present'] for g in G])
        for t0 in T0S:
            for dt in ('float32', 'float64'):
                R = np.concatenate([handset_R(As[i:i + 8], pres[i:i + 8], t0, dt) for i in range(0, len(G), 8)])
                rb = db_u = db_a = dc_a = n_u = n_a = under = 0
                fp_max, sig_min = 0., math.inf
                for (d, cnt), Ri in zip(truth, R):
                    reach = np.isfinite(d)
                    rb += int(((Ri > 0) != reach).sum())
                    fp_max = max(fp_max, float(np.abs(Ri[~reach]).max()) if (~reach).any() else 0.)
                    sig_min = min(sig_min, float(Ri[reach].min()))
                    under += int((reach & (Ri <= 0)).sum())
                    with np.errstate(divide='ignore', invalid='ignore'):
                        x_ = np.log(Ri) / math.log(t0)
                        dh = np.rint(x_)
                        dc = np.ceil(x_ - .01)  # Jing and Meister's ceil(log_t0 R), with a 0.01 guard against rounding up an exact t0^d
                    ok = reach & (Ri > 0)
                    uniq = reach & (cnt == 1)
                    db_u += int((uniq & ~(ok & (dh == d))).sum())
                    db_a += int((reach & ~(ok & (dh == d))).sum())
                    dc_a += int((reach & ~(ok & (dc == d))).sum())
                    n_u += int(uniq.sum())
                    n_a += int(reach.sum())
                r = dict(reach_disagree=rb, reachable_underflowed=under, max_abs_R_unreachable=fp_max, min_R_reachable=sig_min,
                         threshold_separates=bool(sig_min > fp_max), dist_wrong_unique=db_u, unique_pairs=n_u, dist_wrong_all=db_a, dist_ceil_wrong_all=dc_a, reachable_pairs=n_a)
                if name.startswith('bench'):
                    goals = np.array([g['goal'] for g in G])
                    X = np.concatenate([handset_task(As[i:i + 8], pres[i:i + 8], goals[i:i + 8], t0, dt) for i in range(0, len(G), 8)])
                    bad = right = 0
                    for g, (d, _), Xi in zip(G, truth, X):
                        want = (g['A'] == 1) & np.isfinite(d[:, g['goal']])[None, :]
                        F = (g['A'] == 1) & (Xi > 0)
                        bad += int((F != want).sum())
                        sc = np.where(g['A'][g['cur']] == 1, np.log(Xi[g['cur']] + 1e-300), -np.inf)
                        right += int(sc.argmax() == g['answer'] and sc.max() > -np.inf and (sc == sc.max()).sum() == 1)
                    r |= dict(task_decision_disagree=bad, task_entries=int(As.size), task_answer_right=f'{right}/{len(G)}')
                rows['t0'][f'{t0} {dt}'] = r
        rows['seconds'] = round(time.time() - t_s)
        out['sets'][name] = rows
        print(name, json.dumps(rows)[:1500], flush=True)
    return out


# ---------------------------------------------------------------------------------------------------- 3. background
def decision_ok(A, pres, goal, cur, ans, t0, eps):
    """With gates G = t0 A + eps (11^T - A) on present names (float64 solve), is the answer's walk weight to the goal
    strictly the largest among the current vertex's children? Returns (ok, finite)."""
    p = pres > 0
    G = (t0 * A + eps * (1 - A)) * np.outer(p, p)
    R = np.linalg.solve(np.eye(len(A)) - G, np.eye(len(A))[:, goal])  # column R[:, goal]
    ch = np.flatnonzero(A[cur])
    ok = bool(all(R[ans] > R[v] for v in ch if v != ans))
    return ok, bool(np.all(np.isfinite(R)) and R.min() >= -1e-9 * np.abs(R).max())  # past the pole (eps S >= 1) R turns negative


def eps_formula(A, pres, goal, cur, ans, t0):
    """Sherman-Morrison: R = R0 + eps R0 1 1^T R0 / (1 - eps S), R0 = (I - (t0 - eps) A)^{-1}; to first order (R0 at
    eps = 0) the decision fails at eps* = R0_{v*g} / (c_g max_v' (r_v' - r_v*) + S R0_{v*g}), and in any case at 1/S."""
    p = pres > 0
    R0 = np.linalg.inv(np.eye(len(A)) - t0 * A)
    r, c = R0[:, p].sum(1), R0[p].sum(0)
    S = R0[np.ix_(p, p)].sum()
    ch = [v for v in np.flatnonzero(A[cur]) if v != ans]
    gap = max([r[v] - r[ans] for v in ch] + [0.])
    sig = R0[ans, goal]
    return sig / (c[goal] * gap + S * sig), 1 / S, S, sig


def eps_bisect(g, t0, lo=-200., hi=0.):
    """Largest log10 eps with a correct decision (bisection on log10 eps; correct at 10^lo assumed and checked)."""
    args = (g['A'], g['present'], g['goal'], g['cur'], g['answer'], t0)
    if not decision_ok(*args, 10. ** lo)[0]:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        ok, fin = decision_ok(*args, 10. ** mid)
        lo, hi = (mid, hi) if ok and fin else (lo, mid)
    return lo


def source_formula(A, goal, cur, ans, t0):
    """Source background s on k != goal (gates exact): X_cv = R_vg + s (r_v - R_vg), r = R 1. Fails at
    s* = R_{v*g} / max_v' (r_v' - r_v* + R_{v*g}) (exact, linear in s)."""
    R = np.linalg.inv(np.eye(len(A)) - t0 * A)
    r = R.sum(1)
    ch = [v for v in np.flatnonzero(A[cur]) if v != ans]
    den = max([r[v] - r[ans] + R[ans, goal] for v in ch] + [0.])
    return R[ans, goal] / den if den > 0 else math.inf


def background(train_py):
    out = {}
    for V, per in ((49, 4), (125, 2), (250, 1)):
        G = [g for g in bench_graphs(train_py, V, per) if g['look'] is not None]
        for t0 in (.5, .9, .99):
            t_s = time.time()
            meas, pred, pred_cap, Ss, src, ratio, by_look, sigs = [], [], [], [], [], [], {}, []
            for g in G:
                e_f, cap, S, sig = eps_formula(g['A'], g['present'], g['goal'], g['cur'], g['answer'], t0)
                lb = eps_bisect(g, t0)
                if lb is None:
                    continue
                meas.append(lb)
                pred.append(math.log10(min(e_f, cap)))
                sigs.append(sig)
                Ss.append(S)
                ratio.append(lb - math.log10(min(e_f, cap)))
                by_look.setdefault(g['look'], []).append(lb)
                src.append(math.log10(source_formula(g['A'], g['goal'], g['cur'], g['answer'], t0)))
            m = np.array(meas)
            looks = sorted(by_look)
            dev = np.abs(np.array(ratio))
            big = dev > .1  # formula and solve disagree by more than 0.1 decade
            sg = np.array(sigs)
            out[f'V={V} t0={t0}'] = dict(
                questions=len(m), log10_eps_star_min=float(m.min()), log10_eps_star_median=float(np.median(m)),
                formula_minus_measured_log10_max_abs=float(np.max(np.abs(ratio))),
                formula_vs_solve_disagree_over_0p1_decade=int(big.sum()), largest_signal_where_disagree=float(sg[big].max()) if big.any() else None,
                formula_minus_measured_log10_max_abs_signal_over_1e_12=float(dev[sg > 1e-12].max()) if (sg > 1e-12).any() else None,
                log10_eps_star_formula_min=float(np.min(pred)), log10_eps_star_formula_median=float(np.median(pred)), log10_one_over_S_median=float(np.median(-np.log10(Ss))),
                S_median=float(np.median(Ss)), S_max=float(np.max(Ss)), present_names=int(np.median([g['present'].sum() for g in G])),
                log10_eps_star_by_lookahead={str(L): float(min(by_look[L])) for L in (looks[0], looks[len(looks) // 2], looks[-1])},
                log10_source_star_min=float(np.min(src)), log10_source_star_median=float(np.median(src)), seconds=round(time.time() - t_s))
            print(V, t0, json.dumps(out[f'V={V} t0={t0}']), flush=True)
    return out


def background_random(train_py):
    """Random DAGs of 125 to 1,000 vertices, the all-pairs reach decision by one global threshold: it survives eps
    while min over reachable R > max over unreachable R; Sherman-Morrison gives eps* = m0 / (max_unreach r_v c_g + S m0),
    m0 = min reachable R0 (first order), and eps < 1/S. Bisection on the float64 inverse."""
    rng = np.random.default_rng(1)
    out = {}
    for n, cnt in ((125, 3), (250, 3), (500, 2), (1000, 2)):
        for t0 in (.5, .9):
            rows = []
            for g in random_dags(rng, n, cnt):
                A = g['A']
                d, _ = bfs_all(A)
                reach = np.isfinite(d)
                R0 = np.linalg.inv(np.eye(n) - t0 * A)
                r, c, S = R0.sum(1), R0.sum(0), R0.sum()
                m0 = R0[reach].min()
                pred = m0 / (np.outer(r, c)[~reach].max() + S * m0)

                def ok(eps):
                    R = np.linalg.inv(np.eye(n) - (t0 - eps) * A - eps * np.ones((n, n)))
                    return bool(np.all(np.isfinite(R)) and R.min() >= -1e-9 * np.abs(R).max() and R[reach].min() > R[~reach].max())
                lo, hi = -60., 0.
                assert ok(10. ** lo)
                for _ in range(30):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if ok(10. ** mid) else (lo, mid)
                rows.append(dict(log10_eps_star=lo, log10_formula=float(math.log10(pred)), log10_one_over_S=float(-math.log10(S)), S=float(S),
                                 max_distance=int(d[reach].max()), log10_min_reach_R0=float(math.log10(m0))))
            out[f'n={n} t0={t0}'] = rows
            print('random', n, t0, json.dumps(rows), flush=True)
    return out


def rowcap(train_py):
    """The row-sum cap (runs s22c) through search_train.propagator (cfg rowcap, t0 = 0.99), task layer (source =
    goal column, gates = A^T plus eps on every non-edge among present names, rows capped at t0). Per size and eps: the
    benchmark answer is the unique arg max over the current vertex's children; the largest spectral radius; the smallest
    answer signal log10 X_cv*. float32 and float64."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as j
    import search_train as st
    out = {}
    for V, per in ((49, 4), (125, 2), (250, 1)):
        G_ = [g for g in bench_graphs(train_py, V, per) if g['look'] is not None]
        for eps in (0., 1e-4, 1e-3, 1e-2):
            for dt in ('float32', 'float64'):
                right, rho_max, sig_min = 0, 0., math.inf
                for g in G_:
                    N = len(g['A'])
                    gcol = np.zeros((N, N))
                    gcol[:, g['goal']] = 1
                    lb = math.log(eps / (1 - eps)) if eps else -1e4  # sigmoid(lb) = eps; edges at +1e4 (sigmoid 1)
                    t = j.array(np.stack([g['A'].T, gcol])[None], dt)
                    b = dict(pg=j.array([[1e4 - lb], [0.]], dt), pgb=j.array([lb], dt), ps=j.array([[0.], [2e4]], dt), psb=j.array([-1e4], dt))
                    cfg = dict(psrc=True, t0=.99, rowcap=True)
                    X = np.asarray(st.propagator(t, b, j.array(g['present'][None], dt), cfg))[0, 0].astype(np.float64)
                    pm = np.outer(g['present'], g['present'])
                    Gm = .99 * np.where(g['A'].T == 1, 1., eps) * pm
                    Gm = Gm * np.minimum(1., .99 / np.maximum(Gm.sum(1, keepdims=True), 1e-30))
                    rho_max = max(rho_max, float(np.abs(np.linalg.eigvals(Gm)).max()))
                    ch = np.flatnonzero(g['A'][g['cur']])
                    xc = X[g['cur']]
                    right += int(all(xc[g['answer']] > xc[v] for v in ch if v != g['answer']))
                    sig_min = min(sig_min, float(xc[g['answer']]))
                out[f'V={V} eps={eps} {dt}'] = dict(questions=len(G_), answer_right=right, rho_max=rho_max, log10_min_answer_signal=math.log10(sig_min) if sig_min > 0 else None)
                print('rowcap', V, eps, dt, out[f'V={V} eps={eps} {dt}'], flush=True)
    return out


def background_f32(train_py):
    """Through search_train.propagator in float32: gates eps on non-edges at eps = eps*/4 and 4 eps* (formula eps*),
    on the 250-vertex benchmark questions; the decision should hold below and fail above."""
    import jax.numpy as j
    import search_train as st
    G = [g for g in bench_graphs(train_py, 250, 1) if g['look'] is not None]
    out = {}
    for t0 in (.9, .99):
        res = {'0.25': 0, '4': 0}
        for g in G:
            e_f, cap, _, _ = eps_formula(g['A'], g['present'], g['goal'], g['cur'], g['answer'], t0)
            es = min(e_f, cap)
            for k, f in (('0.25', .25), ('4', 4.)):
                eps = es * f
                lb = math.log(eps / t0 / (1 - eps / t0))  # sigmoid(lb) = eps / t0
                N = len(g['A'])
                gcol = np.zeros((N, N))
                gcol[:, g['goal']] = 1
                t = j.array(np.stack([g['A'].T, gcol])[None], j.float32)
                b = dict(pg=j.array([[1e4 - lb], [0.]], j.float32), pgb=j.array([lb], j.float32), ps=j.array([[0.], [2e4]], j.float32), psb=j.array([-1e4], j.float32))
                X = np.asarray(st.propagator(t, b, j.array(g['present'][None], j.float32), dict(psrc=True, t0=t0)))[0, 0]
                ch = np.flatnonzero(g['A'][g['cur']])
                res[k] += int(all(X[g['cur'], g['answer']] > X[g['cur'], v] for v in ch if v != g['answer']))
        out[f't0={t0}'] = dict(questions=len(G), right_at_quarter_eps_star=res['0.25'], right_at_4x_eps_star=res['4'])
        print('f32', t0, out[f't0={t0}'], flush=True)
    return out


if __name__ == '__main__':
    orig, train_py = sys.argv[1], sys.argv[2]
    parts = sys.argv[3:] or ['same', 'exact', 'background', 'background_random', 'background_f32']
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for part in parts:
        t = time.time()
        res[part] = same(orig) if part == 'same' else globals()[part](train_py)
        print(part, round(time.time() - t), 's', flush=True)
        OUT.write_text(json.dumps(res, indent=1))
