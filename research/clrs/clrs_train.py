"""One looped pair layer on the CLRS-30 graph tasks, trained without hints (JAX).

Pair state z_ij in R^D over the nodes of the graph (channel-first [B, D, N, N]).
Start. z_ij = MLP(e_ij) + E e_ij + R f_i + C f_j, with edge features e_ij = (A_ij, A_ji, [A_ij != 0], [A_ji != 0],
adj_ij, adj_ji, [i = j]) (the weights enter through the MLP, width 32) and node features f_i = (pos_i, s_i), s the
source mask where the task has one; the graph has no graph features in these tasks.
Block (default, triangle attention), four heads, with biases on its linear maps: t = norm(z);
z += sigmoid(G t) * sum_k softmax_k(Q t_uk . K t_kv / sqrt(dh)) (V1 t_uk) * (V2 t_kv); z += MLP(norm(z)).
Padded nodes are never partners k. One block, weights shared, run K times; K is drawn per batch from train_iters; the
read-out after run K of one long run equals the read-out of a run of K, so one pass scores every K.
Model key agg replaces triangle attention by an unnormalised sum over the middle vertex k (agg_block), with
a = sigma(A t + b_a), c = sigma(B t + b_b) (0 for padded k):
- agg = 'newton': the OR join, O = 1 - exp(-L), L = sum_{j=1..J} om_j (sum_k a_uk^j c_kv^j) / j (J = 8 by default, the
  first J terms of -sum_k log(1 - a_uk c_kv); om initialised to 1); agg = 'pmean': the power mean
  (sum_k (a_uk c_kv)^beta)^(1/beta), beta per channel, initialised to 4, floored at 1/4;
- agg = 'mix': channels 0..D/2-1 the OR join, D/2..D-1 the power mean, from the same a and c;
- logn = True: both input sigmoids shifted, a = sigma(A t + b_a - kappa log n), c likewise, kappa learned per channel
  (initialised to 1/2), n the graph's real nodes;
then z += sigma(G t + b_g) * (W_o y + b_o); z += MLP(norm(z)).
Read-outs. Node pointer (pi, topo): p(j | i) = softmax_j w . norm(z_ij). Pair pointer (floyd_warshall Pi):
p(k | i, j) = softmax_k (P norm(z_ik) + p) . (R norm(z_kj) + r) / sqrt(D). Node mask_one (topo_head): softmax_i of
w' . norm(z_ii). Masks: node mask (is_cut) sigmoid(u . norm(z_ii) + b); edge mask (is_bridge, in_mst, both symmetric
in the library) sigmoid of (l_ij + l_ji) / 2 with l_ij = u . norm(z_ij) + b; one (u, b) per task, D + 1 parameters,
created only for tasks with a mask output, from a separate key (the other parameters' random draws are unchanged).
Model key kirch = H (heads), keps = eps: one Kirchhoff read-out after the loop, node pointers only.
t = norm(z); per head h, conductances c_uv = sigma((l_uv + l_vu) / 2), l_uv = w_h . t_uv + b_h, zero off the input graph
(A != 0), on the diagonal and on padding; L = diag(C 1) - C grounded at s (row and column replaced by the identity) plus
eps I; M = L_s^{-1} (one batched float32 solve); phi_u = M_uu (effective resistance s..u, depth on a unit tree),
R_uv = M_uu + M_vv - 2 M_uv. Features o_uv = clip((phi_u - phi_v) / (R_uv + 1e-6), -1, 1) (in [-1, 1] because the
effective resistance is a metric; +1 when v lies between u and s) and q_uv = 1 / (1 + R_uv), zero on the diagonal and
padding; z' = z + sigma(G_k t + g_k) * (W_k f + b_k), then the pointer read-out on norm(z'). Model key klogn = True adds
the learned log n shift to the conductance logits, l_uv - kappa_h log n (kappa per head, initialised to 1/2).
Loss: cross-entropy on the outputs only (no hints), averaged over real nodes (or pairs); for masks binary cross-entropy
over the entries the library does not mark -1 (its own loss mask: is_bridge marks non-edges -1).
Job flag depth_mask (off in every run reported here): for node-pointer outputs the loss on node v is dropped in a batch
run K times when v's depth in the true answer tree exceeds 2^K.
Job keys extra_splits (more splits in the task's data folder, predicted and scored like test) and params_from (score
saved parameters, path with {seed}, instead of training).
Every option is off when its key is absent.
python clrs_train.py '<json>' prints one JSON line; writes rows.json, progress.jsonl, preds_s<seed>.npz, params to
LAB_DEST.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
jax.config.update('jax_default_matmul_precision', 'highest')  # no TF32: pointer targets are argmins of sums of weights
DEST = Path(os.environ.get('LAB_DEST', '.'))
OUTS = {'bfs': [('pi', 'node')], 'bellman_ford': [('pi', 'node')], 'dijkstra': [('pi', 'node')], 'dag_shortest_paths': [('pi', 'node')],
        'mst_prim': [('pi', 'node')], 'floyd_warshall': [('Pi', 'edge')], 'topological_sort': [('topo', 'node'), ('topo_head', 'one')],
        'dfs': [('pi', 'node')], 'strongly_connected_components': [('scc_id', 'node')], 'articulation_points': [('is_cut', 'nmask')],
        'bridges': [('is_bridge', 'emask')], 'mst_kruskal': [('in_mst', 'emask')]}
MASKS = ('nmask', 'emask')
NE, NF = 7, 2  # edge and node feature counts


def cnorm(t):
    return (t - t.mean(1, keepdims=True)) * jax.lax.rsqrt(t.var(1, keepdims=True) + 1e-6)


def lin(t, w, b=None):
    y = j.einsum('bcij,cd->bdij', t, w)
    return y if b is None else y + b[None, :, None, None]


def init(key, cfg, outs=()):
    D, M, W = cfg['D'], cfg.get('mlp', 2 * cfg['D']), cfg.get('wmlp', 32)
    ks = iter(jax.random.split(key, 32))
    g = lambda *s: jax.random.normal(next(ks), s) * s[0] ** -.5
    z = lambda *s: j.zeros(s)
    p = dict(e1=g(NE, W), eb1=z(W), e2=g(W, D), E=g(NE, D), eb=z(D), R=g(NF, D), C=g(NF, D))
    p['blk'] = dict(q=g(D, D), k=g(D, D), v1=g(D, D), v2=g(D, D), g=g(D, D), f1=g(D, M), f2=g(M, D),
                    bq=z(D), bk=z(D), bv1=z(D), bv2=z(D), bg=z(D), bf1=z(M), bf2=z(D))
    if cfg.get('agg'):  # power mean or OR join over k in place of triangle attention; own key, so the draws
        ka = iter(jax.random.split(jax.random.fold_in(key, 20), 8))  # of every other parameter are as without it
        ga = lambda *s: jax.random.normal(next(ka), s) * s[0] ** -.5
        p['blk'] = dict(a=ga(D, D), b=ga(D, D), g=ga(D, D), o=ga(D, D), f1=ga(D, M), f2=ga(M, D),
                        ba=z(D), bb=z(D), bg=z(D), bo=z(D), bf1=z(M), bf2=z(D))
        if cfg['agg'] == 'pmean':
            p['blk']['beta'] = j.full((D,), 4.0)
        elif cfg['agg'] == 'mix':  # channels 0..D/2-1 OR join, D/2..D-1 power mean
            p['blk']['om'] = j.ones((cfg.get('J', 8), D // 2))
            p['blk']['beta'] = j.full((D - D // 2,), 4.0)
        else:
            p['blk']['om'] = j.ones((cfg.get('J', 8), D))
        if cfg.get('logn'):  # a = sigma(A t + b_a - kappa log n), c likewise, kappa per channel, initialised to 1/2
            p['blk']['kap'] = j.full((D,), .5)
    p['w_node'] = jax.random.normal(next(ks), (D,)) * D ** -.5
    p['w_one'] = jax.random.normal(next(ks), (D,)) * D ** -.5
    p |= dict(P=g(D, D), bP=z(D), Rr=g(D, D), bR=z(D))
    if any(kind in MASKS for _, kind in outs):  # drawn from a separate key, so the draws above are unchanged
        p |= dict(w_mask=jax.random.normal(jax.random.fold_in(key, 7), (D,)) * D ** -.5, b_mask=j.zeros(()))
    if cfg.get('kirch'):  # the Kirchhoff read-out, from its own key, so every other draw is as without it
        H = cfg['kirch']
        kk = iter(jax.random.split(jax.random.fold_in(key, 15), 4))
        gk = lambda *s: jax.random.normal(next(kk), s) * s[0] ** -.5
        p['kir'] = dict(w=gk(D, H), b=z(H), g=gk(D, D), bg=z(D), o=gk(2 * H, D), bo=z(D))
        if cfg.get('klogn'):  # learned log n shift on the conductance logits, kappa per head, initialised to 1/2
            p['kir']['kap'] = j.full((H,), .5)
    return p


def embed(p, x):
    """x: A, adj [B, N, N]; pos, s1, m [B, N] -> z [B, D, N, N]."""
    A, adj, m = x['A'], x['adj'], x['m']
    N = A.shape[-1]
    has = (A != 0).astype(A.dtype)
    eye = j.broadcast_to(j.eye(N, dtype=A.dtype), A.shape)
    T = lambda t: t.swapaxes(1, 2)
    e = j.stack([A, T(A), has, T(has), adj, T(adj), eye], -1)
    h = jax.nn.gelu(e @ p['e1'] + p['eb1']) @ p['e2'] + e @ p['E'] + p['eb']
    f = j.stack([x['pos'], x['s1']], -1)
    z = h + (f @ p['R'])[:, :, None] + (f @ p['C'])[:, None]
    z = z * (m[:, :, None] * m[:, None, :])[..., None]
    return z.transpose(0, 3, 1, 2)


def newton_series(a, c, om):
    """L = sum_j om_j (a^j @ c^j) / j, one matmul per order (rematerialised)."""
    term = jax.checkpoint(lambda a, c, q: j.matmul(a ** q, c ** q), static_argnums=(2,))
    return sum(om[q - 1][None, :, None, None] * term(a, c, q) / q for q in range(1, om.shape[0] + 1))


def agg_block(b, z, m, agg):
    """The block with an unnormalised sum over k. a = sigma(A t + b_a), c = sigma(B t + b_b) (0 for padded k);
    pmean: M = (sum_k (a_uk c_kv)^beta)^(1/beta); newton: O = 1 - exp(-sum_j om_j P_j / j);
    z += sigma(G t + b_g) * (W_o M + b_o); z += MLP(norm(z))."""
    t = cnorm(z)
    xa, xc = lin(t, b['a'], b['ba']), lin(t, b['b'], b['bb'])
    mk, mr = m[:, None, None, :], m[:, None, :, None]
    if 'kap' in b:  # with n real nodes, sum_k a c over near-zero terms becomes a mean, so it keeps one scale at any n
        sh = b['kap'][None, :, None, None] * j.log(j.maximum(m.sum(1), 1.))[:, None, None, None]
        xa, xc = xa - sh, xc - sh
    def pm(xa, xc, beta):
        bt = j.maximum(beta, .25)[None, :, None, None]
        S = j.matmul(j.exp(bt * jax.nn.log_sigmoid(xa)) * mk, j.exp(bt * jax.nn.log_sigmoid(xc)) * mr)
        return j.exp(j.log(S + 1e-30) / bt)

    def nw(xa, xc, om):
        return -j.expm1(-j.maximum(newton_series(jax.nn.sigmoid(xa) * mk, jax.nn.sigmoid(xc) * mr, om), -20.))
    if agg == 'pmean':
        y = pm(xa, xc, b['beta'])
    elif agg == 'mix':  # the Boolean OR on half the channels, the power mean (soft min-plus) on the other half
        h = b['om'].shape[1]
        y = j.concatenate([nw(xa[:, :h], xc[:, :h], b['om']), pm(xa[:, h:], xc[:, h:], b['beta'])], 1)
    else:
        y = nw(xa, xc, b['om'])
    z = z + jax.nn.sigmoid(lin(t, b['g'], b['bg'])) * lin(y, b['o'], b['bo'])
    return z + lin(jax.nn.gelu(lin(cnorm(z), b['f1'], b['bf1'])), b['f2'], b['bf2'])


def block(b, z, m, H):
    t = cnorm(z)
    Bb, D, N, _ = t.shape
    dh = D // H
    sp = lambda u: u.reshape(Bb, H, dh, N, N)
    q, k_ = sp(lin(t, b['q'], b['bq'])), sp(lin(t, b['k'], b['bk']))
    sc = j.einsum('bhcuk,bhckv->bhkuv', q, k_) / dh ** .5
    sc = j.where(m[:, None, :, None, None] > 0, sc, -1e9)
    att = jax.nn.softmax(sc, axis=2)
    v1, v2 = sp(lin(t, b['v1'], b['bv1'])), sp(lin(t, b['v2'], b['bv2']))
    tri = j.einsum('bhkuv,bhcuk,bhckv->bhcuv', att, v1, v2).reshape(Bb, D, N, N)
    z = z + jax.nn.sigmoid(lin(t, b['g'], b['bg'])) * tri
    return z + lin(jax.nn.gelu(lin(cnorm(z), b['f1'], b['bf1'])), b['f2'], b['bf2'])


def kirchhoff(c, s1, m, eps):
    """The grounded Laplacian's inverse. c [B, H, N, N] symmetric conductances (zero off the input graph, on the
    diagonal and on padding), s1 [B, N] the source marker, m [B, N] real nodes. L = diag(C 1) - C with the rows and
    columns of s and of padded nodes replaced by the identity, plus eps I; M = L^{-1} by one batched float32 solve, then
    zeroed on s and padding. Returns phi [B, H, N] (phi_u = M_uu, the effective resistance from s to u, u's depth on a
    tree with unit conductances; phi_s = 0) and R [B, H, N, N] (R_uv = M_uu + M_vv - 2 M_uv, clipped at 0)."""
    N = c.shape[-1]
    I = j.eye(N, dtype=c.dtype)
    keep = (m * (1 - s1))[:, None]  # [B, 1, N]
    kk = keep[..., :, None] * keep[..., None, :]
    L = (c.sum(-1)[..., None] * I - c) * kk + (1 - keep)[..., None] * I + eps * I
    M = j.linalg.solve(L, j.broadcast_to(I, L.shape)) * kk
    phi = j.diagonal(M, axis1=-2, axis2=-1)
    return phi, j.maximum(phi[..., :, None] + phi[..., None, :] - 2 * M, 0.)


def kirch_features(kb, t, x, m, eps):
    """Conductances from the normalised pair state t, then features [o, q] [B, 2H, N, N] (module docstring)."""
    A = x['A']
    N = A.shape[-1]
    off = (1 - j.eye(N, dtype=A.dtype)) * m[:, :, None] * m[:, None, :]
    E = (((A != 0) | (A.swapaxes(1, 2) != 0)).astype(A.dtype) * off)[:, None]
    l = j.einsum('bcij,ch->bhij', t, kb['w']) + kb['b'][None, :, None, None]
    if 'kap' in kb:  # as the block's input gates, - kappa log n
        l = l - kb['kap'][None, :, None, None] * j.log(j.maximum(m.sum(1), 1.))[:, None, None, None]
    c = jax.nn.sigmoid((l + l.swapaxes(-1, -2)) / 2) * E
    phi, R = kirchhoff(c, x['s1'], m, eps)
    o = j.clip((phi[..., :, None] - phi[..., None, :]) / (R + 1e-6), -1., 1.)
    return j.concatenate([o, 1. / (1. + R)], 1) * off[:, None]


def readout(p, z, m, kind, x=None, cfg=None):
    t = cnorm(z)
    if kind == 'node' and cfg is not None and cfg.get('kirch'):  # Kirchhoff read-out
        kb = p['kir']
        f = kirch_features(kb, t, x, m, cfg.get('keps', 1e-4))
        t = cnorm(z + jax.nn.sigmoid(lin(t, kb['g'], kb['bg'])) * lin(f, kb['o'], kb['bo']))
    if kind == 'node':  # [B, i, j]
        return j.where(m[:, None, :] > 0, j.einsum('bcij,c->bij', t, p['w_node']), -1e9)
    if kind == 'one':  # [B, i]
        return j.where(m > 0, j.einsum('bcn,c->bn', j.diagonal(t, axis1=2, axis2=3), p['w_one']), -1e9)
    if kind == 'nmask':  # [B, i] logits of a node mask, from z_ii
        return j.einsum('bcn,c->bn', j.diagonal(t, axis1=2, axis2=3), p['w_mask']) + p['b_mask']
    if kind == 'emask':  # [B, i, j] logits of a symmetric edge mask, from z_ij and z_ji
        l = j.einsum('bcij,c->bij', t, p['w_mask']) + p['b_mask']
        return (l + l.swapaxes(1, 2)) / 2
    P, R = lin(t, p['P'], p['bP']), lin(t, p['Rr'], p['bR'])  # 'edge': [B, i, j, k] from (z_ik, z_kj)
    return j.where(m[:, None, None, :] > 0, j.einsum('bcik,bckj->bijk', P, R) / P.shape[1] ** .5, -1e9)


def decide(lg, kind):
    """Predicted index (pointers, mask_one) or 0/1 (masks: logit > 0, the library's hard decision)."""
    return (lg > 0).astype(j.int32) if kind in MASKS else lg.argmax(-1)


def forward(p, cfg, x, K, outs, every=False):
    """Logits per output after K runs (every=True: predicted indices after each run 1..K, a list)."""
    z = embed(p, x)
    seq = []
    f = jax.checkpoint(block, static_argnums=3) if cfg.get('remat') else block
    if cfg.get('agg'):
        f = jax.checkpoint(agg_block, static_argnums=3) if cfg.get('remat') else agg_block
    for _ in range(K):
        z = f(p['blk'], z, x['m'], cfg['agg'] if cfg.get('agg') else cfg.get('heads', 4))
        if every:  # predicted indices after this run
            seq.append({name: decide(readout(p, z, x['m'], kind, x, cfg), kind) for name, kind in outs})
    return seq if every else {name: readout(p, z, x['m'], kind, x, cfg) for name, kind in outs}


def loss_fn(p, cfg, x, y, K, outs):
    lg = forward(p, cfg, x, K, outs)
    m = x['m']
    tot = 0.
    for name, kind in outs:
        tgt = y[name]
        if kind in MASKS:
            w = (tgt >= 0) * (m if kind == 'nmask' else m[:, :, None] * m[:, None, :])
            t, l = j.maximum(tgt, 0).astype(lg[name].dtype), lg[name]
            tot = tot + ((jax.nn.softplus(l) - l * t) * w).sum() / w.sum()
            continue
        lp = jax.nn.log_softmax(lg[name], -1)
        if kind == 'node':
            w = m
        elif kind == 'edge':
            w = m[:, :, None] * m[:, None, :]
        else:
            w = j.ones(tgt.shape)
        nll = -j.take_along_axis(lp, j.maximum(tgt, 0)[..., None], -1)[..., 0]
        if kind == 'node' and '_depth' in y:  # depth_mask: only nodes within 2^K of the root are asked for after K runs
            w = w * (y['_depth'] <= 2 ** K)
        tot = tot + (nll * w).sum() / w.sum()
    return tot


def to_x(d, idx, rng=None):
    """Model inputs for samples idx of a split; rng: randomise pos as clrs.process_random_pos does (sorted uniforms)."""
    n = d['n'][idx].astype(np.int32)
    N = d['A'].shape[-1]
    m = (np.arange(N)[None] < n[:, None]).astype(np.float32)
    pos = d['pos'][idx].astype(np.float32)
    if rng is not None:
        u = np.where(m > 0, rng.random(m.shape), np.inf)
        pos = np.where(m > 0, np.sort(u, 1), 0.).astype(np.float32)
    s1 = np.zeros_like(m)
    if 's' in d:
        s1[np.arange(len(idx)), d['s'][idx]] = 1.
    return dict(A=d['A'][idx].astype(np.float32), adj=d['adj'][idx].astype(np.float32), pos=pos, s1=s1, m=m)


def to_y(d, idx, outs):
    y = {name: d[name][idx].astype(np.int32) for name, _ in outs}
    if '_depth' in d:  # depth_mask
        y['_depth'] = d['_depth'][idx].astype(np.int32)
    return y


def tree_depth(pi):
    """Hops from each node to its root along pointers pi [G, N] (roots point to themselves; padding -1 = root)."""
    G, N = pi.shape
    par = np.where(pi < 0, np.arange(N)[None], pi).astype(np.int64)
    idx, dep = np.tile(np.arange(N), (G, 1)), np.zeros((G, N), np.int32)
    for _ in range(N):
        nxt = np.take_along_axis(par, idx, 1)
        dep += nxt != idx
        idx = nxt
    return dep


def load(path):
    return {k: v for k, v in np.load(path).items()}


def main(job):
    task, cfg, data = job['task'], job['model'], Path(job['data_dir']) / job['task']
    outs = OUTS[task]
    ckpt = job.get('params_from')  # score saved parameters (path with {seed}) instead of training
    extra = list(job.get('extra_splits', []))  # further splits in the task's data dir, predicted and scored like test
    parts = [load(data / f'train_L{L}.npz') for L in job.get('sizes', [4, 7, 11, 13, 16])] if not ckpt else [dict(n=np.zeros(1))]
    train = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    splits = {s: load(data / f'{s}.npz') for s in ['val', 'test'] + extra}
    NP = len(train['n'])
    if job.get('depth_mask'):
        assert [k for k, _ in outs] == ['pi'], 'depth_mask is defined for single node-pointer tasks (mst_prim, dfs)'
        train['_depth'] = tree_depth(train['pi'])
    B, steps = job['batch'], job['steps']
    sched = optax.warmup_cosine_decay_schedule(0., job['lr'], job.get('warmup', 1000), steps, job['lr'] * job.get('final_lr_frac', .01))
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=job.get('wd', .01)))
    train_iters, kmax = job['train_iters'], job.get('eval_kmax', 16)

    def make_step(K):
        @jax.jit
        def step(p, s, x, y):
            v, g = jax.value_and_grad(loss_fn)(p, cfg, x, y, K, outs)
            u, s = tx.update(g, s, p)
            return optax.apply_updates(p, u), s, v
        return step
    steps_fn = {K: make_step(K) for K in train_iters}
    ecfg = dict(cfg, remat=False)
    pred_all = jax.jit(lambda p, x: forward(p, ecfg, x, kmax, outs, every=True))

    def predict(p, d):
        """{name: [kmax, samples, ...]} predicted indices after each run, in batches of eval_batch."""
        EB, n = job.get('eval_batch', 32), len(d['n'])
        chunks = [pred_all(p, {k: j.array(v) for k, v in to_x(d, np.arange(i, min(i + EB, n))).items()}) for i in range(0, n, EB)]
        return {name: np.concatenate([np.stack([np.asarray(r[name]) for r in c]) for c in chunks], 1) for name, _ in outs}

    def score(pred, d):
        """Our copy of the library's scorer (pointer: mean of exact index matches; mask_one: argmax match), averaged
        over outputs, per K. The reported numbers come from score.py, which calls clrs.evaluate itself."""
        out = []
        for K in range(kmax):
            sc = []
            for name, kind in outs:
                t = d[name]
                if kind in MASKS:  # clrs evaluation._mask_fn: F1 over entries not marked -1, pooled over the split
                    q, ok = pred[name][K] > .5, t != -1
                    tp, fp, fn = (q & (t > .5) & ok).sum(), (q & (t < .5) & ok).sum(), (~q & (t > .5) & ok).sum()
                    pr = tp / (tp + fp) if tp + fp > 0 else 1.
                    rc = tp / (tp + fn) if tp + fn > 0 else 1.
                    sc.append(float(2 * pr * rc / (pr + rc)) if pr + rc > 0 else 0.)
                elif kind == 'node':
                    sc.append(float(np.mean(pred[name][K][:, :t.shape[1]] == t)))
                elif kind == 'edge':
                    sc.append(float(np.mean(pred[name][K] == t)))
                else:
                    sc.append(float(np.mean(pred[name][K] == t)))
            out.append(float(np.mean(sc)))
        return out
    rows = []
    for seed in job['seeds']:
        rng = np.random.default_rng(seed + 100)
        p = init(jax.random.PRNGKey(seed), cfg, outs)
        s = tx.init(p)
        params = int(sum(w.size for w in jax.tree.leaves(p)))
        t0, curve, it = time.time(), [], 0
        if ckpt:
            W = dict(np.load(ckpt.format(seed=seed)))
            flat, tree = jax.tree_util.tree_flatten_with_path(p)
            p = jax.tree_util.tree_unflatten(tree, [j.array(W[jax.tree_util.keystr(k)]) for k, _ in flat])
        for it in range(1, (0 if ckpt else steps) + 1):
            K = int(train_iters[rng.integers(len(train_iters))])
            idx = rng.integers(NP, size=B)
            x, y = to_x(train, idx, rng), to_y(train, idx, outs)
            p, s, v = steps_fn[K](p, s, {k: j.array(a) for k, a in x.items()}, {k: j.array(a) for k, a in y.items()})
            if it % job.get('check_every', 250) == 0 or it == steps:
                row = dict(seed=seed, step=it, loss=float(v), K=K, sec=round(time.time() - t0, 1))
                if it % job.get('test_every', 2500) == 0 or it == steps:
                    row['val'] = score(predict(p, splits['val']), splits['val'])
                    row['test'] = score(predict(p, splits['test']), splits['test'])
                curve.append(row)
                with (DEST / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                if not np.isfinite(float(v)):
                    break
            if time.time() - t0 > job.get('max_train_seconds', 10 ** 9):  # wall-clock cap: stop training, still score
                break
        train_sec, steps_done = time.time() - t0, it
        flat, _ = jax.tree_util.tree_flatten_with_path(p)
        if not ckpt:
            np.savez_compressed(DEST / f'params_s{seed}.npz', **{jax.tree_util.keystr(k): np.asarray(w) for k, w in flat})
        preds = {sp: predict(p, splits[sp]) for sp in ['val', 'test'] + extra}
        np.savez_compressed(DEST / f'preds_s{seed}.npz', **{f'{sp}/{name}': a.astype(np.int16) for sp, pr in preds.items() for name, a in pr.items()})
        rows.append(dict(task=task, cfg=cfg, seed=seed, params=params, steps=steps, steps_done=steps_done, batch=B, lr=job['lr'], train_iters=train_iters,
                         seconds=train_sec, val=score(preds['val'], splits['val']), test=score(preds['test'], splits['test']), curve=curve,
                         **{sp: score(preds[sp], splits[sp]) for sp in extra}, **({'params_from': ckpt.format(seed=seed)} if ckpt else {})))
        (DEST / 'rows.json').write_text(json.dumps(rows))
    return dict(rows=[{k: v for k, v in r.items() if k != 'curve'} for r in rows])


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
