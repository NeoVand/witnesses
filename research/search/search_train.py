"""Graph search on Saparov et al.'s benchmark (ICLR 2025): pair-state model against Transformers.

Data: their C++ generator (`generator.generate_training_set`, balanced distribution, fresh examples every step) and
their test sets (`l2s_eval.py`, per lookahead). Input: tokens `E u v ... Q s g P s ...`, left-padded.
Models:
- 'pair': a state z_ij in R^D for every pair of token positions. Start: D attention-style scores between tokens i and
  j plus an embedding of the offset j - i. Each layer: z_ij += sigmoid(G t_ij) * sum_k (A t_ik)(B t_kj) / sqrt(n),
  t = norm(z), then a per-pair MLP. Read-out points at a position: p(j) ∝ exp(w . norm(z_last,j)).
- 'sa': pre-norm Transformer, no causal mask (as theirs), width d, H heads, MLP width mlp.
Read-outs: 'pointer' (attend from the last token over positions; the answer is the vertex written there) or 'vocab'
(linear from the last token to the vertex names, as theirs).
python search_train.py '<json>' prints one JSON line; writes rows.json and progress.jsonl to LAB_DEST.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
DEST = Path(os.environ.get('LAB_DEST', '.'))


def norm(x):
    return (x - x.mean(-1, keepdims=True)) * jax.lax.rsqrt(x.var(-1, keepdims=True) + 1e-6)


def mats(key, **shapes):
    keys = iter(jax.random.split(key, len(shapes) + 1))
    return {k: jax.random.normal(next(keys), s) * (s[0] ** -.5) for k, s in shapes.items()}


OFFSETS = 9  # offsets -4..4 get their own embedding; one more for everything farther
NEAR = (-4, -3, -2, -1, 1, 2, 3, 4)  # 'pairn': offsets whose token pairs are pooled into name pairs


def init(key, cfg, vocab, n):
    k = iter(jax.random.split(key, 64))
    e = cfg['emb']
    p = dict(tok=jax.random.normal(next(k), (vocab, e)))
    kpos = next(k)
    if not cfg.get('nopos'):  # S8: models that read graphs of any size have no absolute position embedding
        p['pos'] = jax.random.normal(kpos, (n, e)) * .5
    L = cfg['layers']
    if cfg['kind'] == 'pair':
        D, r = cfg['D'], cfg.get('r', 8)
        p |= dict(U=jax.random.normal(next(k), (e, D, r)) * e ** -.5, V=jax.random.normal(next(k), (e, D, r)) * e ** -.5,
                  rel=jax.random.normal(next(k), (OFFSETS + 1, D)), w=jax.random.normal(next(k), (D,)) * D ** -.5)
        p['blocks'] = [mats(next(k), a=(D, D), b=(D, D), g=(D, D), f1=(D, 2 * D), f2=(2 * D, D)) for _ in range(L)]
    elif cfg['kind'] == 'pairn':
        D, r = cfg['D'], cfg.get('r', 8)
        p |= dict(U=jax.random.normal(next(k), (e, D, r)) * e ** -.5, V=jax.random.normal(next(k), (e, D, r)) * e ** -.5,
                  C=jax.random.normal(next(k), (2, e, e)) * e ** -.5 * .5, w=jax.random.normal(next(k), (D,)) * D ** -.5,
                  A=jax.random.normal(next(k), (len(NEAR), e, D)) * e ** -.5, Bm=jax.random.normal(next(k), (len(NEAR), e, D)) * e ** -.5,
                  c=j.zeros((len(NEAR), D)))
        p['blocks'] = [mats(next(k), a=(D, D), b=(D, D), g=(D, D), f1=(D, 2 * D), f2=(2 * D, D)) | (mats(next(k), v1=(D, D), v2=(D, D)) if cfg.get('tri') == 'attn' else {}) for _ in range(1 if cfg.get('loop') else L)]  # S8: one block, looped
        if cfg.get('tri') in ('newton', 'pmean'):  # S20: soft OR (Newton power sums) or power mean over the middle vertex
            ka = jax.random.fold_in(key, 20)  # own key, so the shared iterator (and every earlier config's init) is untouched
            for i, b in enumerate(p['blocks']):
                b['o'] = mats(jax.random.fold_in(ka, i), o=(D, D))['o']
                if cfg['tri'] == 'newton':
                    b['om'] = j.ones((cfg.get('J', 8), D))  # omega_j per channel, 1 = the series of -log prod_k (1 - a c)
                else:
                    b['beta'] = j.full((D,), 4.0)
        if cfg.get('tri') == 'prop':  # S22: the propagator join, X_h = S_h (I - G_h)^{-1}; replaces the join's a and b
            kp = jax.random.fold_in(key, 22)  # own key, so the shared iterator (and every earlier config's init) is untouched
            Hp, t0 = cfg.get('pheads', 4), cfg.get('t0', .9)
            for i, b in enumerate(p['blocks']):
                del b['a'], b['b']
                b |= mats(jax.random.fold_in(kp, i), pg=(D, Hp), ps=(D, Hp), po=(2 * Hp, D))
                b['pgb'] = j.full((Hp,), cfg.get('gate_bias', -6.))  # gates start near 0, so rho(G) < 1 at initialisation
                b['psb'] = j.full((Hp,), cfg.get('src_bias', -2.))
                if cfg.get('learn_t0', True):  # t0 = sigmoid(pt) < 1, one per channel ('t0s' spreads the starting values)
                    ts = np.array(cfg.get('t0s', [t0] * Hp), np.float64)
                    b['pt'] = j.array(np.log(ts / (1 - ts)), j.float32)
        if cfg.get('ssmax'):  # S16: scalable softmax over the middle vertex, scores times s_h log n (Nakanishi 2025)
            for b in p['blocks']:
                b['ss'] = j.full((cfg.get('heads', 4),), 1.0 / math.log(49.0))
    elif cfg['kind'] == 'sann':
        D, r, d, H = cfg['D'], cfg.get('r', 8), cfg['d'], cfg['H']
        p |= dict(U=jax.random.normal(next(k), (e, D, r)) * e ** -.5, V=jax.random.normal(next(k), (e, D, r)) * e ** -.5,
                  C=jax.random.normal(next(k), (2, e, e)) * e ** -.5 * .5,
                  A=jax.random.normal(next(k), (len(NEAR), e, D)) * e ** -.5, Bm=jax.random.normal(next(k), (len(NEAR), e, D)) * e ** -.5,
                  c=j.zeros((len(NEAR), D)), inp=jax.random.normal(next(k), (e, d)) * e ** -.5)
        p['blocks'] = [mats(next(k), qkv=(d, 3 * d), o=(d, d), f1=(d, cfg['mlp']), f2=(cfg['mlp'], d), bias=(D, H)) for _ in range(1 if cfg.get('loop') else L)]  # S17: one block, looped
        p |= mats(next(k), pq=(d, d), pk=(d, d))
        if cfg.get('pair_read'):  # S23: the pair state's read-out input, w_r . zn[cur, v]; zeros, no random draw
            p['wr'] = j.zeros((D,))
    else:
        d, m = cfg['d'], cfg['mlp']
        p['inp'] = mats(next(k), w=(e, d))['w']
        p['blocks'] = [mats(next(k), qkv=(d, 3 * d), o=(d, d), f1=(d, m), f2=(m, d)) for _ in range(L)]
        p |= mats(next(k), pq=(d, d), pk=(e, d), voc=(d, vocab))
    return p


def offset_index(n):
    off = j.arange(n)[None, :] - j.arange(n)[:, None]
    return j.where(j.abs(off) <= 4, off + 4, OFFSETS)


def newton_series(a, c, om, cd=j.float32):
    """S20: soft OR over the middle vertex, its exponent. a [B, D, N, K], c [B, D, K, N] in [0, 1), om [J, D].
    L = sum_j om_j P_j / j with P_j = a^j @ c^j (Hadamard powers, one matmul each; no N x N x N array), which for om = 1
    is -log prod_k (1 - a_uk c_kv) truncated at J (Newton's identities, generating-function form); returns L.
    Each order is rematerialised, so the backward pass keeps a and c, not J copies of their powers."""
    def term(a, c, q):
        return j.matmul((a ** q).astype(cd), (c ** q).astype(cd), preferred_element_type=j.float32)
    term = jax.checkpoint(term, static_argnums=(2,))
    return sum(om[q - 1][None, :, None, None] * term(a, c, q) / q for q in range(1, om.shape[0] + 1))


def newton_or(a, c, om, cd=j.float32):
    """S20: the soft OR 1 - exp(-L) of newton_series."""
    L = newton_series(a, c, om, cd)
    return -j.expm1(-j.maximum(L, -20.))  # the floor only guards exp against overflow if learned omegas turn negative


def power_mean(la, lc, mk, mr, beta, cd=j.float32):
    """S20: power mean over the middle vertex, M = (sum_k (a_uk c_kv)^beta)^(1/beta), beta per channel (one matmul).
    la, lc are log a and log c; mk, mr mask absent partners k (in a's last axis and c's first)."""
    bt = j.maximum(beta, .25)[None, :, None, None]
    S = j.matmul((j.exp(bt * la) * mk).astype(cd), (j.exp(bt * lc) * mr).astype(cd), preferred_element_type=j.float32)
    return j.exp(j.log(S + 1e-30) / bt)


PROP_EPS = 1e-20  # S22: floor inside log(X + eps); present pairs have X >= S_uv > 0, so it binds only on absent names


def propagator(t, b, m, cfg):
    """S22: the propagator join on a channel-first pair state t [B, D, N, N] (normalised), m [B, N] marks real names.
    Per channel h: gates G_uv = t0_h sigma(w_h . t_uv + b_h) and source S_uk = sigma(a_h . t_uk + c_h), both zero on
    rows and columns of absent names; X = S (I - G)^{-1} = S + S G + S G^2 + ..., every walk of every length weighted by
    the product of its gates, by one batched linear solve ((I - G)^T X^T = S^T; no walk is formed). With cfg['psrc']
    False, X = R = (I - G)^{-1}. Float32 throughout. Returns X [B, H, N, N], clipped at 0 (a guard: negative entries
    arise only if the spectral radius of G reaches 1)."""
    N = t.shape[-1]
    mm = m[:, None, :, None] * m[:, None, None, :]
    proj = lambda w, c: j.einsum('bcij,ch->bhij', t, w) + c[None, :, None, None]
    t0 = jax.nn.sigmoid(b['pt'])[None, :, None, None] if 'pt' in b else cfg.get('t0', .9)
    G = t0 * jax.nn.sigmoid(proj(b['pg'], b['pgb'])) * mm
    if cfg.get('rowcap'):  # S22c: rows scaled so sum_v G_uv <= t0; by Gershgorin rho(G) <= t0 < 1 on any graph
        G = G * (t0 / j.maximum(G.sum(-1, keepdims=True), t0))  # = min(1, t0 / row sum); no division by an empty row's 0
    I = j.eye(N, dtype=G.dtype)
    if cfg.get('psrc', True):
        S = jax.nn.sigmoid(proj(b['ps'], b['psb'])) * mm
        X = j.swapaxes(j.linalg.solve(j.swapaxes(I - G, -1, -2), j.swapaxes(S, -1, -2)), -1, -2)
    else:
        X = j.linalg.solve(I - G, j.broadcast_to(I, G.shape))
    return j.maximum(X, 0.)


def prop_features(X):
    """S22: log(X + eps) (at low temperature, minus a distance) and 1 - exp(-X) (a walk exists), channels stacked."""
    return j.concatenate([j.log(X + PROP_EPS), -j.expm1(-X)], 1)


def pair_stack(p, cfg, z, m, nv):
    """The triangle layers on a channel-first pair state z [B, D, N, N]; m [B, N] marks real partners k."""
    cd = j.bfloat16 if cfg.get('bf16', True) else j.float32
    lin = lambda t, w: j.einsum('bcij,cd->bdij', t.astype(cd), w.astype(cd), preferred_element_type=j.float32)
    cnorm = lambda t: (t - t.mean(1, keepdims=True)) * jax.lax.rsqrt(t.var(1, keepdims=True) + 1e-6)

    H = cfg.get('heads', 4)

    def block(z, b):
        t = cnorm(z)
        if cfg.get('tri') == 'attn':
            # Triangle attention (Edge Transformer form): pair (u, v) attends over middle elements k with score
            # q(z_uk) . k(z_kv), softmax over k, and reads v1(z_uk) * v2(z_kv): a join of two pair entries.
            Bb, D, N, _ = t.shape
            dh = D // H
            q = lin(t, b['a']).reshape(Bb, H, dh, N, N)
            k_ = lin(t, b['b']).reshape(Bb, H, dh, N, N)
            sc = j.einsum('bhcuk,bhckv->bhkuv', q.astype(cd), k_.astype(cd), preferred_element_type=j.float32) / dh ** .5
            if 'ss' in b:  # S16: keep the softmax over k as sharp on larger graphs as on the training sizes
                sc = sc * (b['ss'][None, :, None, None, None] * j.log(nv[:, :, :, :, None]))
            sc = j.where(m[:, None, :, None, None] > 0, sc, -1e9)
            att = jax.nn.softmax(sc, axis=2)
            v1 = lin(t, b['v1']).reshape(Bb, H, dh, N, N)
            v2 = lin(t, b['v2']).reshape(Bb, H, dh, N, N)
            tri = j.einsum('bhkuv,bhcuk,bhckv->bhcuv', att, v1, v2).reshape(Bb, D, N, N)
            z = z + jax.nn.sigmoid(lin(t, b['g'])) * tri
        elif cfg.get('tri') in ('newton', 'pmean'):
            # S20. a_uk = sigma(A t_uk), c_kv = sigma(B t_kv); absent partners k give a = c = 0. 'newton': the soft OR
            # O = 1 - exp(-sum_j omega_j P_j / j), P_j = sum_k a_uk^j c_kv^j (one matmul per order j); 'pmean': the
            # power mean M = (sum_k (a_uk c_kv)^beta)^(1/beta) (one matmul). Then z += sigma(G t) * W_o O. No 1/sqrt(n).
            xa, xc = lin(t, b['a']), lin(t, b['b'])
            mk, mr = m[:, None, None, :], m[:, None, :, None]
            if cfg['tri'] == 'newton':
                agg = newton_or(jax.nn.sigmoid(xa) * mk, jax.nn.sigmoid(xc) * mr, b['om'], cd)
            else:
                agg = power_mean(jax.nn.log_sigmoid(xa), jax.nn.log_sigmoid(xc), mk, mr, b['beta'], cd)
            z = z + jax.nn.sigmoid(lin(t, b['g'])) * lin(agg, b['o'])
        elif cfg.get('tri') == 'prop':
            # S22: z += sigma(G t) * W_o [log(X + eps), 1 - exp(-X)], X = S (I - G)^{-1} (propagator), then the MLP.
            f = prop_features(propagator(t, b, m, cfg))
            z = z + jax.nn.sigmoid(lin(t, b['g'])) * j.einsum('bcij,cd->bdij', f, b['po'])
        else:
            a, c = lin(t, b['a']) * m[:, None, None, :], lin(t, b['b']) * m[:, None, :, None]
            tri = j.matmul(a.astype(cd), c.astype(cd), preferred_element_type=j.float32)  # sum_k a_ik c_kj
            z = z + jax.nn.sigmoid(lin(t, b['g'])) * tri / j.sqrt(nv)
        return z + lin(jax.nn.gelu(lin(cnorm(z), b['f1'])), b['f2'])
    blocks = [p['blocks'][0]] * cfg['iters'] if cfg.get('loop') else p['blocks']  # S8: the same block, run iters times
    for b in blocks:
        z = jax.checkpoint(block)(z, b) if cfg.get('remat', True) else block(z, b)
    return z


def pairn_start(p, x, pad, h0, pool_norm=False):
    """Pair state over vertex names, not positions. Tokens are pooled into names by identity (a parameter-free
    bag, like an embedding lookup run backwards); the graph itself is never given. Returns the starting state."""
    B, n = x.shape
    Nn = pad - 2  # names are token ids 0..V; markers E, Q, P and padding are not names
    hn = norm(h0)
    sh = lambda t, o: j.roll(t, -o, axis=1) * ((j.arange(n) + o >= 0) & (j.arange(n) + o < n))[None, :, None]
    ctx = norm(hn + sh(hn, -1) @ p['C'][0] + sh(hn, 1) @ p['C'][1])  # each token with its two neighbours
    OH = jax.nn.one_hot(x, Nn, dtype=hn.dtype)  # [B, n, Nn]
    cnt = OH.sum(1)  # [B, Nn]
    if pool_norm:  # S16: a name's pooled features keep one scale however often it is written (degree-invariant)
        g = norm(j.einsum('bin,bie->bne', OH, ctx) / j.maximum(cnt, 1.)[..., None])
    else:
        g = j.einsum('bin,bie->bne', OH, ctx) * jax.lax.rsqrt(j.maximum(cnt, 1.))[..., None]
    r = p['U'].shape[-1]
    u, w = j.einsum('bne,edr->bdnr', g, p['U']), j.einsum('bne,edr->bdnr', g, p['V'])
    z = j.einsum('bdur,bdvr->bduv', u, w) / r ** .5
    for oi, o in enumerate(NEAR):  # nearby token pairs, pooled by the names written there
        M = (ctx @ p['A'][oi]) * (sh(ctx, o) @ p['Bm'][oi]) + p['c'][oi]
        z = z + j.einsum('biu,bid,biv->bduv', OH, M, sh(OH, o))
    present = (cnt > 0).astype(z.dtype)
    nv = present.sum(1)[:, None, None, None]
    return z, present, nv, Nn


def sann_start(p, x, pad, h0):
    """S17: the Transformer over names (S3 'sann') split into start / blocks / read-out, so its one block can be looped.
    Same pooled names and starting pair features as pairn_start; the pair features enter only as attention biases."""
    B, n = x.shape
    Nn = pad - 2
    hn = norm(h0)
    sh = lambda t, o: j.roll(t, -o, axis=1) * ((j.arange(n) + o >= 0) & (j.arange(n) + o < n))[None, :, None]
    ctx = norm(hn + sh(hn, -1) @ p['C'][0] + sh(hn, 1) @ p['C'][1])
    OH = jax.nn.one_hot(x, Nn, dtype=hn.dtype)
    cnt = OH.sum(1)
    g = j.einsum('bin,bie->bne', OH, ctx) * jax.lax.rsqrt(j.maximum(cnt, 1.))[..., None]
    r = p['U'].shape[-1]
    u, w = j.einsum('bne,edr->bdnr', g, p['U']), j.einsum('bne,edr->bdnr', g, p['V'])
    z = j.einsum('bdur,bdvr->bduv', u, w) / r ** .5
    for oi, o in enumerate(NEAR):
        M = (ctx @ p['A'][oi]) * (sh(ctx, o) @ p['Bm'][oi]) + p['c'][oi]
        z = z + j.einsum('biu,bid,biv->bduv', OH, M, sh(OH, o))
    zn = (z - z.mean(1, keepdims=True)) * jax.lax.rsqrt(z.var(1, keepdims=True) + 1e-6)
    return g @ p['inp'], zn, cnt > 0, Nn


def sann_stack(p, cfg, h, zn, present):
    B, Nn, d = h.shape
    H = cfg['H']
    sp = lambda t: t.reshape(B, Nn, H, d // H)
    blocks = [p['blocks'][0]] * cfg['iters'] if cfg.get('loop') else p['blocks']
    for b in blocks:
        q, k, v = (sp(t) for t in j.split(norm(h) @ b['qkv'], 3, -1))
        sc = j.einsum('buhc,bvhc->bhuv', q, k) / (d // H) ** .5 + (0. if cfg.get('nobias') else j.einsum('bcuv,ch->bhuv', zn, b['bias']))
        a = jax.nn.softmax(j.where(present[:, None, None], sc, -1e9), -1)
        h = h + j.einsum('bhuv,bvhc->buhc', a, v).reshape(B, Nn, d) @ b['o']
        h = h + jax.nn.relu(norm(h) @ b['f1']) @ b['f2']
    return h


def sann_read(p, h, x, present, Nn, pad, zn=None):
    B, _, d = h.shape
    cur = jax.nn.one_hot(x[:, -1], Nn, dtype=h.dtype)
    hc = j.einsum('bud,bu->bd', norm(h), cur)
    lg = j.einsum('bd,bvd->bv', hc @ p['pq'], norm(h) @ p['pk']) / d ** .5
    if 'wr' in p:  # S23: plus the starting pair features of (cur, v), as the pair state's read-out sees them
        lg = lg + j.einsum('bcuv,bu,c->bv', zn, cur, p['wr'])
    logits = j.where(present, lg, -1e9)
    return j.concatenate([logits, j.full((B, pad + 2 - Nn), -1e9)], -1)


def pairn_read(p, z, x, present, Nn, pad):
    """Vocabulary logits from the current vertex's row of the pair state."""
    B = x.shape[0]
    cur = jax.nn.one_hot(x[:, -1], Nn, dtype=z.dtype)
    row = j.einsum('bduv,bu->bvd', z, cur)
    logits = j.where(present > 0, norm(row) @ p['w'], -1e9)
    return j.concatenate([logits, j.full((B, pad + 2 - Nn), -1e9)], -1)


def forward(p, cfg, x, pad):
    """x [B, n] tokens -> (position logits [B, n] or None, vocab logits [B, vocab] or None)."""
    B, n = x.shape
    valid = x != pad
    h0 = p['tok'][x] + (0. if cfg.get('nopos') else p['pos'][None, :n])
    neg = -1e9
    if cfg['kind'] == 'sann':
        # Control: the same pooled names and the same starting pair features, but a token state per name with
        # attention between names; the pair features enter only as per-head attention biases (no pair state).
        Nn = pad - 2
        hn = norm(h0)
        sh = lambda t, o: j.roll(t, -o, axis=1) * ((j.arange(n) + o >= 0) & (j.arange(n) + o < n))[None, :, None]
        ctx = norm(hn + sh(hn, -1) @ p['C'][0] + sh(hn, 1) @ p['C'][1])
        OH = jax.nn.one_hot(x, Nn, dtype=hn.dtype)
        cnt = OH.sum(1)
        g = j.einsum('bin,bie->bne', OH, ctx) * jax.lax.rsqrt(j.maximum(cnt, 1.))[..., None]
        r = p['U'].shape[-1]
        u, w = j.einsum('bne,edr->bdnr', g, p['U']), j.einsum('bne,edr->bdnr', g, p['V'])
        z = j.einsum('bdur,bdvr->bduv', u, w) / r ** .5
        for oi, o in enumerate(NEAR):
            M = (ctx @ p['A'][oi]) * (sh(ctx, o) @ p['Bm'][oi]) + p['c'][oi]
            z = z + j.einsum('biu,bid,biv->bduv', OH, M, sh(OH, o))
        present = cnt > 0
        h = g @ p['inp']
        H = cfg['H']
        d = h.shape[-1]
        sp = lambda t: t.reshape(B, Nn, H, d // H)
        zn = (z - z.mean(1, keepdims=True)) * jax.lax.rsqrt(z.var(1, keepdims=True) + 1e-6)
        for b in p['blocks']:
            q, k, v = (sp(t) for t in j.split(norm(h) @ b['qkv'], 3, -1))
            sc = j.einsum('buhc,bvhc->bhuv', q, k) / (d // H) ** .5 + (0. if cfg.get('nobias') else j.einsum('bcuv,ch->bhuv', zn, b['bias']))
            a = jax.nn.softmax(j.where(present[:, None, None], sc, neg), -1)
            h = h + j.einsum('bhuv,bvhc->buhc', a, v).reshape(B, Nn, d) @ b['o']
            h = h + jax.nn.relu(norm(h) @ b['f1']) @ b['f2']
        cur = jax.nn.one_hot(x[:, -1], Nn, dtype=h.dtype)
        hc = j.einsum('bud,bu->bd', norm(h), cur)
        logits = j.einsum('bd,bvd->bv', hc @ p['pq'], norm(h) @ p['pk']) / d ** .5
        if 'wr' in p:  # S23: plus w_r . zn[cur, v], the starting pair features the pair state's read-out also sees
            logits = logits + j.einsum('bcuv,bu,c->bv', zn, cur, p['wr'])
        logits = j.where(present, logits, neg)
        return None, j.concatenate([logits, j.full((B, pad + 2 - Nn), neg)], -1)
    if cfg['kind'] == 'pairn':
        z, present, nv, Nn = pairn_start(p, x, pad, h0, cfg.get('pool_norm', False))
        z = pair_stack(p, cfg, z, present, nv)
        return None, pairn_read(p, z, x, present, Nn, pad)
    if cfg['kind'] == 'pair':
        r = p['U'].shape[-1]
        hn = norm(h0)
        u, w = j.einsum('bie,edr->bdir', hn, p['U']), j.einsum('bie,edr->bdir', hn, p['V'])
        # Channel-first pair state z [B, D, n, n], so the triangle step is one batched n x n matmul per channel.
        z = j.einsum('bdir,bdjr->bdij', u, w) / r ** .5  # D attention-style scores per pair
        if not cfg.get('norel'):  # S6 ablation: no learned embedding of the token offset j - i
            z = z + p['rel'][offset_index(n)].transpose(2, 0, 1)[None]
        m = valid.astype(z.dtype)  # partners k must be real tokens
        nv = valid.sum(1)[:, None, None, None].astype(z.dtype)
        z = pair_stack(p, cfg, z, m, nv)
        z = z.transpose(0, 2, 3, 1)  # back to [B, n, n, D] for the read-out
        pos = norm(z[:, -1]) @ p['w']
        return j.where(valid, pos, neg), None
    h = h0 @ p['inp']
    H = cfg['H']
    d = h.shape[-1]
    sp = lambda t: t.reshape(B, n, H, d // H)
    for b in p['blocks']:
        q, k, v = (sp(t) for t in j.split(norm(h) @ b['qkv'], 3, -1))
        s = j.einsum('bihc,bjhc->bhij', q, k) / (d // H) ** .5
        a = jax.nn.softmax(j.where(valid[:, None, None], s, neg), -1)
        h = h + j.einsum('bhij,bjhc->bihc', a, v).reshape(B, n, d) @ b['o']
        h = h + jax.nn.relu(norm(h) @ b['f1']) @ b['f2']
    last = norm(h[:, -1])
    if cfg['readout'] == 'vocab':
        return None, last @ p['voc']
    pos = j.einsum('bd,bjd->bj', last @ p['pq'], norm(h0) @ p['pk']) / d ** .5
    return j.where(valid, pos, neg), None


def answer_logits(p, cfg, x, pad, vocab):
    """Vocabulary log-probabilities either way: a pointer's mass on position j goes to the token written there."""
    pos, voc = forward(p, cfg, x, pad)
    if voc is not None:
        return jax.nn.log_softmax(voc, -1)
    lp = jax.nn.log_softmax(pos, -1)
    onehot = jax.nn.one_hot(x, vocab, dtype=bool)
    return jax.nn.logsumexp(j.where(onehot, lp[..., None], -1e9), axis=1)


def flops_per_example(cfg, n):
    if cfg['kind'] == 'sann':
        d, m, L, N = cfg['d'], cfg['mlp'], cfg['layers'], (n - 5) // 3 + 1
        return 2 * (L * (N * (4 * d * d + 2 * d * m) + 2 * N * N * d) + len(NEAR) * n * N * N * cfg['D'])
    if cfg['kind'] == 'pairn':
        D, L, N = cfg['D'], cfg['layers'], (n - 5) // 3 + 1
        if cfg.get('tri') == 'prop':  # S22: per channel an LU (2N^3/3) and N right-hand sides (2N^3), as multiply-adds
            Hp = cfg.get('pheads', 4)
            return 2 * (L * (Hp * N ** 3 * 4 // 3 + N * N * (5 * D * D + 4 * D * Hp)) + len(NEAR) * n * N * N * D)
        return 2 * (L * (N ** 3 * D * 2 + N * N * 7 * D * D) + len(NEAR) * n * N * N * D)
    if cfg['kind'] == 'pair':
        D, r, L = cfg['D'], cfg.get('r', 8), cfg['layers']
        per = n ** 3 * D * 2 + n * n * (3 * D * D + 4 * D * D)
        return 2 * (L * per + n * n * D * r * 2)
    d, m, L = cfg['d'], cfg['mlp'], cfg['layers']
    return 2 * L * (n * (4 * d * d + 2 * d * m) + 2 * n * n * d)


def main(job):
    sys.path.insert(0, job.get('generator_dir', '/root/l2s'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generator, l2s_eval
    V = job['vertices']
    n = 3 * V + 5
    pad, vocab = V + 3, V + 5
    max_look = (V - 1) // 2
    ns = l2s_eval.load(job.get('their_train_py', '/root/l2s/train.py'))
    dfs = 1 if job.get('first_step') else -1  # their padded variant asks only the first step from the start
    tests = l2s_eval.test_sets(ns, n, job.get('per_lookahead', 1000), rng_seed=job.get('test_seed', 12345), distance_from_start=dfs)
    reserved = {tuple(int(t) for t in row if t != pad) for L, (xs, _) in tests.items() for row in xs}
    cfg = job['model']
    B = job['batch']
    sched = optax.join_schedules([optax.linear_schedule(0., job['lr'], job.get('warmup', 2000)), optax.constant_schedule(job['lr'])], [job.get('warmup', 2000)])
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=job.get('wd', .01)))
    rows = []
    for seed in job['seeds']:
        generator.set_seed(seed + 1000)
        mix_rng = np.random.default_rng(seed + 5000)
        p = init(jax.random.PRNGKey(seed), cfg, vocab, n)
        st = tx.init(p)
        params = int(sum(w.size for w in jax.tree.leaves(p)))

        @jax.jit
        def step(p, st, x, lab):
            def loss(p):
                lp = answer_logits(p, cfg, x, pad, vocab)
                return -j.take_along_axis(lp, lab[:, None], 1).mean(), lp
            (v, lp), g = jax.value_and_grad(loss, has_aux=True)(p)
            u, st = tx.update(g, st, p)
            return optax.apply_updates(p, u), st, v, lp.argmax(-1), optax.global_norm(g)
        said_fn = jax.jit(lambda p, x: answer_logits(p, cfg, x, pad, vocab).argmax(-1))

        def test(p):
            out = {}
            for L, (xs, ys) in tests.items():
                said = np.concatenate([np.asarray(said_fn(p, j.array(xs[i:i + B]))) for i in range(0, len(xs), B)])
                out[str(L)] = float((said == ys).mean())
            return out
        t0, seen, curve, hits, done_at = time.time(), 0, [], 0, None
        while seen < job['max_examples']:
            look = -1 if mix_rng.random() < job.get('mix_random', 0.) else max_look  # -1: their random-graph distribution
            x, y, lab, _ = generator.generate_training_set(n, B, look, V, reserved, dfs, n, True)
            p, st, v, said, gn = step(p, st, j.array(x), j.array(lab.astype(np.int32)))
            seen += B
            if (seen // B) % job.get('check_every', 500) == 0:
                acc = float(np.mean(np.asarray(y)[np.arange(B), np.asarray(said)] > 0))
                row = dict(seed=seed, examples=seen, loss=float(v), train_acc=acc, sec=round(time.time() - t0), gnorm=float(gn))
                if job.get('abort_nonfinite') and not (np.isfinite(row['loss']) and np.isfinite(row['gnorm'])):  # S23 (as S22c)
                    with (DEST / 'progress.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                    raise RuntimeError(f'non-finite loss or gradient norm at {seen} examples: {row}')
                if (seen // B) % job.get('test_every', 5000) == 0:
                    row['test'] = test(p)
                curve.append(row)
                with (DEST / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                hits = hits + 1 if acc >= .995 else 0
                if hits >= 2 and done_at is None:
                    done_at = seen
                    if job.get('stop_on_converge', True):
                        break
        if job.get('assert_smoke'):  # S23 (as S22c): finite losses and gradient norms at every check, and the loss falls
            ls, gs = [r['loss'] for r in curve], [r['gnorm'] for r in curve]
            assert len(ls) >= 6 and all(np.isfinite(ls)) and all(np.isfinite(gs)), (ls, gs)
            assert np.mean(ls[-3:]) < np.mean(ls[:3]) - .1, ('loss did not fall', ls)
        final = test(p)
        mean_look = float(np.mean([final[str(L)] for L in range(1, max_look + 1)]))
        rows.append(dict(cfg=cfg, seed=seed, vertices=V, tokens=n, params=params, flops=flops_per_example(cfg, n), lr=job['lr'], batch=B,
                         examples=seen, converged_at=done_at, seconds=time.time() - t0, test=final, test_mean_lookahead=mean_look, curve=curve))
        (DEST / 'rows.json').write_text(json.dumps(rows))
        if job.get('save'):
            flat, _ = jax.tree_util.tree_flatten_with_path(p)
            np.savez_compressed(DEST / f'params_s{seed}.npz', **{jax.tree_util.keystr(k): np.asarray(w) for k, w in flat})
    return dict(rows=[{k: v for k, v in r.items() if k != 'curve'} for r in rows])


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
