# Attention layers and synthetic tasks: pairwise attention, tree and Strassen attention, cubic (2-simplicial) attention,
# Newton attention and related forms. SET uses attention()'s Newton path (bench2.py, bench4.py, bench_nset.py).
import argparse, itertools, json, math, pathlib, sys, time, os, pickle
import numpy as np
import jax, jax.numpy as jnp
import optax

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import groups  # noqa: E402

# ----------------------------------------------------------------------------- tasks
# Each generator returns fields [B,T,F] (int), targets [B,T] (int), loss mask [B,T] (float).
# Position 0 is always a BOS/sink token.

def task_match3(rng, B, n=16, M=128):
    x = rng.integers(0, M, size=(B, n))
    s = (x[:, :, None, None] + x[:, None, :, None] + x[:, None, None, :]) % M == 0
    i = np.arange(n)
    distinct = (i[:, None, None] != i[None, :, None]) & (i[:, None, None] != i[None, None, :]) & (i[None, :, None] != i[None, None, :])
    y = (s & distinct[None]).any(axis=(2, 3)).astype(np.int32)
    tok = np.concatenate([np.full((B, 1), M), x], 1)[..., None]
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def task_match3_homa(rng, B, n=6, M=30):
    # Match3 on HOMA's protocol (Amiraslani & Gao 2026, Sec. 3.1 and App. D.1, after Sanford et al.):
    # x uniform on Z_M, y_i = 1[exists j, z: x_i + x_j + x_z = 0 mod M], where j and z range over the whole
    # sequence and may equal each other or i. No BOS token; every position is scored.
    x = rng.integers(0, M, size=(B, n))
    y = ((x[:, :, None, None] + x[:, None, :, None] + x[:, None, None, :]) % M == 0).any(axis=(2, 3))
    return x[..., None], y.astype(np.int32), np.ones((B, n))

def task_set(rng, B, n=12):
    # SET cards = Z_3^4. Three distinct cards form a SET iff they sum to 0 coordinate-wise.
    ids = np.argsort(rng.random((B, 81)), axis=1)[:, :n]
    feats = np.stack([(ids // 3 ** a) % 3 for a in range(4)], -1)  # [B,n,4]
    s = ((feats[:, :, None, None] + feats[:, None, :, None] + feats[:, None, None, :]) % 3 == 0).all(-1)
    i = np.arange(n)
    distinct = (i[:, None, None] != i[None, :, None]) & (i[:, None, None] != i[None, None, :]) & (i[None, :, None] != i[None, None, :])
    y = (s & distinct[None]).any(axis=(2, 3)).astype(np.int32)
    tok = np.concatenate([np.full((B, 1, 4), 3), feats], 1)  # attribute value 3 = BOS
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def task_compose(rng, B, n=8, hops=2):
    # Function tables f_1..f_h as (slot, value) tokens, then the query x; predict f_h(...f_1(x)).
    fs = rng.integers(0, n, size=(hops, B, n))
    x = rng.integers(0, n, size=B)
    ans = x.copy()
    for h in range(hops):
        ans = fs[h, np.arange(B), ans]
    slots = np.concatenate([np.arange(n) + h * n for h in range(hops)])
    slot_tok = np.concatenate([np.full((B, 1), hops * n + 1), np.broadcast_to(slots, (B, hops * n)), np.full((B, 1), hops * n)], 1)
    val_tok = np.concatenate([np.full((B, 1), n), np.concatenate(list(fs), 1), x[:, None]], 1)
    tok = np.stack([slot_tok, val_tok], -1)
    T = tok.shape[1]
    tgt = np.zeros((B, T), np.int32); tgt[:, -1] = ans
    msk = np.zeros((B, T)); msk[:, -1] = 1
    return tok, tgt, msk

def task_triangle(rng, B, V=8, m=10):
    # Tokens are edges of a random graph; label = edge lies on a triangle.
    pairs = np.array([(u, v) for u in range(V) for v in range(u + 1, V)])
    sel = np.argsort(rng.random((B, len(pairs))), axis=1)[:, :m]
    e = pairs[sel]  # [B,m,2]
    adj = np.zeros((B, V, V), bool)
    bi = np.repeat(np.arange(B), m)
    adj[bi, e[..., 0].ravel(), e[..., 1].ravel()] = True
    adj = adj | adj.transpose(0, 2, 1)
    common = (adj[np.arange(B)[:, None], e[..., 0]] & adj[np.arange(B)[:, None], e[..., 1]]).any(-1)
    y = common.astype(np.int32)
    tok = np.concatenate([np.full((B, 1, 2), V), e], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, m))], 1)
    return tok, tgt, msk

def _pairs(rng, B, n, M, P):
    key = rng.integers(0, M, size=(B, n)); pay = rng.integers(0, P, size=(B, n))
    i = np.arange(n)
    distinct = (i[:, None, None] != i[None, :, None]) & (i[:, None, None] != i[None, None, :]) & (i[None, :, None] != i[None, None, :])
    match = ((key[:, :, None, None] + key[:, None, :, None] + key[:, None, None, :]) % M == 0) & distinct[None]  # [B,i,j,k]
    return key, pay, match

def _wrap(key, pay, M, P):
    B = key.shape[0]
    return np.stack([np.concatenate([np.full((B, 1), M), key], 1), np.concatenate([np.full((B, 1), P), pay], 1)], -1)

def task_pairbind(rng, B, n=16, M=64, P=4):
    # Label at i: some matched pair (x_i+x_j+x_k = 0 mod M) has EQUAL payloads. Needs binding of j with k.
    key, pay, match = _pairs(rng, B, n, M, P)
    eq = pay[:, None, :, None] == pay[:, None, None, :]
    y = (match & eq).any(axis=(2, 3)).astype(np.int32)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return _wrap(key, pay, M, P), tgt, msk

def task_pairjoint(rng, B, n=16, M=96, P=5):
    # Multi-label at i: which joint payload combos (p_j, p_k) occur among matched pairs.
    key, pay, match = _pairs(rng, B, n, M, P)
    joint = pay[:, :, None] * P + pay[:, None, :]  # [B,j,k]
    oh = np.eye(P * P)[joint]  # [B,j,k,P*P]
    y = (np.einsum("bijk,bjkc->bic", match.astype(np.float32), oh) > 0).astype(np.float32)
    tgt = np.concatenate([np.zeros((B, 1, P * P), np.float32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return _wrap(key, pay, M, P), tgt, msk

def task_setbind(rng, B, n=12, P=2):
    # Query card (last slot) lies in exactly two SETs. Target = how many of those SETs have partners with
    # equal payloads (0/1/2). Additive pair values see only the payload histogram, which is ambiguous.
    cards = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
    toks = np.zeros((B, n + 1, 5), np.int64); tgt = np.zeros((B, n + 1), np.int32); msk = np.zeros((B, n + 1))
    toks[:, 0] = [3, 3, 3, 3, P]
    for b in range(B):
        while True:
            q = rng.integers(81)
            j1 = rng.integers(81)
            if j1 == q: continue
            k1 = int(((-(cards[q] + cards[j1])) % 3) @ (3 ** np.arange(4)))
            j2 = rng.integers(81)
            if j2 in (q, j1, k1): continue
            k2 = int(((-(cards[q] + cards[j2])) % 3) @ (3 ** np.arange(4)))
            if k2 in (q, j1, k1): continue
            chosen = [j1, k1, j2, k2]
            pool = [c for c in rng.permutation(81) if c not in chosen and c != q]
            ok = True
            for c in pool:
                if len(chosen) == n - 1: break
                comp = int(((-(cards[q] + cards[c])) % 3) @ (3 ** np.arange(4)))
                if comp in chosen: continue  # would create a third SET through q
                chosen.append(c)
            if len(chosen) == n - 1: break
        perm = rng.permutation(n - 1); chosen = [chosen[i] for i in perm]
        pay = rng.integers(0, P, size=n)
        pos = {c: i for i, c in enumerate(chosen)}
        y = int(pay[pos[j1]] == pay[pos[k1]]) + int(pay[pos[j2]] == pay[pos[k2]])
        ids = chosen + [q]
        toks[b, 1:, :4] = cards[ids]; toks[b, 1:, 4] = pay
        tgt[b, n] = y; msk[b, n] = 1
    return toks, tgt, msk

def task_set4(rng, B, n=8):
    # Card i belongs to 4 distinct cards summing to 0 in Z_3^4 (arity-4 constraint, no chain structure).
    ids = np.argsort(rng.random((B, 81)), axis=1)[:, :n]
    f = np.stack([(ids // 3 ** a) % 3 for a in range(4)], -1)
    s = ((f[:, :, None, None, None] + f[:, None, :, None, None] + f[:, None, None, :, None] + f[:, None, None, None, :]) % 3 == 0).all(-1)
    i = np.arange(n)
    I, J, K, L = i[:, None, None, None], i[None, :, None, None], i[None, None, :, None], i[None, None, None, :]
    distinct = (I != J) & (I != K) & (I != L) & (J != K) & (J != L) & (K != L)
    y = (s & distinct[None]).any(axis=(2, 3, 4)).astype(np.int32)
    tok = np.concatenate([np.full((B, 1, 4), 3), f], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def _zero_sum_members(f, k):
    # f: [B, n, m] cards; returns [B, n] bool, card i is in some k-subset of distinct cards summing to 0
    import itertools
    B, n, _ = f.shape
    y = np.zeros((B, n), bool)
    for comb in itertools.combinations(range(n), k):
        hit = (f[:, list(comb)].sum(1) % 3 == 0).all(-1)
        if hit.any():
            y[:, list(comb)] |= hit[:, None]
    return y

def _deal(rng, B, n, m):
    ids = np.argsort(rng.random((B, 3 ** m)), axis=1)[:, :n]
    return np.stack([(ids // 3 ** a) % 3 for a in range(m)], -1)

def task_matchk(rng, B, n=10, k=4, m=4):
    # Match-k on Z_3^m: does card i lie in a k-subset of distinct cards summing to zero?
    f = _deal(rng, B, n, m)
    y = _zero_sum_members(f, k).astype(np.int32)
    tok = np.concatenate([np.full((B, 1, m), 3), f], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def task_mix(rng, B, n=9, m=4, ks=(3, 4, 5), fixed=None):
    # Match-k with k drawn per table from ks and announced by the first token ([k - 3, 3, 3, 3]):
    # one model, several arities, so features found on small k can serve large k.
    f = _deal(rng, B, n, m)
    k = np.full(B, fixed) if fixed else rng.choice(np.array(ks), B)
    y = np.zeros((B, n), np.int32)
    for kk in set(k.tolist()):
        y = np.where((k == kk)[:, None], _zero_sum_members(f, kk).astype(np.int32), y)
    bos = np.full((B, 1, m), 3)
    bos[:, 0, 0] = k - 3
    tok = np.concatenate([bos, f], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def task_arity(rng, B, n=9, m=4, ks=(3, 4, 5)):
    # The smallest k in ks for which card i completes a zero-sum k-subset (class 1..len(ks)), or 0 for none.
    f = _deal(rng, B, n, m)
    y = np.zeros((B, n), np.int32)
    for c, k in enumerate(ks):
        hit = _zero_sum_members(f, k)
        y = np.where((y == 0) & hit, c + 1, y)
    tok = np.concatenate([np.full((B, 1, m), 3), f], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def task_keyed(rng, B, U=3, g=10, k=3):
    # Keyed Match-k: every card carries one of U keys (g cards per key, shuffled). Card i is positive iff it
    # lies in a zero-sum k-subset of distinct cards WITH ITS OWN KEY. Zero sums across keys are distractors,
    # so the layer must select by key (query-dependent) and relate by group algebra (arity k) at once.
    import itertools
    n = U * g
    f = _deal(rng, B, n, 4)
    key = np.stack([rng.permutation(np.repeat(np.arange(U), g)) for _ in range(B)])
    y = np.zeros((B, n), bool)
    for comb in itertools.combinations(range(n), k):
        c = list(comb)
        hit = (f[:, c].sum(1) % 3 == 0).all(-1) & (key[:, c] == key[:, c[:1]]).all(-1)
        y[:, c] |= hit[:, None]
    bos = np.concatenate([np.full((B, 1, 4), 3), np.full((B, 1, 1), U)], -1)
    tok = np.concatenate([bos, np.concatenate([f, key[..., None]], -1)], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y.astype(np.int32)], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk

def task_keydist(rng, B, g=9, D=7, k=4, fixed=None):
    # Keyed Match-k with a distractor curriculum: g cards with key 0, d ~ U{0..D} distractor cards with key 1
    # (or d = fixed), and D - d padding tokens that are not cards. Every card is scored within its own key.
    # With d = 0 this is plain Match-k (Newton attention's signal exists); as d grows, zero sums across keys
    # multiply and only a layer that selects by key keeps the relation clean.
    import itertools
    n = g + D
    f = _deal(rng, B, n, 4)
    d = np.full(B, fixed) if fixed is not None else rng.integers(0, D + 1, B)
    slot = np.arange(n)[None]
    key = np.where(slot < g, 0, np.where(slot < g + d[:, None], 1, 2))   # 2 = padding
    perm = np.argsort(rng.random((B, n)), axis=1)
    f, key = np.take_along_axis(f, perm[..., None], 1), np.take_along_axis(key, perm, 1)
    y = np.zeros((B, n), bool)
    for comb in itertools.combinations(range(n), k):
        c = list(comb)
        hit = (f[:, c].sum(1) % 3 == 0).all(-1) & (key[:, c] == key[:, c[:1]]).all(-1) & (key[:, c[0]] < 2)
        y[:, c] |= hit[:, None]
    f = np.where((key == 2)[..., None], 3, f)                           # padding shows no card
    bos = np.concatenate([np.full((B, 1, 4), 3), np.full((B, 1, 1), 3)], -1)
    tok = np.concatenate([bos, np.concatenate([f, key[..., None]], -1)], 1)
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y.astype(np.int32)], 1)
    msk = np.concatenate([np.zeros((B, 1)), (key < 2).astype(np.float64)], 1)
    return tok, tgt, msk

def task_chase(rng, B, n=12, NID=16, P=8, maxlinks=4):
    # Linked lists; terminals point to themselves. The query token is a bare pointer to a start node.
    # Target: payload of the terminal reached from the start (0-3 hops after the first lookup).
    toks = np.zeros((B, n + 2, 3), np.int64); tgt = np.zeros((B, n + 2), np.int32); msk = np.zeros((B, n + 2))
    toks[:, 0] = [NID, NID, P]
    for b in range(B):
        ids = rng.permutation(NID)[:n]; pay = rng.integers(0, P, n)
        lens, left = [], n
        while left > 0:
            L = min(left, rng.integers(1, maxlinks + 1)); lens.append(L); left -= L
        ptr = np.empty(n, np.int64); term = np.empty(n, np.int64); pos = 0
        for L in lens:
            seg = np.arange(pos, pos + L)
            ptr[seg[:-1]] = ids[seg[1:]]; ptr[seg[-1]] = ids[seg[-1]]; term[seg] = seg[-1]; pos += L
        perm = rng.permutation(n)
        start = rng.integers(n)
        toks[b, 1:n + 1, 0] = ids[perm]; toks[b, 1:n + 1, 1] = ptr[perm]; toks[b, 1:n + 1, 2] = pay[perm]
        toks[b, n + 1] = [NID + 1, ids[start], P + 1]
        tgt[b, n + 1] = pay[term[start]]; msk[b, n + 1] = 1
    return toks, tgt, msk

def task_xortree(rng, B, n=14, NID=24, maxd=3):
    # Shuffled expression-tree records: LEAF(bit) | ID(child) | XOR(child1, child2). The query points at a
    # root; target = root value. Depth 1..maxd plus distractor trees. XOR of +-1 values = Hadamard product.
    NONE = NID; LEAF, IDN, XOR = 0, 1, 2
    toks = np.zeros((B, n + 2, 5), np.int64); tgt = np.zeros((B, n + 2), np.int32); msk = np.zeros((B, n + 2))
    toks[:, 0] = [NID + 1, 4, NID + 1, NID + 1, 3]
    def gen(depth, recs, ids):
        my = ids.pop()
        if depth == 0:
            b = int(rng.integers(2)); recs.append([my, LEAF, NONE, NONE, b]); return my, b
        if rng.random() < 0.7:
            c1, v1 = gen(depth - 1, recs, ids); c2, v2 = gen(int(rng.integers(0, depth)), recs, ids)
            if rng.random() < 0.5: c1, c2, v1, v2 = c2, c1, v2, v1
            recs.append([my, XOR, c1, c2, 2]); return my, v1 ^ v2
        c, v = gen(depth - 1, recs, ids); recs.append([my, IDN, c, NONE, 2]); return my, v
    for b in range(B):
        while True:
            ids = list(rng.permutation(NID)); recs = []
            root, val = gen(int(rng.integers(1, maxd + 1)), recs, ids)
            if len(recs) <= n - 2: break
        while len(recs) < n:
            extra = []
            try:
                gen(int(rng.integers(0, 2)), extra, ids)
            except IndexError:
                break
            if len(recs) + len(extra) <= n: recs += extra
        while len(recs) < n:
            recs.append([ids.pop() if ids else NONE, LEAF, NONE, NONE, int(rng.integers(2))])
        recs = np.array(recs)[rng.permutation(n)]
        toks[b, 1:n + 1] = recs
        toks[b, n + 1] = [NID + 1, 3, root, NONE, 3]
        tgt[b, n + 1] = val; msk[b, n + 1] = 1
    return toks, tgt, msk

def task_andortree(rng, B, n=14, NID=24, maxd=3):
    # Like task_xortree but with monotone gates on {0,1}: AND = Hadamard product, OR = 1-(1-a)(1-b).
    NONE = NID; LEAF, IDN, AND, OR = 0, 1, 2, 3
    toks = np.zeros((B, n + 2, 5), np.int64); tgt = np.zeros((B, n + 2), np.int32); msk = np.zeros((B, n + 2))
    toks[:, 0] = [NID + 1, 5, NID + 1, NID + 1, 3]
    def gen(depth, recs, ids):
        my = ids.pop()
        if depth == 0:
            b = int(rng.integers(2)); recs.append([my, LEAF, NONE, NONE, b]); return my, b
        r = rng.random()
        if r < 0.7:
            c1, v1 = gen(depth - 1, recs, ids); c2, v2 = gen(int(rng.integers(0, depth)), recs, ids)
            if rng.random() < 0.5: c1, c2, v1, v2 = c2, c1, v2, v1
            op = AND if r < 0.35 else OR
            recs.append([my, op, c1, c2, 2]); return my, (v1 & v2) if op == AND else (v1 | v2)
        c, v = gen(depth - 1, recs, ids); recs.append([my, IDN, c, NONE, 2]); return my, v
    for b in range(B):
        while True:
            ids = list(rng.permutation(NID)); recs = []
            root, val = gen(int(rng.integers(1, maxd + 1)), recs, ids)
            if len(recs) <= n - 2: break
        while len(recs) < n:
            extra = []
            try:
                gen(int(rng.integers(0, 2)), extra, ids)
            except IndexError:
                break
            if len(recs) + len(extra) <= n: recs += extra
        while len(recs) < n:
            recs.append([ids.pop() if ids else NONE, LEAF, NONE, NONE, int(rng.integers(2))])
        recs = np.array(recs)[rng.permutation(n)]
        toks[b, 1:n + 1] = recs
        toks[b, n + 1] = [NID + 1, 4, root, NONE, 3]
        tgt[b, n + 1] = val; msk[b, n + 1] = 1
    return toks, tgt, msk

TASKS = {
    "kzs3": (lambda r, B: task_keyed(r, B, 3, 10, 3), [4, 4, 4, 4, 4], 2, False),
    "kzs4": (lambda r, B: task_keyed(r, B, 2, 8, 4), [4, 4, 4, 4, 3], 2, False),
    "kd4": (lambda r, B: task_keydist(r, B, 9, 7, 4), [4, 4, 4, 4, 4], 2, False),
    "zs3": (lambda r, B: task_matchk(r, B, 12, 3, 4), [4, 4, 4, 4], 2, False),
    "zs4": (lambda r, B: task_matchk(r, B, 9, 4, 4), [4, 4, 4, 4], 2, False),
    "zs5": (lambda r, B: task_matchk(r, B, 9, 5, 4), [4, 4, 4, 4], 2, False),
    "arity": (lambda r, B: task_arity(r, B, 9, 4), [4, 4, 4, 4], 4, False),
    "mix345": (lambda r, B: task_mix(r, B, 9, 4, (3, 4, 5)), [4, 4, 4, 4], 2, False),
    "mix34": (lambda r, B: task_mix(r, B, 9, 4, (3, 4)), [4, 4, 4, 4], 2, False),
    "andortree": (task_andortree, [26, 6, 26, 26, 4], 2, False),
    "xortree": (task_xortree, [26, 5, 26, 26, 4], 2, False),
    "set4": (task_set4, [4, 4, 4, 4], 2, False),
    "compose3s": (lambda r, B: task_compose(r, B, 5, 3), [17, 6], 5, False),
    "chase": (task_chase, [18, 18, 10], 8, False),
    "setbind": (task_setbind, [4, 4, 4, 4, 3], 3, False),
    "match3s": (lambda r, B: task_match3(r, B, 8, 32), [129], 2, False),
    "match3m": (lambda r, B: task_match3(r, B, 12, 64), [129], 2, False),
    "pairbind": (task_pairbind, [65, 5], 2, False),
    "pairjoint": (task_pairjoint, [97, 6], 25, "multilabel"),
    "match3": (task_match3, [129], 2, False),
    # HOMA's Match3 benchmark: N = 6 with M = 30 and N = 8 with M = 56 (their Table 2); run with
    # --embed fourier --pos 0 --ffn 0 and their data budget (see homa_m3 in modal_run.py)
    "match3_n6": (lambda r, B: task_match3_homa(r, B, 6, 30), [30], 2, False),
    "match3_n8": (lambda r, B: task_match3_homa(r, B, 8, 56), [56], 2, False),
    "set": (task_set, [4, 4, 4, 4], 2, True),   # True = all fields share one table per field index
    "compose2": (lambda r, B: task_compose(r, B, 8, 2), [18, 9], 8, False),
    "compose3": (lambda r, B: task_compose(r, B, 8, 3), [26, 9], 8, False),
    "triangle": (task_triangle, [9, 9], 2, "shared"),
    # Ordered Match-3 over a group (does element i lie in positions i1 < i2 < i3 whose product
    # g_i1 g_i2 g_i3 is the identity?), two group sizes, one abelian and one non-abelian group at each
    "g3_z12": (lambda r, B: groups.task_gmatch(r, B, "Z12", 6, 3), [13], 2, False),
    "g3_a4": (lambda r, B: groups.task_gmatch(r, B, "A4", 6, 3), [13], 2, False),
    "g3_z60": (lambda r, B: groups.task_gmatch(r, B, "Z60", 11, 3), [61], 2, False),
    "g3_a5": (lambda r, B: groups.task_gmatch(r, B, "A5", 11, 3), [61], 2, False),
    # Two independent four-variable swap programs, three swaps each, then a namespace/variable query.
    "swap3": (groups.task_swap_program, [21], 4, False),
}
# The word problem (prefix products) over A5 and S5, as |G| classes (wp_) or as
# membership of the point stabiliser (ws_), at lengths 16 and 64
for _g, _n in (("a5", 60), ("s5", 120)):
    for _T in (16, 64):
        TASKS[f"wp_{_g}_{_T}"] = ((lambda g, T: lambda r, B: groups.task_word(r, B, g.upper(), T, "class"))(_g, _T), [_n + 1], _n, False)
        TASKS[f"ws_{_g}_{_T}"] = ((lambda g, T: lambda r, B: groups.task_word(r, B, g.upper(), T, "stab"))(_g, _T), [_n + 1], 2, False)
WORD_GROUP = {"a5": "A5", "s5": "S5"}

# ----------------------------------------------------------------------------- model
ORDER3 = {"tree", "strassen", "cubic", "cubic_add", "cubic_marg", "cubic_qgate", "cubic_wide"}

def dense(key, fan_in, fan_out):
    return {"w": jax.random.normal(key, (fan_in, fan_out)) / math.sqrt(fan_in), "b": jnp.zeros(fan_out)}

def mix_cfgs(cfg):
    # cubic, quartic, hyperchain (L=3) and path walk (L=4): one head each, head dim d/heads
    import copy
    out = []
    for m, L in (("cubic", 3), ("quartic", 3), ("chain", 3), ("path", 4)):
        c = copy.copy(cfg); c.model, c.plen, c.heads, c.dh, c.layers, c.recur = m, L, 1, cfg.d // cfg.heads, 1, 1
        out.append(c)
    return out

def init(key, cfg, vocabs, C, T):
    D, H, dh = cfg.d, cfg.heads, (getattr(cfg, "dh", 0) or cfg.d // cfg.heads)
    ks = iter(jax.random.split(key, 256))
    p = {"emb": [jax.random.normal(next(ks), (v, D)) * 0.5 for v in vocabs],
         "pos": jax.random.normal(next(ks), (T, D)) * 0.5, "layers": []}
    if cfg.embed == "fourier":
        # HOMA's input: a frozen sine-cosine basis of Z_M (M - 1 coordinates), then a learned projection
        p["emb"] = dense(next(ks), vocabs[0] - 1, D)
    if not cfg.pos:
        del p["pos"]
    if cfg.model == "mix":
        p["layers"] = []
        for _ in range(cfg.layers):
            subs = {}
            for sub in mix_cfgs(cfg):
                sp = init(next(ks), sub, vocabs, C, T)
                subs[sub.model + str(sub.plen)] = sp["layers"][0]["attn"]
            p["layers"].append({"ln1": {"g": jnp.ones(D), "b": jnp.zeros(D)}, "attn": subs,
                                "ln2": {"g": jnp.ones(D), "b": jnp.zeros(D)},
                                "fc1": dense(next(ks), D, 4 * D), "fc2": dense(next(ks), 4 * D, D)})
        p["lnf"] = {"g": jnp.ones(D), "b": jnp.zeros(D)}
        p["head"] = dense(next(ks), D, C)
        return p
    for _ in range(cfg.layers):
        names = {"sa": ["q", "k", "v"], "settransformer": ["q", "k", "v"],
                 "quartic": ["q", "k1", "k2", "k3", "v1", "v2", "v3"],
                 "chain": ["q0", "k0"] + [f"{n}{t}" for t in range(2, cfg.plen + 1) for n in ("q", "a", "b")] + [f"v{t}" for t in range(1, cfg.plen + 1)],
                 "path": ["q0", "k0", "q", "a", "b", "v1", "ve"],
                 "branch": ["q", "k", "q3", "u", "w", "v"],
                 "branchg": ["q", "k", "q3", "u", "w", "v", "leaf"],
                 "boson": ["q", "k1", "k2", "k3", "v1", "v2", "v3"],
                 "dspa": ["q", "u", "w", "v"],
                 "dspa2": ["q", "u", "w", "v"],
                 "dspa3": ["q", "u", "w", "v"],
                 "dspa4": ["q", "u", "w", "v"],
                 "dspa5": ["q", "u", "w", "v"],
                 "dpa": ["q", "k", "q3", "u", "w", "q4", "u4", "w4", "z4", "v"],
                 }.get(cfg.model, ["q", "k1", "k2", "v1", "v2"])
        dv = dh * (cfg.wide if cfg.model == "cubic_wide" else 1)
        if cfg.model == "cubic_full":
            dv = cfg.vdim
        a = {nm: dense(next(ks), D, H * (dv if nm.startswith("v") else dh)) for nm in names}
        a["o"] = dense(next(ks), H * (dv * dv if cfg.model == "cubic_full" else dv), D)
        if cfg.model == "deepsets":
            a = {"phi1": dense(next(ks), D, 2 * D), "phi2": dense(next(ks), 2 * D, D),
                 "rho": dense(next(ks), 2 * D, D)}
        if cfg.model == "newton":
            # Newton attention: r complex phase channels per head, every arity 1..K read out at once
            r, K = cfg.rank, cfg.order
            a = {"th": dense(next(ks), D, H * r), "g": dense(next(ks), D, H * r)}
            if cfg.readout == "shared":
                # one projection for every arity, and a learned per-arity scale on its output
                a["o"] = dense(next(ks), H * r * 2, D)
                a["alpha"] = jnp.ones((K, D))
            else:
                a["o"] = dense(next(ks), H * K * r * 2, D)
        if cfg.model == "holo":
            # holonomy attention: every token gets one d x d rotation per head and a gate; the layer reads
            # ordered products over subsets of positions, every arity 1..K, through one shared projection
            d, K = cfg.gdim, cfg.order
            a = {"s": dense(next(ks), D, H * d * d), "g": dense(next(ks), D, H),
                 "o": dense(next(ks), H * d * d, D), "alpha": jnp.ones((K + getattr(cfg, "prefix", 0), D))}
            if getattr(cfg, "hh", 0):
                # DeltaProduct's transitions instead of rotations: hh generalised Householders per token and head
                a["s"] = dense(next(ks), D, H * cfg.hh * d)
                a["hb"] = dense(next(ks), D, H * cfg.hh)
                a["hb"]["b"] = jnp.full(H * cfg.hh, float(getattr(cfg, "beta0", 0.0)))
        if cfg.model == "orthornn":
            # Input-dependent orthogonal matrix-state recurrence, a standard composition baseline.
            d = cfg.gdim
            a = {"s": dense(next(ks), D, H * d * d), "o": dense(next(ks), H * d * d, D)}
        if cfg.model == "deltaprod":
            # DeltaProduct (Siems et al. 2025): n_h generalised Householder steps per token on a matrix state,
            # beta in (0, 2) so the transitions can reflect (negative eigenvalues, Grazzi et al. 2024)
            nh = cfg.nh
            a = {"q": dense(next(ks), D, H * dh), "k": dense(next(ks), D, H * nh * dh), "v": dense(next(ks), D, H * nh * dh),
                 "beta": dense(next(ks), D, H * nh), "o": dense(next(ks), H * dh, D), "on": jnp.ones((H, dh))}
            # --beta0 b starts every beta at 2 sigmoid(b): 0 is DeltaNet's default (beta = 1, a projection that
            # erases the state along k), large b starts near reflections, which keep it
            a["beta"]["b"] = jnp.full(H * nh, float(getattr(cfg, "beta0", 0.0)))
        if cfg.model == "sfa":
            # subset attention: one softmax per head selects, Newton's identities read every arity from it
            r, K = cfg.rank, cfg.order
            a = {"q": dense(next(ks), D, H * dh), "k": dense(next(ks), D, H * dh),
                 "th": dense(next(ks), D, H * r), "g": dense(next(ks), D, H * r)}
            if cfg.qzero:
                # zero queries give uniform weights: the layer starts as Newton attention
                a["q"]["w"] = jnp.zeros_like(a["q"]["w"])
            if cfg.vpath:
                a["v"] = dense(next(ks), D, H * dh)
                a["ov"] = dense(next(ks), H * dh, D)
            a["o"] = dense(next(ks), H * r * 2, D)
            a["alpha"] = jnp.ones((K, D))
            if cfg.hops:
                # the walk sector: R hop matrices (untied), each a softmax over every token, self-loops allowed
                for l in range(cfg.hops):
                    a[f"hq{l}"], a[f"hk{l}"] = dense(next(ks), D, H * dh), dense(next(ks), D, H * dh)
                a["hv"] = dense(next(ks), D, H * dh)
                a["ow"] = dense(next(ks), cfg.hops * H * dh, D)
        if cfg.model == "path":
            a["len"] = dense(next(ks), D, H * cfg.plen)
        if cfg.model == "branch":
            a["gate"] = dense(next(ks), D, H * 3)
        if cfg.model == "branchg":
            a["leaf"] = dense(next(ks), D, H)
        if cfg.model in ("dspa", "dspa2", "dspa3", "dspa4"):
            for nm in ("u0", "w0", "uc", "wc"):
                a[nm] = jax.random.normal(next(ks), (H, dh))
            for nm in ("um", "wm", "qm"):
                a[nm] = jax.random.normal(next(ks), (H, cfg.modes, dh))
            a["tau"] = jnp.ones((H,)) if cfg.model == "dspa" else jnp.zeros((H,))
            a["clock"] = jnp.ones((cfg.plen, H, dh))
            if cfg.model in ("dspa2", "dspa3", "dspa4"):
                a["mu"] = jnp.tile(-jnp.log(jnp.array([1.0, T, T * T, T * cfg.modes, 1.0])), (H, 1))
        if cfg.model == "dspa5":
            for nm in ("u0", "w0", "uc", "wc", "us", "ws"):
                a[nm] = jax.random.normal(next(ks), (H, dh))
            for nm in ("um", "wm", "gk"):
                a[nm] = jax.random.normal(next(ks), (H, cfg.modes, dh))
            a["gb"] = jnp.zeros((H, cfg.modes))
            a["clock"] = jnp.ones((cfg.plen, H, dh))
            a["remb"] = jnp.zeros((cfg.plen, H, dh))
            # learned maps before every product, so complex character products are reachable
            for nm in ("Ls", "Rs", "Lt", "Rt", "Lm", "Rm"):
                a[nm] = jnp.tile(jnp.eye(dh), (H, 1, 1)) + 0.1 * jax.random.normal(next(ks), (H, dh, dh)) / math.sqrt(dh)
            M = cfg.modes
            # sectors: leaf | hop | cubic | carry | self | token x pool | pool x pool | pool
            sizes = [1.0, T, T * T, 1.0, 1.0, M, M * M, M]
            a["mu"] = jnp.tile(-jnp.log(jnp.array(sizes)), (H, 1))
        if cfg.model == "dpa":
            a["gate"] = dense(next(ks), D, H * 4)
            a["BL"] = jax.random.normal(next(ks), (H, cfg.modes, dh))
            a["BR"] = jax.random.normal(next(ks), (H, cfg.modes, dh))
            a["gam"] = jnp.ones((H,))
        if cfg.model == "boson":
            a["BL"] = jax.random.normal(next(ks), (H, cfg.modes, dh))
            a["BR"] = jax.random.normal(next(ks), (H, cfg.modes, dh))
            a["gam"] = jnp.ones((H,))
        layer = {"ln1": {"g": jnp.ones(D), "b": jnp.zeros(D)}, "attn": a}
        if cfg.ffn:
            layer.update({"ln2": {"g": jnp.ones(D), "b": jnp.zeros(D)},
                          "fc1": dense(next(ks), D, 4 * D), "fc2": dense(next(ks), 4 * D, D)})
        p["layers"].append(layer)
    p["lnf"] = {"g": jnp.ones(D), "b": jnp.zeros(D)}
    p["head"] = dense(next(ks), D, C)
    return p

def ln(p, x):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * p["g"] + p["b"]

def lin(p, x):
    return x @ p["w"] + p["b"]

def triple_mask(T):
    i = jnp.arange(T)
    I, J, K = i[:, None, None], i[None, :, None], i[None, None, :]
    return ((J != K) | (J == 0)) & (J != I) & (K != I)

def path_attention(cfg, a, x, heads, dh):
    # Globally normalised sum over pointer paths i -> j1 -> ... -> jL with query-dressed cubic transfer
    # matrices s^(i)_{jk} = <q_i ⊙ a_j, b_k>. "chain": untied steps, Hadamard of role values, arity L+1.
    # "path": tied steps (a transfer matrix), learned per-query mixture over lengths 1..L, values =
    # (first-hop value carried along) ⊙ (endpoint value). Self-loops allowed; the BOS token acts as a sink.
    B, T, D = x.shape; H = cfg.heads; sc = 1 / math.sqrt(dh); L = cfg.plen
    i = jnp.arange(T)
    first_ok = i[None, :] != i[:, None]                        # [i,k]: k != i
    step_ok = (i[None, None, :] != i[:, None, None])           # [i,j,k]: k != i
    q0, k0 = heads(lin(a["q0"], x)), heads(lin(a["k0"], x))
    alpha = jnp.where(first_ok, jnp.einsum("bhic,bhkc->bhik", q0, k0) * sc, -1e9)
    beta = jnp.broadcast_to(heads(lin(a["v1"], x))[:, :, None], (B, H, T, T, dh))
    alphas, betas = [alpha], [beta]
    for t in range(2, L + 1):
        tag = "" if cfg.model == "path" else str(t)
        q, aa, bb = heads(lin(a["q" + tag], x)), heads(lin(a["a" + tag], x)), heads(lin(a["b" + tag], x))
        s = jnp.where(step_ok, jnp.einsum("bhic,bhjc,bhkc->bhijk", q, aa, bb) * sc, -1e9)
        z = alpha[..., :, :, None] + s                        # [b,h,i,j,k]
        alpha = jax.scipy.special.logsumexp(z, axis=3)        # [b,h,i,k]
        w = jnp.exp(z - alpha[..., :, None, :])
        beta = jnp.einsum("bhijk,bhijc->bhikc", w, beta)
        if cfg.model == "chain":
            beta = beta * heads(lin(a[f"v{t}"], x))[:, :, None]
        alphas.append(alpha); betas.append(beta)
    if cfg.model == "chain":
        P = jax.nn.softmax(alpha, -1)
        return jnp.einsum("bhik,bhikc->bhic", P, beta)
    omega = heads(lin(a["len"], x))                           # [b,h,i,L] per-query length preference
    logits = jnp.stack(alphas, 3) + omega[..., None]          # [b,h,i,L,k]
    P = jax.nn.softmax(logits.reshape(B, H, T, -1), -1).reshape(B, H, T, L, T)
    ve = heads(lin(a["ve"], x))
    return jnp.einsum("bhitk,bhitkc,bhkc->bhic", P, jnp.stack(betas, 3), ve)

HI = jax.lax.Precision.HIGHEST

def subset_moments(p, phi, K, mu=0.2, bose=False):
    """
    Arity read-outs of a weighted token set with a Poisson vacuum, for m = 0..K-1:

        A_m(q) = [z^m] e^{mu z} prod_t (1 + z p_qt phi_t)  /  [z^m] e^{mu z} prod_t (1 + z p_qt)

    Without the vacuum (mu = 0) this is the mean of prod_{t in S} phi_t over distinct m-subsets S drawn with
    probability proportional to prod_{t in S} p_qt (a k-DPP with a diagonal kernel); A_1 is then softmax
    attention with values phi, and uniform p gives Newton attention. The vacuum lets a subset hold empty
    slots (value 1) at price mu each, so the read-out is a mixture of arities m, m - 1, ..., weighted by
    e_i(p) mu^(m-i) / (m-i)!. It is also what makes Newton's identities safe in float32: the denominator is
    at least mu^m / m!, so the error stays near eps ((1 + mu) / mu)^m however sharp p is. In power sums the
    vacuum is one term, mu added to the first. p: [B, H, Q, T] (zero on excluded tokens, rows summing to
    at most 1), phi: [B, H, T, r] complex. The power sums are K - 1 attention passes with weights p^j and
    values phi^j, at full float32 precision: TF32 matmuls would break the cancellations.
    bose: complete homogeneous polynomials instead (multisets, repeated tokens allowed).
    """
    pj, fj, P, Z = p, phi, [], []
    for j in range(1, K):
        P.append(jnp.einsum("bhqt,bhtc->bhqc", pj, fj.real, precision=HI)
                 + 1j * jnp.einsum("bhqt,bhtc->bhqc", pj, fj.imag, precision=HI) + (mu if j == 1 else 0.0))
        Z.append(pj.sum(-1, keepdims=True) + (mu if j == 1 else 0.0))
        pj, fj = pj * p, fj * phi
    sign = (lambda j: 1.0) if bose else (lambda j: (-1.0) ** (j - 1))
    one = jnp.ones(p.shape[:3] + (phi.shape[-1],), phi.dtype)
    E, N = [one], [jnp.ones(p.shape[:3] + (1,), p.dtype)]
    for m in range(1, K):
        E.append(sum(sign(j) * E[m - j] * P[j - 1] for j in range(1, m + 1)) / m)
        N.append(sum(sign(j) * N[m - j] * Z[j - 1] for j in range(1, m + 1)) / m)
    if mu > 0:
        return [E[m] / N[m] for m in range(K)]
    # no vacuum: exact canonical read-outs, zero where no m-subset has weight (ill-conditioned in float32)
    return [E[0]] + [jnp.where(N[m] > 1e-30, E[m] / jnp.where(N[m] > 1e-30, N[m], 1.0), 0.0) for m in range(1, K)]

def sfa_inputs(cfg, a, x, heads, dh):
    """Scores, the query's weights on the other tokens, and the complex channels of subset attention."""
    T = x.shape[1]
    q, k = heads(lin(a["q"], x)), heads(lin(a["k"], x))
    s = jnp.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(dh)
    if cfg.cap > 0:
        s = cfg.cap * jnp.tanh(s / cfg.cap)
    others = ~jnp.eye(T, dtype=bool)
    p = jax.nn.softmax(jnp.where(others, s, -1e9), -1) * others
    th = cfg.phase * heads(lin(a["th"], x))
    phi = jax.nn.sigmoid(heads(lin(a["g"], x))) * jnp.exp(1j * th)
    return s, p, phi

def newton_parts(cfg, a, feats, H, K, r, D):
    """Each arity's contribution to the layer output, [B, T, K, D], before the bias."""
    if cfg.readout == "shared" or cfg.model == "sfa":
        W = a["o"]["w"].reshape(H, 2 * r, D)
        return jnp.einsum("bthkc,hcd->btkd", feats, W) * a["alpha"][:K][None, None]
    W = a["o"]["w"].reshape(H, K, 2 * r, D)
    return jnp.einsum("bthkc,hkcd->btkd", feats, W)

def holonomy_sectors(M, K):
    """
    M: [B, T, H, d, d], one matrix per token and head. Returns K arrays [B, T, H, d, d]: entry m at position i
    is the sum over the (m+1)-subsets of positions that contain i of their products taken in position order,
    F_m(i) = sum_{a+b=m} L_a(i) M_i R_b(i), where L_a(i) sums the ordered a-products of positions before i and
    R_b(i) those after i. Commuting matrices give Newton attention's phi_i e_m(others); non-commuting ones
    see order. O(T K^2 d^3) per head.
    """
    B, T, H, d, _ = M.shape
    eye = jnp.broadcast_to(jnp.eye(d, dtype=M.dtype), (B, H, d, d))
    zero = jnp.zeros_like(eye)
    run, Ls = [eye] + [zero] * (K - 1), []
    for i in range(T):
        Ls.append(run)
        run = [eye] + [run[m] + run[m - 1] @ M[:, i] for m in range(1, K)]
    run, Rs = [eye] + [zero] * (K - 1), [None] * T
    for i in reversed(range(T)):
        Rs[i] = run
        run = [eye] + [run[m] + M[:, i] @ run[m - 1] for m in range(1, K)]
    return [jnp.stack([sum(Ls[i][b] @ M[:, i] @ Rs[i][m - b] for b in range(m + 1)) for i in range(T)], 1)
            for m in range(K)]


def holo_mats(cfg, a, x):
    """The rotations U = exp(A), A skew-symmetric and linear in the token, [..., H, d, d]; with --hh n instead
    the product of n generalised Householders I - beta k k^T, k unit, beta = 2 sigmoid in (0, 2) (DeltaProduct)."""
    d = cfg.gdim
    if getattr(cfg, "hh", 0):
        k = lin(a["s"], x).reshape(x.shape[:-1] + (cfg.heads, cfg.hh, d))
        k = k / jnp.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)
        beta = 2 * jax.nn.sigmoid(lin(a["hb"], x)).reshape(x.shape[:-1] + (cfg.heads, cfg.hh))
        U = jnp.broadcast_to(jnp.eye(d, dtype=x.dtype), x.shape[:-1] + (cfg.heads, d, d))
        for j in range(cfg.hh):
            Uk = jnp.einsum("...ij,...j->...i", U, k[..., j, :])
            U = U - beta[..., j, None, None] * jnp.einsum("...i,...j->...ij", Uk, k[..., j, :])
        return U
    S = lin(a["s"], x).reshape(x.shape[:-1] + (cfg.heads, d, d))
    return jax.scipy.linalg.expm((S - jnp.swapaxes(S, -1, -2)) * (cfg.phase / 2))


def orthogonal_prefix(U):
    """U: [B,T,H,d,d] -> H_t = U_t H_{t-1}, H_0 = I, at every position."""
    B, _, H, d, _ = U.shape
    initial = jnp.broadcast_to(jnp.eye(d, dtype=U.dtype), (B, H, d, d))
    def step(state, matrix):
        state = matrix @ state
        return state, state
    _, states = jax.lax.scan(step, initial, jnp.swapaxes(U, 0, 1))
    return jnp.swapaxes(states, 0, 1)


def holonomy_causal(M, U, K, prefix):
    """
    The causal holonomy sectors. M: gated matrices, U: the rotations without gates,
    both [B, T, H, d, d]. Returns K arrays [B, T, H, d, d], entry m at position i the sum over the
    (m+1)-subsets of positions <= i that contain i of their products in position order (L_m(i) M_i), and,
    with prefix, the product of every rotation up to i, U_0 U_1 ... U_i: the full-arity sector with its
    gates divided out, which is one running product (an orthogonal RNN with matrix state). O(T K d^3).
    """
    B, T, H, d, _ = M.shape
    eye = jnp.broadcast_to(jnp.eye(d, dtype=M.dtype), (B, H, d, d))

    def step(carry, inp):
        run, P = carry                     # run[m]: sum over m-subsets of earlier positions of their products
        Mi, Ui = inp
        out = jnp.einsum("kbhij,bhjl->kbhil", run, Mi) if K else run
        run = jnp.concatenate([run[:1], run[1:] + out[:-1]], 0) if K else run
        P = P @ Ui
        return (run, P), (out, P)

    run0 = jnp.concatenate([eye[None], jnp.zeros((max(K, 1) - 1, B, H, d, d), M.dtype)], 0)
    _, (outs, Ps) = jax.lax.scan(step, (run0, eye), (jnp.moveaxis(M, 1, 0), jnp.moveaxis(U, 1, 0)))
    secs = [jnp.moveaxis(outs[:, m], 0, 1) for m in range(K)]
    return secs + ([jnp.moveaxis(Ps, 0, 1)] if prefix else [])


def deltaprod_scan(q, k, v, beta):
    """
    q: [B, T, H, dk]; k: [B, T, H, nh, dk] (unit); v: [B, T, H, nh, dv]; beta: [B, T, H, nh]. Per token,
    nh delta-rule steps S <- S + beta (v - S k) k^T = S (I - beta k k^T) + beta v k^T on the state
    S [B, H, dv, dk], then o_t = S_t q_t. So S_t = S_0 A_1 ... A_t + writes, A_t a product of nh
    generalised Householders.
    """
    B, T, H, nh, dk = k.shape
    dv = v.shape[-1]

    def step(S, inp):
        qt, kt, vt, bt = inp
        for j in range(nh):
            Sk = jnp.einsum("bhvk,bhk->bhv", S, kt[:, :, j])
            S = S + bt[:, :, j, None, None] * jnp.einsum("bhv,bhk->bhvk", vt[:, :, j] - Sk, kt[:, :, j])
        return S, jnp.einsum("bhvk,bhk->bhv", S, qt)

    S0 = jnp.zeros((B, H, dv, dk), q.dtype)
    _, o = jax.lax.scan(step, S0, tuple(jnp.moveaxis(t, 1, 0) for t in (q, k, v, beta)))
    return jnp.moveaxis(o, 0, 1)                    # [B, T, H, dv]


def attention(cfg, a, x):
    B, T, D = x.shape
    H = cfg.heads; dh = getattr(cfg, "dh", 0) or D // H
    if cfg.model == "mix":   # one head per diagram type, each with its own projections; outputs summed
        return sum(attention(sub, a[sub.model + str(sub.plen)], x) for sub in mix_cfgs(cfg))
    def heads(t, w=dh):
        return t.reshape(B, T, H, -1).transpose(0, 2, 1, 3)
    if cfg.model == "orthornn":
        states = orthogonal_prefix(holo_mats(cfg, a, x))
        # Orthogonal matrices have entrywise RMS 1/sqrt(d), independent of sequence length.
        return lin(a["o"], states.reshape(B, T, -1) * math.sqrt(cfg.gdim))
    if cfg.model == "deepsets":
        features = lin(a["phi2"], jax.nn.gelu(lin(a["phi1"], x)))
        pooled = jnp.broadcast_to(features[:, 1:].mean(1, keepdims=True), features.shape)
        return lin(a["rho"], jnp.concatenate([features, pooled], -1))
    if cfg.model == "newton":
        # Newton attention. Channel features phi_i = g_i exp(i theta_i) (a character is one linear map away).
        # For every query q and m < K, the elementary symmetric polynomial e_m of the OTHER tokens (a sum over
        # distinct m-subsets) follows from leave-one-out power sums by Newton's identities, so phi_q * e_m is
        # the sum over (m+1)-subsets containing q of their channel products: arity m+1, linear in T.
        r, K = cfg.rank, (getattr(cfg, "maxarity", 0) or cfg.order)
        th = cfg.phase * heads(lin(a["th"], x))
        g = jax.nn.sigmoid(heads(lin(a["g"], x)))
        phi = g * jnp.exp(1j * th)
        partner = phi.at[:, :, 0, :].set(0) if cfg.bosfix else phi
        pw = [partner]
        for _ in range(K - 2):
            pw.append(pw[-1] * partner)
        pl = [P.sum(2, keepdims=True) - P for P in pw]
        e = [jnp.ones_like(phi)]
        for m in range(1, K):
            e.append(sum((-1) ** (j - 1) * e[m - j] * pl[j - 1] for j in range(1, m + 1)) / m)
        out = jnp.stack([phi * e[m] / math.comb(max(T - 2, m), m) for m in range(K)], 3)
        feats = jnp.concatenate([out.real, out.imag], -1).transpose(0, 2, 1, 3, 4)  # [B, T, H, K, 2r]
        if cfg.fnorm:
            # every arity's features on one scale, so no arity dominates the read-out by magnitude
            feats = feats / jnp.sqrt((feats ** 2).mean((2, 4), keepdims=True) + cfg.nfloor)
        parts = newton_parts(cfg, a, feats, H, K, r, D)
        if getattr(cfg, "knock", -1) >= 0:
            parts = parts * (jnp.arange(K) != cfg.knock)[None, None, :, None]
        if getattr(cfg, "want_parts", False):
            return parts
        return parts.sum(2) + a["o"]["b"]
    if cfg.model == "holo":
        # holonomy attention: ordered products of gated per-token rotations over every subset
        # of positions, arity 1..K, normalised per arity like Newton attention and read by one projection
        d, K = cfg.gdim, cfg.order
        U = holo_mats(cfg, a, x)
        M = jax.nn.sigmoid(lin(a["g"], x))[..., None, None] * U
        if getattr(cfg, "causal", 0):
            # subsets of positions <= i that end at i, divided by their count C(i, m); then the prefix product
            F = holonomy_causal(M, U, K, getattr(cfg, "prefix", 0))
            cnt = np.array([[max(math.comb(i, m), 1) for m in range(K)] + [1] * getattr(cfg, "prefix", 0) for i in range(T)], np.float32)
            feats = jnp.stack(F, 2) / cnt[None, :, :, None, None, None]
            feats = jnp.moveaxis(feats, 2, 3).reshape(B, T, H, K + getattr(cfg, "prefix", 0), d * d)
        else:
            F = holonomy_sectors(M, K)
            feats = jnp.stack([F[m] / math.comb(max(T - 1, m), m) for m in range(K)], 3).reshape(B, T, H, K, d * d)
        if cfg.fnorm:
            feats = feats / jnp.sqrt((feats ** 2).mean((2, 4), keepdims=True) + 1e-6)
        parts = jnp.einsum("bthkc,hcd->btkd", feats, a["o"]["w"].reshape(H, d * d, D)) * a["alpha"][None, None]
        if getattr(cfg, "want_parts", False):
            return parts
        return parts.sum(2) + a["o"]["b"]
    if cfg.model == "deltaprod":
        nh = cfg.nh
        unit = lambda t: t / jnp.sqrt((t ** 2).sum(-1, keepdims=True) + 1e-6)
        q = unit(lin(a["q"], x).reshape(B, T, H, dh))
        k = unit(lin(a["k"], x).reshape(B, T, H, nh, dh))
        v = lin(a["v"], x).reshape(B, T, H, nh, dh)
        beta = 2 * jax.nn.sigmoid(lin(a["beta"], x).reshape(B, T, H, nh))
        o = deltaprod_scan(q, k, v, beta)
        o = o / jnp.sqrt((o ** 2).mean(-1, keepdims=True) + 1e-6) * a["on"]   # per-head RMS norm, as DeltaNet
        return lin(a["o"], o.reshape(B, T, H * dh))
    if cfg.model == "sfa":
        # Subset attention. One softmax per head picks which tokens a query reads; Newton's identities on
        # the weighted power sums give, for every arity m + 1 <= K, the query's channels times the mean
        # channel product over distinct m-subsets of the others, drawn with probability prod p_qt, with
        # empty slots allowed at price vac. Arity 2 is softmax attention on the channels (plus a sink of
        # value 1); with vac = 0 and uniform weights it is Newton attention.
        r, K = cfg.rank, cfg.order
        s, p, phi = sfa_inputs(cfg, a, x, heads, dh)
        A = subset_moments(p, phi, K, cfg.vac, bose=cfg.stats == "bose")
        out = jnp.stack([phi * A[m] for m in range(K)], 3)
        feats = jnp.concatenate([out.real, out.imag], -1).transpose(0, 2, 1, 3, 4)  # [B, T, H, K, 2r]
        if cfg.fnorm:
            feats = feats / jnp.sqrt((feats ** 2).mean((2, 4), keepdims=True) + 1e-6)
        parts = newton_parts(cfg, a, feats, H, K, r, D)
        if getattr(cfg, "knock", -1) >= 0:
            parts = parts * (jnp.arange(K) != cfg.knock)[None, None, :, None]
        if getattr(cfg, "want_parts", False):
            return parts
        y = parts.sum(2) + a["o"]["b"]
        if cfg.vpath:
            # the same scores read linear values too, query included: standard attention, exactly
            v = heads(lin(a["v"], x))
            o = jnp.einsum("bhij,bhjc->bhic", jax.nn.softmax(s, -1), v)
            y = y + lin(a["ov"], o.transpose(0, 2, 1, 3).reshape(B, T, -1))
        if cfg.hops:
            # walks from the query: after l hops the walker is at M_l = A_1 A_2 ... A_l, and reads the values
            # there. Arity counts how many tokens meet at once; walks count how many steps lie between them.
            hv = heads(lin(a["hv"], x))
            M, walks = None, []
            for l in range(cfg.hops):
                A_l = jax.nn.softmax(jnp.einsum("bhic,bhjc->bhij", heads(lin(a[f"hq{l}"], x)), heads(lin(a[f"hk{l}"], x))) / math.sqrt(dh), -1)
                M = A_l if M is None else jnp.einsum("bhij,bhjk->bhik", M, A_l)
                walks.append(jnp.einsum("bhik,bhkc->bhic", M, hv))
            y = y + lin(a["ow"], jnp.concatenate(walks, -1).transpose(0, 2, 1, 3).reshape(B, T, -1))
        return y
    if cfg.model in ("dspa", "dspa2", "dspa3", "dspa4"):
        # Dyson-Schwinger polyattention: ONE trilinear vertex <q_p * clock_r, u_a * w_b> over
        # tokens + vacuum (value = 1, the Hadamard identity) + M virtual modes. A single softmax per parent
        # chooses leaf (vac,vac) | hop (j,vac) | cubic (j,k) | exchange (j,m); values recurse R rounds.
        q, u, w = (heads(lin(a[n], x)) for n in ("q", "u", "w"))
        v = jnp.tanh(heads(lin(a["v"], x)))
        M = cfg.modes; sc = 1 / math.sqrt(dh)
        i_ = jnp.arange(T)
        offd = i_[:, None] != i_[None, :]
        trip = triple_mask(T)
        Phi = v
        for r in range(cfg.plen):
            om = a["clock"][r]
            qr = q * om[None, :, None, :]
            Sm = jnp.einsum("hmc,bhkc,bhlc->bhmkl", a["qm"] * om[:, None, :], u, w) * sc
            Sm = jnp.where(offd, Sm, -1e9).reshape(B, H, M, T * T)
            Fm = jax.scipy.special.logsumexp(Sm, -1)
            Pm = jax.nn.softmax(Sm, -1).reshape(B, H, M, T, T)
            Phim = jnp.einsum("bhmkl,bhkc,bhlc->bhmc", Pm, Phi, Phi)
            leaf = jnp.einsum("bhpc,hc->bhp", qr, a["u0"] * a["w0"])[..., None] * sc
            hop = jnp.where(offd, jnp.einsum("bhpc,bhjc,hc->bhpj", qr, u, a["w0"]) * sc, -1e9)
            cub = jnp.where(trip, jnp.einsum("bhpc,bhjc,bhkc->bhpjk", qr, u, w) * sc, -1e9)
            exc = jnp.einsum("bhpc,bhjc,hmc->bhpjm", qr, u, a["wm"]) * sc + a["tau"][None, :, None, None, None] * Fm[:, :, None, None, :]
            exc = jnp.where(offd[None, None, :, :, None], exc, -1e9)
            if cfg.model in ("dspa2", "dspa3", "dspa4"):   # sector chemical potentials: equal prior mass per diagram type at init
                mu = a["mu"][None, :, None, :]
                leaf, hop = leaf + mu[..., 0:1], hop + mu[..., 1:2]
                cub, exc = cub + mu[..., 2:3, None], exc + mu[..., 3:4, None]
            if cfg.nointeract:
                cub, exc = jnp.full_like(cub, -1e9), jnp.full_like(exc, -1e9)
            parts = [leaf, hop, cub.reshape(B, H, T, T * T), exc.reshape(B, H, T, T * M)]
            if cfg.model == "dspa4":   # absorbing carry state: keep this token's previous-round value
                parts.append(jnp.einsum("bhpc,hc->bhp", qr, a["uc"] * a["wc"])[..., None] * sc + a["mu"][None, :, None, 4:5])
            z = jnp.concatenate(parts, -1)
            P = jax.nn.softmax(z, -1)
            pl, ph = P[..., 0:1], P[..., 1:T + 1]
            pc = P[..., T + 1:T + 1 + T * T].reshape(B, H, T, T, T)
            pe = P[..., T + 1 + T * T:T + 1 + T * T + T * M].reshape(B, H, T, T, M)
            carry = P[..., T + 1 + T * T + T * M:] * Phi if cfg.model == "dspa4" else 0.0
            Phi = carry + (pl * v + jnp.einsum("bhpj,bhjc->bhpc", ph, Phi)
                   + jnp.einsum("bhpjk,bhjc,bhkc->bhpc", pc, Phi, Phi)
                   + jnp.einsum("bhpjm,bhjc,bhmc->bhpc", pe, Phi, Phim))
        if cfg.model in ("dspa3", "dspa4"):   # products of bounded values shrink; renormalise at readout
            Phi = Phi * jax.lax.rsqrt(jnp.mean(Phi ** 2, -1, keepdims=True) + 1e-6)
        y = Phi
    elif cfg.model == "dspa5":
        # DS-PA v5: the primitives Newton's identities need, offered in one softmax per token and round.
        # Sum-pooled modes S_m = sum_t sigmoid(gate) Phi_t (counts, not averages, empty buckets allowed),
        # a self-product Phi_p * v_p (Hadamard powers), token x pool and pool x pool products (the
        # recursion), plus the v4 sectors. Signed combinations happen in the output projection and MLP.
        q, u, w = (heads(lin(a[n], x)) for n in ("q", "u", "w"))
        v = jnp.tanh(heads(lin(a["v"], x)))
        M = cfg.modes; sc = 1 / math.sqrt(dh)
        i_ = jnp.arange(T)
        offd = i_[:, None] != i_[None, :]
        trip = triple_mask(T)
        tok_map = lambda W, X: jnp.einsum("hcd,bhtd->bhtc", W, X)
        use_cubic = not cfg.nocubic
        Phi = v
        for r in range(cfg.plen):
            om = a["clock"][r]
            qr = q * om[None, :, None, :] + a["remb"][r][None, :, None, :]
            gate = jax.nn.sigmoid(jnp.einsum("bhtc,hmc->bhmt", u * om[None, :, None, :], a["gk"]) * sc
                                  + a["gb"][None, :, :, None])
            S = jnp.einsum("bhmt,bhtc->bhmc", gate, Phi) / T
            mu = a["mu"][None, :, None, :]
            parts = [
                jnp.einsum("bhpc,hc->bhp", qr, a["u0"] * a["w0"])[..., None] * sc + mu[..., 0:1],
                jnp.where(offd, jnp.einsum("bhpc,bhjc,hc->bhpj", qr, u, a["w0"]) * sc, -1e9) + mu[..., 1:2],
            ]
            if use_cubic:
                cub = jnp.where(trip, jnp.einsum("bhpc,bhjc,bhkc->bhpjk", qr, u, w) * sc, -1e9) + mu[..., 2:3, None]
                parts.append(cub.reshape(B, H, T, T * T))
            parts += [
                jnp.einsum("bhpc,hc->bhp", qr, a["uc"] * a["wc"])[..., None] * sc + mu[..., 3:4],
                jnp.einsum("bhpc,hc->bhp", qr, a["us"] * a["ws"])[..., None] * sc + mu[..., 4:5],
                jnp.einsum("bhpc,hc,hmc->bhpm", qr, a["u0"], a["wm"]) * sc + mu[..., 5:6],
                (jnp.einsum("bhpc,hmc,hnc->bhpmn", qr, a["um"], a["wm"]) * sc + mu[..., 6:7, None]).reshape(B, H, T, M * M),
                jnp.einsum("bhpc,hmc,hc->bhpm", qr, a["um"], a["w0"]) * sc + mu[..., 7:8],
            ]
            P = jax.nn.softmax(jnp.concatenate(parts, -1), -1)
            o = 0
            def take(width):
                nonlocal o
                out = P[..., o:o + width]; o += width
                return out
            pl, ph = take(1), take(T)
            pc = take(T * T).reshape(B, H, T, T, T) if use_cubic else None
            pcar, pself, ptm = take(1), take(1), take(M)
            pmm, ppm = take(M * M).reshape(B, H, T, M, M), take(M)
            new = (pl * v + jnp.einsum("bhpj,bhjc->bhpc", ph, Phi) + pcar * Phi
                   + pself * (tok_map(a["Ls"], Phi) * tok_map(a["Rs"], v))
                   + jnp.einsum("bhpm,bhpc,bhmc->bhpc", ptm, tok_map(a["Lt"], Phi), tok_map(a["Rt"], S))
                   + jnp.einsum("bhpmn,bhmc,bhnc->bhpc", pmm, tok_map(a["Lm"], S), tok_map(a["Rm"], S))
                   + jnp.einsum("bhpm,bhmc->bhpc", ppm, S))
            if use_cubic:
                new = new + jnp.einsum("bhpjk,bhjc,bhkc->bhpc", pc, Phi, Phi)
            Phi = new
        Phi = Phi * jax.lax.rsqrt(jnp.mean(Phi ** 2, -1, keepdims=True) + 1e-6)
        y = Phi
    elif cfg.model == "dpa":
        # Diagrammatic polyattention: tree-level resummation with four Feynman rules sharing one value field H:
        # leaf (stop), propagator (pairwise hop), cubic 3-vertex, and a 4-vertex split by boson exchange.
        q, k, q3, u, w, q4, u4, w4, z4 = (heads(lin(a[n], x)) for n in ("q", "k", "q3", "u", "w", "q4", "u4", "w4", "z4"))
        v = jnp.tanh(heads(lin(a["v"], x)))
        i_ = jnp.arange(T)
        A = jax.nn.softmax(jnp.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(dh), -1)
        s3 = jnp.where(triple_mask(T), jnp.einsum("bhic,bhjc,bhkc->bhijk", q3, u, w) / math.sqrt(dh), -1e9)
        C = jax.nn.softmax(s3.reshape(B, H, T, T * T), -1).reshape(B, H, T, T, T)
        pair_ok = (i_[:, None] != i_[None, :]) | (i_[:, None] == 0)
        R = jnp.where(pair_ok, jnp.einsum("hmc,bhkc,bhlc->bhmkl", a["BR"], w4, z4) / math.sqrt(dh), -1e9).reshape(B, H, cfg.modes, T * T)
        Z = jax.scipy.special.logsumexp(R, -1)
        PR = jax.nn.softmax(R, -1).reshape(B, H, cfg.modes, T, T)
        Lg = jnp.einsum("hmc,bhic,bhjc->bhijm", a["BL"], q4, u4) / math.sqrt(dh) + a["gam"][None, :, None, None, None] * Z[:, :, None, None, :]
        Lg = jnp.where((i_[:, None] != i_[None, :])[None, None, :, :, None], Lg, -1e9)
        PL = jax.nn.softmax(Lg.reshape(B, H, T, -1), -1).reshape(B, H, T, T, cfg.modes)
        g = jax.nn.softmax(heads(lin(a["gate"], x)), -1)       # stop / hop / 3-vertex / 4-vertex
        Hs = v
        for _ in range(cfg.plen):
            U = jnp.einsum("bhmkl,bhkc,bhlc->bhmc", PR, Hs, Hs)
            Hs = (g[..., 0:1] * v + g[..., 1:2] * jnp.einsum("bhij,bhjc->bhic", A, Hs)
                  + g[..., 2:3] * jnp.einsum("bhijk,bhjc,bhkc->bhic", C, Hs, Hs)
                  + g[..., 3:4] * jnp.einsum("bhijm,bhjc,bhmc->bhic", PL, Hs, U))
        y = Hs
    elif cfg.model == "branchg":
        # Berends-Giele / inside recursion: options at node i are leaf, hop to j, or branch to (j,k); each
        # option is dressed by the children's free energies F, so the tree distribution is globally normalised.
        q, k, q3, u, w = (heads(lin(a[n], x)) for n in ("q", "k", "q3", "u", "w"))
        v = jnp.tanh(heads(lin(a["v"], x)))
        sl = heads(lin(a["leaf"], x))[..., 0]                   # [b,h,i] leaf energy
        i_ = jnp.arange(T)
        A0 = jnp.where(i_[None, :] != i_[:, None], jnp.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(dh), -1e9)
        C0 = jnp.where(triple_mask(T), jnp.einsum("bhic,bhjc,bhkc->bhijk", q3, u, w) / math.sqrt(dh), -1e9)
        F, Hs = sl, v
        for _ in range(cfg.plen):
            un = A0 + F[:, :, None, :]
            bi = (C0 + F[:, :, None, :, None] + F[:, :, None, None, :]).reshape(B, H, T, T * T)
            z = jnp.concatenate([sl[..., None], un, bi], -1)
            F = jax.scipy.special.logsumexp(z, -1)
            p = jax.nn.softmax(z, -1)
            pl, pu, pb = p[..., :1], p[..., 1:T + 1], p[..., T + 1:].reshape(B, H, T, T, T)
            Hs = pl * v + jnp.einsum("bhij,bhjc->bhic", pu, Hs) + jnp.einsum("bhijk,bhjc,bhkc->bhic", pb, Hs, Hs)
        y = Hs
    elif cfg.model == "boson":
        # Hubbard-Stratonovich split of a 4-body vertex: (k,l) pairs are hashed into M virtual modes once;
        # each (i,j) then couples to a mode. Arity 4 at O(n^2 M d) instead of O(n^4 d).
        q, k1, k2, k3, v1, v2, v3 = (heads(lin(a[n], x)) for n in ("q", "k1", "k2", "k3", "v1", "v2", "v3"))
        i_ = jnp.arange(T)
        pair_ok = (i_[:, None] != i_[None, :]) | (i_[:, None] == 0)
        R = jnp.einsum("hmc,bhkc,bhlc->bhmkl", a["BR"], k2, k3) / math.sqrt(dh)
        R = jnp.where(pair_ok, R, -1e9).reshape(B, H, cfg.modes, T * T)
        Z = jax.scipy.special.logsumexp(R, -1)                  # [b,h,m]
        PR = jax.nn.softmax(R, -1).reshape(B, H, cfg.modes, T, T)
        U = jnp.einsum("bhmkl,bhkc,bhlc->bhmc", PR, v2, v3)     # [b,h,m,c]
        Lg = jnp.einsum("hmc,bhic,bhjc->bhijm", a["BL"], q, k1) / math.sqrt(dh) + a["gam"][None, :, None, None, None] * Z[:, :, None, None, :]
        Lg = jnp.where((i_[:, None] != i_[None, :])[None, None, :, :, None], Lg, -1e9)
        P = jax.nn.softmax(Lg.reshape(B, H, T, -1), -1).reshape(B, H, T, T, cfg.modes)
        y = jnp.einsum("bhijm,bhjc,bhmc->bhic", P, v1, U)
    elif cfg.model == "branch":
        # Branching resummation (tree-level Dyson-Schwinger for a phi^3 theory): random computation trees
        # whose vertices stop (leaf value), hop (pairwise attention) or branch (cubic attention, value product).
        q, k, q3, u, w = (heads(lin(a[n], x)) for n in ("q", "k", "q3", "u", "w"))
        v = jnp.tanh(heads(lin(a["v"], x)))
        A = jax.nn.softmax(jnp.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(dh), -1)
        s = jnp.where(triple_mask(T), jnp.einsum("bhic,bhjc,bhkc->bhijk", q3, u, w) / math.sqrt(dh), -1e9)
        C = jax.nn.softmax(s.reshape(B, H, T, T * T), -1).reshape(B, H, T, T, T)
        g = jax.nn.softmax(heads(lin(a["gate"], x)), -1)       # [b,h,i,3] stop / hop / branch
        Hs = v
        for _ in range(cfg.plen):
            Hs = (g[..., 0:1] * v + g[..., 1:2] * jnp.einsum("bhij,bhjc->bhic", A, Hs)
                  + g[..., 2:3] * jnp.einsum("bhijk,bhjc,bhkc->bhic", C, Hs, Hs))
        y = Hs
    elif cfg.model in ("chain", "path"):
        y = path_attention(cfg, a, x, heads, dh)
    elif cfg.model == "quartic":
        q, k1, k2, k3, v1, v2, v3 = (heads(lin(a[n], x)) for n in ("q", "k1", "k2", "k3", "v1", "v2", "v3"))
        s = jnp.einsum("bhic,bhjc,bhkc,bhlc->bhijkl", q, k1, k2, k3) / math.sqrt(dh)
        i = jnp.arange(T)
        I, J, K, L = i[:, None, None, None], i[None, :, None, None], i[None, None, :, None], i[None, None, None, :]
        ok = (J != I) & (K != I) & (L != I) & ((J != K) | (J == 0)) & ((J != L) | (J == 0)) & ((K != L) | (K == 0))
        s = jnp.where(ok, s, -1e9)
        P = jax.nn.softmax(s.reshape(B, H, T, T ** 3), -1).reshape(B, H, T, T, T, T)
        Dv = v1[:, :, :, None, None, :] * v2[:, :, None, :, None, :] * v3[:, :, None, None, :, :]
        y = jnp.einsum("bhijkl,bhjklc->bhic", P, Dv)
    elif cfg.model in ("sa", "settransformer"):
        q, k, v = (heads(lin(a[n], x)) for n in ("q", "k", "v"))
        s = jnp.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(dh)
        if getattr(cfg, "causal", 0):
            s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -1e9)
        y = jnp.einsum("bhij,bhjc->bhic", jax.nn.softmax(s, -1), v)
    else:
        q, k1, k2, v1, v2 = (heads(lin(a[n], x)) for n in ("q", "k1", "k2", "v1", "v2"))
        sc = 1 / math.sqrt(dh)
        if cfg.model == "tree":       # h = x1x2 + x2x3  (tree attention)
            s = (jnp.einsum("bhic,bhjc->bhij", q, k1)[..., None] + jnp.einsum("bhjc,bhkc->bhjk", k1, k2)[:, :, None]) * sc
        elif cfg.model == "strassen":  # h = x1x2 + x2x3 + x3x1
            s = (jnp.einsum("bhic,bhjc->bhij", q, k1)[..., None] + jnp.einsum("bhjc,bhkc->bhjk", k1, k2)[:, :, None]
                 + jnp.einsum("bhic,bhkc->bhik", q, k2)[:, :, :, None, :]) * sc
        else:                          # h = x1x2x3  (trilinear polyattention)
            s = jnp.einsum("bhic,bhjc,bhkc->bhijk", q, k1, k2) * sc
        if not cfg.repeat:   # --repeat 1: every pair (j, k), as HOMA's triadic pathway
            s = jnp.where(triple_mask(T), s, -1e9)
        P = jax.nn.softmax(s.reshape(B, H, T, T * T), -1).reshape(B, H, T, T, T)
        if cfg.model == "cubic_add":     # additive pair value: no binding
            y = jnp.einsum("bhijk,bhjc->bhic", P, v1) + jnp.einsum("bhijk,bhkc->bhic", P, v2)
        elif cfg.model == "cubic_marg":  # sum out k: plain pairwise value of j
            y = jnp.einsum("bhijk,bhjc->bhic", P, v1)
        elif cfg.model == "cubic_qgate":  # the official tree_simple indexing: V at query i
            y = v1 * jnp.einsum("bhijk,bhkc->bhic", P, v2)
        elif cfg.model == "cubic_full":  # full pair-moment M_i = V1^T P_i V2 (Hadamard keeps only its diagonal)
            y = jnp.einsum("bhja,bhijk,bhkc->bhiac", v1, P, v2).reshape(B, H, T, -1)
        else:                            # D_jk = V1_j ⊙ V2_k
            y = jnp.einsum("bhijk,bhjc,bhkc->bhic", P, v1, v2)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return lin(a["o"], y)

def fourier(x, M):
    """The real Fourier basis of Z_M without the constant: cos for k = 1..M//2, sin for k = 1..(M-1)//2."""
    a = 2 * jnp.pi / M * x[..., None]
    return jnp.concatenate([jnp.cos(a * jnp.arange(1, M // 2 + 1)), jnp.sin(a * jnp.arange(1, (M - 1) // 2 + 1))], -1)

# the 40 characters of Z_3^4 up to conjugation (first nonzero coordinate 1), and a fixed rotation of them
CHAR_REPS = np.array([a for a in itertools.product(range(3), repeat=4) if any(a) and a[next(i for i, v in enumerate(a) if v)] == 1], float)
CHAR_ROT = np.linalg.qr(np.random.default_rng(1234).normal(size=(80, 80)))[0]


def char_embed(cfg, tok):
    """
    A frozen embedding of Z_3^4 cards by their characters, [X, -X, bos, 0...] with
    X = (cos, sin)(2 pi a.c / 3) over the 40 conjugate pairs a, so every card has zero mean and the same norm
    and LayerNorm is a constant scale. 'charrot' rotates X by a fixed orthogonal matrix: the same Gram
    matrix (the same geometry as one-hot) in other coordinates.
    """
    f = tok[..., :4]
    bos = f[..., 0] == 3
    ang = (2 * jnp.pi / 3) * (jnp.where(bos[..., None], 0, f).astype(jnp.float32) @ jnp.asarray(CHAR_REPS.T, jnp.float32))
    X = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], -1) * (~bos)[..., None]
    if cfg.embed == "charrot":
        X = X @ jnp.asarray(CHAR_ROT, jnp.float32)
    assert cfg.d >= 161, "--embed char needs --d >= 161"
    pad = jnp.zeros(X.shape[:-1] + (cfg.d - 161,), X.dtype)
    return jnp.concatenate([X, -X, 4.0 * bos[..., None].astype(X.dtype), pad], -1)


def embed(cfg, p, tok):
    if cfg.embed in ("char", "charrot"):
        x = char_embed(cfg, tok)
    elif cfg.embed == "fourier":
        x = lin(p["emb"], fourier(tok[..., 0], p["emb"]["w"].shape[0] + 1))
    else:
        x = sum(p["emb"][f][tok[..., f]] for f in range(tok.shape[-1]))
    return x + p["pos"][None, : tok.shape[1]] if "pos" in p else x

def forward(cfg, p, tok):
    x = embed(cfg, p, tok)
    for L in p["layers"]:
        for _ in range(cfg.recur):
            # Newton attention can read its phases from the raw residual stream: LayerNorm divides each
            # token by its own norm, which breaks the additivity a character needs (theta(a+b) = theta(a)+theta(b))
            if cfg.model == "settransformer":
                x = ln(L["ln1"], x + attention(cfg, L["attn"], x))
                x = ln(L["ln2"], x + lin(L["fc2"], jax.nn.gelu(lin(L["fc1"], x))))
                continue
            h = x if (cfg.model == "newton" and cfg.rawphase) else ln(L["ln1"], x)
            x = x + attention(cfg, L["attn"], h)
            if "fc1" in L:
                x = x + lin(L["fc2"], jax.nn.gelu(lin(L["fc1"], ln(L["ln2"], x))))
    return lin(p["head"], ln(p["lnf"], x))

# ----------------------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="match3"); ap.add_argument("--model", default="cubic")
    ap.add_argument("--layers", type=int, default=1); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--wide", type=int, default=4); ap.add_argument("--vdim", type=int, default=8); ap.add_argument("--plen", type=int, default=3); ap.add_argument("--recur", type=int, default=1); ap.add_argument("--modes", type=int, default=81); ap.add_argument("--nocubic", type=int, default=0); ap.add_argument("--rank", type=int, default=32); ap.add_argument("--order", type=int, default=6); ap.add_argument("--phase", type=float, default=2.0); ap.add_argument("--rawphase", type=int, default=0); ap.add_argument("--readout", default="full"); ap.add_argument("--fnorm", type=int, default=0); ap.add_argument("--diag", type=int, default=1); ap.add_argument("--gdim", type=int, default=3)
    # word problems: --causal 1 (softmax and holonomy attention), --prefix 1 (holonomy's
    # running product as one more sector; with --order 0 it is the only one, an orthogonal RNN with matrix
    # state), --nh Householders per token (--model deltaprod), --evallen L also scores at length L
    ap.add_argument("--causal", type=int, default=0); ap.add_argument("--prefix", type=int, default=0)
    ap.add_argument("--nh", type=int, default=2); ap.add_argument("--evallen", type=int, default=0)
    ap.add_argument("--beta0", type=float, default=0.0); ap.add_argument("--hh", type=int, default=0)
    # subset attention (--model sfa): Poisson vacuum, optional logit cap, zero-initialised queries, statistics, linear-value path
    ap.add_argument("--vac", type=float, default=0.2); ap.add_argument("--cap", type=float, default=0.0); ap.add_argument("--qzero", type=int, default=1); ap.add_argument("--stats", default="fermi"); ap.add_argument("--vpath", type=int, default=1); ap.add_argument("--hops", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=0)
    # backbone: --embed fourier (frozen Fourier basis of Z_M, then a learned projection), --pos 0 (no
    # positional embeddings), --ffn 0 (no MLP sublayer); --repeat 1 lets cubic attention pick j = k or i
    ap.add_argument("--embed", default="table"); ap.add_argument("--pos", type=int, default=1)
    ap.add_argument("--ffn", type=int, default=1); ap.add_argument("--repeat", type=int, default=0)
    # data and optimiser: --trainset N trains --epochs passes over N fixed examples instead of a fresh
    # batch every step; --evalset is the held-out size; --sched const --wd 0 --clip 0 is plain Adam
    ap.add_argument("--trainset", type=int, default=0); ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--evalset", type=int, default=2048); ap.add_argument("--sched", default="cosine")
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--bosfix", type=int, default=0)
    ap.add_argument("--nfloor", type=float, default=1e-6)
    ap.add_argument("--nointeract", type=int, default=0)
    ap.add_argument("--evaloffset", type=int, default=10000)
    ap.add_argument("--save-final", type=int, default=0)
    ap.add_argument("--maxarity", type=int, default=0)
    cfg = ap.parse_args()
    assert not cfg.maxarity or (cfg.model == "newton" and cfg.readout == "shared" and 1 <= cfg.maxarity <= cfg.order)
    gen, vocabs, C, kind = TASKS[cfg.task]
    multilabel = kind == "multilabel"
    rng = np.random.default_rng(cfg.seed)
    tok0, _, _ = gen(rng, 2); T = max(tok0.shape[1], cfg.evallen + 1)
    if cfg.trainset:
        # a fixed training set, reshuffled every epoch; the last batch of an epoch may be short
        dtok, dtgt, dmsk = gen(rng, cfg.trainset)
        per_epoch = -(-cfg.trainset // cfg.batch)
        cfg.steps = cfg.epochs * per_epoch
    params = init(jax.random.PRNGKey(cfg.seed), cfg, vocabs, C, T)
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    sched = cfg.lr if cfg.sched == "const" else optax.warmup_cosine_decay_schedule(0.0, cfg.lr, min(200, cfg.steps // 10), cfg.steps, cfg.lr * 0.02)
    clip = [optax.clip_by_global_norm(cfg.clip)] if cfg.clip > 0 else []
    opt = optax.chain(*clip, optax.adamw(sched, weight_decay=cfg.wd))
    st = opt.init(params)

    def loss_fn(p, tok, tgt, msk):
        lg = forward(cfg, p, tok)
        if multilabel:
            ce = optax.sigmoid_binary_cross_entropy(lg, tgt).sum(-1)
        else:
            ce = optax.softmax_cross_entropy_with_integer_labels(lg, tgt)
        return (ce * msk).sum() / msk.sum()

    @jax.jit
    def step(p, st, tok, tgt, msk):
        l, g = jax.value_and_grad(loss_fn)(p, tok, tgt, msk)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l

    def eval_with(c):
        return jax.jit(lambda p, tok, tgt, msk: eval_impl(c, p, tok, tgt, msk))

    def eval_impl(c, p, tok, tgt, msk):
        word = c.task[:3] in ("wp_", "ws_")
        lg = forward(c, p, tok)
        if multilabel:
            hit = ((lg > 0) == (tgt > 0.5)).all(-1)
        else:
            hit = lg.argmax(-1) == tgt
        ok = hit | (msk == 0)
        tokacc = (hit * msk).sum() / msk.sum()
        if multilabel:
            return tokacc, ok.all(-1).mean(), tokacc
        # balanced accuracy: mean per-class recall over the classes present, immune to label imbalance
        recalls = []
        present = []
        for c in range(C):
            cm = (tgt == c) * msk
            recalls.append((hit * cm).sum() / jnp.maximum(cm.sum(), 1))
            present.append((cm.sum() > 0).astype(jnp.float32))
        bal = (jnp.stack(recalls) * jnp.stack(present)).sum() / jnp.maximum(jnp.stack(present).sum(), 1)
        if word:
            # word problems: also accuracy per position and balanced accuracy over the second half
            half = jnp.arange(tgt.shape[1]) > tgt.shape[1] // 2
            rec2 = [(hit * (tgt == k) * msk * half).sum() / jnp.maximum(((tgt == k) * msk * half).sum(), 1) for k in range(C)]
            pr2 = [(((tgt == k) * msk * half).sum() > 0).astype(jnp.float32) for k in range(C)]
            bal2 = (jnp.stack(rec2) * jnp.stack(pr2)).sum() / jnp.maximum(jnp.stack(pr2).sum(), 1)
            return tokacc, ok.all(-1).mean(), bal, (hit * msk).sum(0) / jnp.maximum(msk.sum(0), 1), bal2
        return tokacc, ok.all(-1).mean(), bal

    evaluate = eval_with(cfg)

    erng = np.random.default_rng(cfg.evaloffset + cfg.seed)
    etok, etgt, emsk = gen(erng, cfg.evalset)
    base = float((etgt.reshape(etgt.shape[0], etgt.shape[1], -1).max(-1) * emsk).sum() / emsk.sum())
    t0 = time.time(); curve = []
    for it in range(cfg.steps):
        if cfg.trainset:
            b = it % per_epoch
            if b == 0:
                order = rng.permutation(cfg.trainset)
            idx = order[b * cfg.batch:(b + 1) * cfg.batch]
            tok, tgt, msk = dtok[idx], dtgt[idx], dmsk[idx]
        else:
            tok, tgt, msk = gen(rng, cfg.batch)
        params, st, l = step(params, st, tok, tgt, msk)
        if (it + 1) % max(1, cfg.steps // 10) == 0:
            ta, sa, ba = evaluate(params, etok, etgt, emsk)[:3]
            curve.append((it + 1, round(float(l), 4), round(float(ta), 4), round(float(sa), 4), round(float(ba), 4)))
    res = evaluate(params, etok, etgt, emsk)
    ta, sa, ba = res[:3]
    tag = cfg.model + (f"_p{cfg.plen}" if cfg.model in ("chain", "path", "branch", "branchg", "dpa", "dspa", "dspa2", "dspa3", "dspa4", "dspa5") else "") + (f"_K{cfg.order}_r{cfg.rank}" + ("_raw" if cfg.rawphase else "") + ("_shared" if cfg.readout == "shared" else "") + ("_fnorm" if cfg.fnorm else "") if cfg.model == "newton" else "") + (f"_K{cfg.order}_r{cfg.rank}" + ("_bose" if cfg.stats == "bose" else "") + ("" if cfg.vpath else "_novp") + ("" if cfg.qzero else "_qrand") + (f"_v{cfg.vac:g}" if cfg.vac != 0.2 else "") + (f"_c{cfg.cap:g}" if cfg.cap else "") + (f"_w{cfg.hops}" if cfg.hops else "") if cfg.model == "sfa" else "") + ("_lin" if cfg.nocubic else "") + (f"_r{cfg.recur}" if cfg.recur > 1 else "")
    if cfg.model == "holo":
        tag = f"holo_d{cfg.gdim}_h{cfg.heads}_K{cfg.order}" + ("_fnorm" if cfg.fnorm else "") + ("_causal" if cfg.causal else "") + ("_prefix" if cfg.prefix else "")
        if cfg.order == 0:
            tag = f"orth_d{cfg.gdim}_h{cfg.heads}"
        if cfg.hh:
            tag = tag.replace("orth", "hhrnn").replace("holo", "holohh") + f"_nh{cfg.hh}" + (f"_b{cfg.beta0:g}" if cfg.beta0 else "")
    if cfg.model == "orthornn":
        tag = f"orthornn_d{cfg.gdim}_h{cfg.heads}"
    if cfg.model == "deltaprod":
        tag = f"deltaprod_nh{cfg.nh}" + (f"_b{cfg.beta0:g}" if cfg.beta0 else "")
    if cfg.model == "sa" and cfg.causal:
        tag = "sa_causal"
    if cfg.embed != "table":
        tag += f"_{cfg.embed}"
    out = {"task": cfg.task, "model": tag, "layers": cfg.layers, "seed": cfg.seed, "d": cfg.d, "pos": cfg.pos,
           "params": int(nparams), "tok_acc": round(float(ta), 4), "seq_acc": round(float(sa), 4),
           "bal_acc": round(float(ba), 4), "backend": jax.default_backend(),
           "pos_rate": round(base, 3), "secs": round(time.time() - t0, 1), "steps": cfg.steps, "curve": curve}
    if cfg.task.startswith("mix"):
        # score each arity on its own held-out tables
        out["per_k"] = {}
        for k in [int(c) for c in cfg.task[3:]]:
            ktok, ktgt, kmsk = task_mix(np.random.default_rng(20_000 + cfg.seed + k), 2048, 9, 4, fixed=k)
            out["per_k"][str(k)] = round(float(evaluate(params, ktok, ktgt, kmsk)[2]), 4)
    if cfg.task == "kd4":
        # score the clean and the most distracted tables separately
        out["per_d"] = {}
        for dd in (0, 3, 7):
            dtok, dtgt, dmsk = task_keydist(np.random.default_rng(30_000 + cfg.seed + dd), 2048, 9, 7, 4, fixed=dd)
            out["per_d"][str(dd)] = round(float(evaluate(params, dtok, dtgt, dmsk)[2]), 4)
    if cfg.model == "holo" and cfg.task.startswith("g3_") and not cfg.pos and cfg.layers == 1:
        out["rep"] = holo_rep_content(cfg, params)
    if cfg.task[:3] in ("wp_", "ws_"):
        out["per_pos"] = [round(float(v), 4) for v in np.asarray(res[3])[1:]]
        out["bal_late"] = round(float(res[4]), 4)
        if cfg.model == "holo" and not cfg.pos:
            # the first layer reads only the embedding, so its rotations are a function of the element
            out["rep"] = holo_rep_content(cfg, params)
        if cfg.evallen:
            # length generalisation: the same task at another length, positions beyond training included
            lt = cfg.task.rsplit("_", 1)[0] + f"_{cfg.evallen}"
            ltok, ltgt, lmsk = TASKS[lt][0](np.random.default_rng(40_000 + cfg.seed), 1024)
            lres = evaluate(params, ltok, ltgt, lmsk)
            out["len"] = {"task": lt, "tok_acc": round(float(lres[0]), 4), "seq_acc": round(float(lres[1]), 4),
                          "bal_acc": round(float(lres[2]), 4), "bal_late": round(float(lres[4]), 4),
                          "per_pos": [round(float(v), 4) for v in np.asarray(lres[3])[1:]]}
    if cfg.model in ("newton", "sfa") and cfg.diag and cfg.layers == 1 and cfg.recur == 1 and not multilabel:
        out["diag"] = newton_diagnostics(cfg, params, etok, etgt, emsk, eval_with)
    out["config"] = vars(cfg)
    out["final_checkpoint"] = True
    out["versions"] = {"jax": jax.__version__, "numpy": np.__version__, "optax": optax.__version__}
    if cfg.task == "chase":
        def depth_scores(tx, ty, mx):
            pred = np.asarray(jax.jit(lambda p, t: forward(cfg, p, t))(params, tx)).argmax(-1)[:, -1]
            depths = []
            for row in tx:
                links = {int(a): int(b) for a,b,_ in row[1:-1]}
                q = int(row[-1, 1]); depth = 0
                while links[q] != q:
                    q = links[q]; depth += 1
                depths.append(depth)
            depths = np.array(depths)
            return {str(d): {"n": int((depths == d).sum()), "accuracy": float((pred[depths==d] == ty[depths==d,-1]).mean())}
                    for d in sorted(set(depths.tolist()))}
        out["by_depth"] = depth_scores(etok, etgt, emsk)
        out["long_by_depth"] = depth_scores(*task_chase(np.random.default_rng(40000 + cfg.seed), 2048, maxlinks=8))
    if cfg.save_final:
        dest = pathlib.Path(os.environ["HOA_ARTIFACT_DIR"])
        with (dest / "final.pkl").open("wb") as fh:
            pickle.dump({"config": vars(cfg), "params": jax.device_get(params)}, fh)
        out["checkpoint_sha256"] = __import__("hashlib").sha256((dest / "final.pkl").read_bytes()).hexdigest()
        pred = np.asarray(jax.jit(lambda p, t: forward(cfg, p, t))(params, etok)).argmax(-1)
        np.savez_compressed(dest / "predictions.npz", prediction=pred, target=etgt, mask=emsk)
    print(json.dumps(out))

def holo_rep_content(cfg, params):
    """Which irreps of the task's group the trained layer's rotations hold (groups.rep_content)."""
    G = groups.get({"g3_z12": "Z12", "g3_a4": "A4", "g3_z60": "Z60", "g3_a5": "A5"}.get(cfg.task) or WORD_GROUP[cfg.task.split("_")[1]])
    L = params["layers"][0]
    x = ln(L["ln1"], params["emb"][0][: G.n])                       # every element, no positions
    U = holo_mats(cfg, L["attn"], x).transpose(1, 0, 2, 3)           # [H, |G|, d, d]
    out = groups.rep_content(G, U)
    out["reachable_mass"] = round(groups.plancherel(G, groups.reachable(G, cfg.gdim)), 3)
    return out


def newton_diagnostics(cfg, params, etok, etgt, emsk, eval_with):
    """
    What a trained one-layer Newton attention uses:
      contrib  - each arity's share of the layer's output norm on scored tokens;
      knockout - balanced accuracy with each arity's read-out switched off (the causal test);
      chars    - how close each learned channel is to a character of Z_3^F: for every channel the
                 best resultant |sum_t g_t exp(i(theta_t - 2pi/3 <c, x_t>))| / sum_t g_t over all c,
                 so 1 means an exact character up to a constant phase, across every position.
    """
    import copy
    K, r, H = cfg.order, cfg.rank, cfg.heads
    L0 = params["layers"][0]
    sub = slice(0, 512)
    tok, msk = etok[sub], emsk[sub]
    x = embed(cfg, params, tok)
    h = x if cfg.rawphase else ln(L0["ln1"], x)
    c2 = copy.copy(cfg); c2.want_parts = True
    parts = attention(c2, L0["attn"], h)
    norms = jnp.sqrt((parts ** 2).sum(-1))
    m = jnp.asarray(msk)[..., None]
    contrib = (norms * m).sum((0, 1)) / m.sum()
    contrib = contrib / contrib.sum()
    knock = []
    for k in range(K):
        ck = copy.copy(cfg); ck.knock = k
        knock.append(round(float(eval_with(ck)(params, etok, etgt, emsk)[2]), 4))
    res = {"contrib": [round(float(v), 4) for v in contrib], "knockout": knock}
    cards = np.asarray(tok)[np.asarray(msk) > 0]
    if cards.size and cards.max() < 3 and not cfg.task.startswith("k"):
        th = np.asarray(cfg.phase * lin(L0["attn"]["th"], h))[np.asarray(msk) > 0]
        g = np.asarray(jax.nn.sigmoid(lin(L0["attn"]["g"], h)))[np.asarray(msk) > 0]
        F = cards.shape[-1]
        cs = np.array([[(v // 3 ** a) % 3 for a in range(F)] for v in range(3 ** F)])
        psi = 2 * np.pi / 3 * ((cards @ cs.T) % 3)
        num = (g * np.exp(1j * th)).T @ np.exp(-1j * psi)
        R = np.abs(num) / g.sum(0)[:, None]
        best, Rb = R.argmax(1), R.max(1)
        negate = lambda v: int(sum(((3 - cs[v][a]) % 3) * 3 ** a for a in range(F)))
        good = [int(v) for v, rb in zip(best, Rb) if rb > 0.9 and v != 0]
        res["chars"] = {
            "channels": int(len(Rb)),
            "frac_exact": round(float((Rb > 0.9).mean()), 4),
            "distinct": len({min(v, negate(v)) for v in good}),
            "of": (3 ** F - 1) // 2,
            "quantiles": [round(float(q), 3) for q in np.quantile(Rb, [0.1, 0.25, 0.5, 0.75, 0.9])],
            "hist": np.histogram(Rb, bins=10, range=(0, 1))[0].tolist(),
        }
        # the four most character-like channels: every card's mean phase on each, for phase wheels
        top = np.argsort(-Rb)[:4]
        ids = (cards * (3 ** np.arange(F))).sum(-1)
        ex = []
        for ch in top:
            z = np.zeros(3 ** F, complex)
            np.add.at(z, ids, g[:, ch] * np.exp(1j * th[:, ch]))
            seen = np.bincount(ids, minlength=3 ** F) > 0
            ex.append({"char": [int(v) for v in cs[best[ch]]], "R": round(float(Rb[ch]), 4),
                       "phase": [round(float(np.angle(z[v])), 3) if seen[v] else None for v in range(3 ** F)]})
        res["chars"]["examples"] = ex
    return res

if __name__ == "__main__":
    main()
