"""New branch families for the screen. All outputs start at zero, so every model starts identical to its baseline."""
import math
import jax
import jax.numpy as j
import hoa_lm as lm

R, V = 32, 120


def rank(c):
    return getattr(c, 'br', R)


def init(c, keys, D):
    H = c.branch_heads
    weight = lambda a, b: jax.random.normal(next(keys), (a, b)) / math.sqrt(a)
    b = {'norm': lm.ln_init(D), 'o': j.zeros((H * V, D))}
    res = getattr(c, 'res', 'none')
    if res in ('ls', 'rz'):
        b['o'] = weight(H * V, D)
        if res == 'ls':
            b['gamma'] = .1 * j.ones(D)
        else:
            b['alpha'] = j.zeros(())
    if res == 'gate':
        b['gw'] = weight(D, D)
    if c.arm == 'pstream':
        P = c.dp
        out = dict(norm=lm.ln_init(D), left=weight(D, P), right=weight(D, P), a=weight(P, P), b=weight(P, P), gate=weight(P, P), q=weight(D, P), o=j.zeros((P, D)), znorm=lm.ln_init(P))
        if getattr(c, 'ps_rel', 0):
            out['rel'] = jax.random.normal(next(keys), (c.window, P)) * .02
        return out
    r = rank(c)
    b.update(q=weight(D, H * r), k1=weight(D, H * r), k2=weight(D, H * r), v1=weight(D, H * V), v2=weight(D, H * V))
    if getattr(c, 'bpos', 'none') == 'rope3L':
        n = r // 2
        periods = 2. ** (j.arange(n) * math.log2(2 * c.window + 2) / max(n - 1, 1) + 1)
        b['w1'], b['w2'] = 2 * math.pi / periods, 2 * math.pi / periods[::-1]
    if c.arm == 'topm':
        b.update(sq=weight(D, H * R), sk=weight(D, H * R))
    if getattr(c, 'temp0', 0):
        b['tau'] = j.log(float(c.temp0)) * j.ones((H,))
    if getattr(c, 'tri', 0):
        b['k3'] = weight(D, H * R)
    if getattr(c, 'bpos', 'none') == 'rel':
        W = c.window
        b.update(ra=j.zeros((W, H)), rb=j.zeros((W, H)), rc=j.zeros((2 * W, H)))
    return b


def shifted(t, d):
    """t at position p - d, zeros where that is before the start. Axis 1 is position."""
    return j.where((j.arange(t.shape[1]) >= d)[(None, ...) + (None,) * (t.ndim - 2)], j.roll(t, d, axis=1), 0.)


def window_cubic(c, b, x, stats):
    B, T, D = x.shape
    H = c.branch_heads
    h = lm.ln(b['norm'], x)
    proj = lambda n, w: lm.mm(h, b[n]).reshape(B, T, H, w)
    r = rank(c)
    q, k1, k2, v1, v2 = proj('q', r), proj('k1', r), proj('k2', r), proj('v1', V), proj('v2', V)
    W1, W2 = c.window, getattr(c, 'window2', 0) or c.window
    start = 0 if getattr(c, 'selfdiag', 0) else 1
    i1 = j.arange(T)[:, None] - (start + j.arange(W1))[None]
    i2 = j.arange(T)[:, None] - (start + j.arange(W2))[None]
    ok1, ok2 = i1 >= 0, i2 >= 0
    i1, i2 = j.maximum(i1, 0), j.maximum(i2, 0)
    a, e = k1[:, i1], k2[:, i2]
    pos = getattr(c, 'bpos', 'none')
    if pos in ('rope3', 'rope3L'):
        # Momentum conserving: the query carries w1 + w2, so only i-j and i-k matter. Here i is at offset zero.
        n = r // 2
        if pos == 'rope3L':
            w1, w2 = b['w1'], b['w2']
        else:
            periods = 2. ** (j.arange(n) * math.log2(2 * W1 + 2) / max(n - 1, 1) + 1)
            w1, w2 = 2 * math.pi / periods, 2 * math.pi / periods[::-1]
        d1, d2 = start + j.arange(W1), start + j.arange(W2)
        cx = lambda t: t[..., 0::2] + 1j * t[..., 1::2]
        ac = cx(a) * j.exp(1j * d1[:, None] * w1)[None, None, :, None, :]
        ec = cx(e) * j.exp(1j * d2[:, None] * w2)[None, None, :, None, :]
        s = j.real(j.einsum('bthc,btjhc,btkhc->bthjk', cx(q), ac, ec)) / math.sqrt(n)
    elif pos == 'det':
        # Roy et al.: a determinant of 3-vectors, rotated about one axis by position. Invariant to a shared rotation, antisymmetric in the partners.
        m = r // 3
        chunk = lambda t, ax: [t[..., i::3][..., :m] for i in range(3)]
        periods = 2. ** (j.arange(m) * math.log2(2 * W1 + 2) / max(m - 1, 1) + 1)
        w = 2 * math.pi / periods
        def spin(parts, d):
            if getattr(c, 'det_nospin', 0):
                return parts
            cos, sin = j.cos(d[:, None] * w)[None, None, :, None, :], j.sin(d[:, None] * w)[None, None, :, None, :]
            return [parts[0] * cos - parts[1] * sin, parts[0] * sin + parts[1] * cos, parts[2]]
        Q, A, E = chunk(q, -1), spin(chunk(a, -1), start + j.arange(W1)), spin(chunk(e, -1), start + j.arange(W2))
        s = 0.
        for (x_, y_, z_), sign in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
            s = s + sign * j.einsum('bthc,btjhc,btkhc->bthjk', Q[x_], A[y_], E[z_])
        s = s / math.sqrt(6 * m)
    else:
        if getattr(c, 'sbias', 0):
            q, a, e = q + 1, a + 1, e + 1
        s = j.einsum('bthc,btjhc,btkhc->bthjk', q, a, e) / math.sqrt(r)
    if pos == 'rel':
        d = j.arange(W1)[:, None] - j.arange(W2)[None] + W1
        s = s + b['ra'].T[None, None, :, :, None] + b['rb'].T[None, None, :, None, :] + b['rc'][d].transpose(2, 0, 1)[None, None]
    if 'tau' in b:
        s = s * j.exp(b['tau'])[None, None, :, None, None]
    mask = ok1[:, :, None] & ok2[:, None, :]
    if not getattr(c, 'selfdiag', 0):
        mask = mask & jnp_offdiag(W1, W2)[None]
    if getattr(c, 'ordered', 0):
        mask = mask & (j.arange(W1)[:, None] > j.arange(W2)[None])[None]
    s = j.where(mask[None, :, None], s, -j.inf)
    s = j.where(mask.any((-1, -2))[None, :, None, None, None], s, 0)
    p = jax.nn.softmax(s.reshape(*s.shape[:3], -1), -1).reshape(s.shape)
    p = j.where(mask[None, :, None], p, 0)
    u, w = v1[:, i1], v2[:, i2]
    if getattr(c, 'vbias', 0):
        u, w = u + 1, w + 1
    y = j.einsum('bthjk,btjhv,btkhv->bthv', p, u, w)
    if getattr(c, 'vbias', 0):
        y = y - p.sum((-1, -2))[..., None]
    if stats is not None:
        fac = j.einsum('bthj,btjhv->bthv', p.sum(-1), u) * j.einsum('bthk,btkhv->bthv', p.sum(-2), w)
        stats['factorised_share'] = (fac ** 2).sum() / ((y ** 2).sum() + 1e-12)
        flat = p[:, W1:].reshape(B, T - W1, H, -1)
        stats['effective_pairs'] = j.exp(-(flat * j.log(flat + 1e-12)).sum(-1)).mean()
        stats['top_pair'] = flat.max(-1).mean()
    if getattr(c, 'res', 'none') == 'post':
        centred = y - y.mean(-1, keepdims=True)
        y = centred / j.sqrt((centred ** 2).mean(-1, keepdims=True) + 1e-5)
    return lm.mm(y.reshape(B, T, -1), b['o'])


def residual(c, b, x, y):
    """The form in which a branch's output joins the token stream."""
    res = getattr(c, 'res', 'none')
    if res == 'ls':
        return y * b['gamma']
    if res == 'rz':
        return y * b['alpha']
    if res == 'gate':
        return jax.nn.sigmoid(lm.mm(lm.ln(b['norm'], x), b['gw'])) * y
    if res == 'mul':
        return x * j.tanh(y)
    return y


def branch_rope(t, base=10000.):
    # t: B,T,H,C with even C.
    T, C = t.shape[1], t.shape[3]
    freq = base ** (-j.arange(0, C, 2) / C)
    angle = j.arange(T)[None, :, None, None] * freq[None, None, None, :]
    cos, sin = j.cos(angle), j.sin(angle)
    a, b = t[..., 0::2], t[..., 1::2]
    return j.stack([a * cos - b * sin, a * sin + b * cos], -1).reshape(t.shape)


def jnp_offdiag(W1, W2):
    return j.arange(W1)[:, None] != j.arange(W2)[None]


def band(c, b, x, stats):
    """Two-factor keys from neighbouring tokens, so reach is the whole past at softmax cost per offset."""
    B, T, D = x.shape
    H = c.branch_heads
    h = lm.ln(b['norm'], x)
    proj = lambda n, w: lm.mm(h, b[n]).reshape(B, T, H, w)
    q, k1, k2, v1, v2 = proj('q', R), proj('k1', R), proj('k2', R), proj('v1', V), proj('v2', V)
    kind = c.arm
    copy = getattr(c, 'copy_next', 0)
    tril = j.tril(j.ones((T, T), bool), -1 if copy else 0)
    if getattr(c, 'bandwin', 0):
        tril = tril & ~j.tril(j.ones((T, T), bool), -c.bandwin)
    scores, values, masks = [], [], []
    for d in c.offsets:
        k2d, v2d = shifted(k2, d), shifted(v2, d)
        reach = j.arange(T) >= d
        if kind == 'qband':
            qq = q * shifted(k2, d)
            key = k1
            mask = tril & reach[:, None]
            val = v1[:, None] * v2d[:, :, None]
            s = j.einsum('bthc,bjhc->bthj', qq, key) / math.sqrt(R)
            scores.append(s); masks.append(mask); values.append(val)
            continue
        qq = q * shifted(q, 1) if kind == 'band2' else q
        key = {'prod': k1 * k2d, 'sum': k1 + k2d, 'single': k1}[getattr(c, 'keymode', 'prod')]
        if getattr(c, 'tri', 0):
            key = key * shifted(proj('k3', R), 2 * d)
        mask = tril & (reach[None, :] if getattr(c, 'keymode', 'prod') != 'single' else True)
        if getattr(c, 'bpos', 'none') == 'rope':
            qq, key = branch_rope(qq), branch_rope(key)
        s = j.einsum('bthc,bjhc->bthj', qq, key) / math.sqrt(R)
        # With copy, read the token after the matched one. Only j <= i - 1 is allowed, so this stays causal.
        val = j.roll(v1, -1, axis=1) * v2 if copy else {'prod': v1 * v2d, 'sum': v1 + v2d, 'single': v1}[getattr(c, 'valmode', 'prod')]
        scores.append(s); masks.append(mask); values.append(val)
    s = j.stack(scores, -1)
    mask = j.stack(masks, -1)
    s = j.where(mask[None, :, None], s, -j.inf)
    s = j.where(mask.any((-1, -2))[None, :, None, None, None], s, 0)
    p = jax.nn.softmax(s.reshape(B, T, H, -1), -1).reshape(s.shape)
    p = j.where(mask[None, :, None], p, 0)
    if kind == 'qband':
        y = sum(j.einsum('bthj,btjhv->bthv', p[..., n], values[n]) for n in range(len(c.offsets)))
    else:
        y = sum(j.einsum('bthj,bjhv->bthv', p[..., n], values[n]) for n in range(len(c.offsets)))
    if stats is not None:
        flat = p[:, T // 2:].reshape(B, T - T // 2, H, -1)
        stats['effective_pairs'] = j.exp(-(flat * j.log(flat + 1e-12)).sum(-1)).mean()
        stats['top_pair'] = flat.max(-1).mean()
    return lm.mm(y.reshape(B, T, -1), b['o'])


def local(c, b, x, stats):
    """A product or sum of this token and the previous one, with no attention."""
    h = lm.ln(b['norm'], x)
    v1, v2 = lm.mm(h, b['v1']), shifted(lm.mm(h, b['v2']), 1)
    return lm.mm(v1 * v2 if getattr(c, 'localmode', 'prod') == 'prod' else v1 + v2, b['o'])


def topm(c, b, x, stats):
    """Choose m partners by a pair score over all past, then form all pairs among them. Selection is not differentiated."""
    B, T, D = x.shape
    H, m = c.branch_heads, c.m
    h = lm.ln(b['norm'], x)
    proj = lambda n, w: lm.mm(h, b[n]).reshape(B, T, H, w)
    q, k1, k2, v1, v2 = proj('q', R), proj('k1', R), proj('k2', R), proj('v1', V), proj('v2', V)
    pair = j.einsum('bthc,bjhc->bhtj', proj('sq', R), proj('sk', R)) / math.sqrt(R)
    pair = j.where(j.tril(j.ones((T, T), bool), -1), pair, -1e9)
    top_score, idx = jax.lax.top_k(pair, m)
    good = top_score > -1e8
    pick = lambda t: jax.vmap(jax.vmap(lambda tb, ib: tb[ib], (1, 0), 0), (0, 0), 0)(t, idx)  # B,H,T,m,C
    a, e, u, w = pick(k1), pick(k2), pick(v1), pick(v2)
    qq = q.transpose(0, 2, 1, 3)
    s = j.einsum('bhtc,bhtjc,bhtkc->bhtjk', qq, a, e) / math.sqrt(R) + top_score[..., None] + top_score[..., None, :]
    mask = good[..., None] & good[..., None, :] & ~j.eye(m, dtype=bool)
    s = j.where(mask, s, -j.inf)
    s = j.where(mask.any((-1, -2), keepdims=True), s, 0)
    p = j.where(mask, jax.nn.softmax(s.reshape(B, H, T, -1), -1).reshape(s.shape), 0)
    y = j.einsum('bhtjk,bhtjv,bhtkv->bhtv', p, u, w).transpose(0, 2, 1, 3)
    if stats is not None:
        flat = p[:, :, T // 2:].reshape(B, H, T - T // 2, -1)
        stats['effective_pairs'] = j.exp(-(flat * j.log(flat + 1e-12)).sum(-1)).mean()
        stats['top_pair'] = flat.max(-1).mean()
        fac = j.einsum('bhtj,bhtjv->bhtv', p.sum(-1), u) * j.einsum('bhtk,bhtkv->bhtv', p.sum(-2), w)
        stats['factorised_share'] = (fac ** 2).sum() / ((y ** 2).sum() + 1e-12)
    return lm.mm(y.reshape(B, T, -1), b['o'])


def pstream_step(c, b, x, z):
    """One block of the local pair stream. z[t, w] is the pair (t, t-1-w). Returns the token update and the new z."""
    B, T, D = x.shape
    W, P = c.window, c.dp
    h = lm.ln(b['norm'], x)
    left, right = lm.mm(h, b['left']), lm.mm(h, b['right'])
    dyadic = getattr(c, 'ps_dy', 0)
    # Dyadic form: row w holds the pair (t, t - 2**w), so W rows reach 2**(W-1) tokens back.
    new = left[:, :, None] * j.stack([shifted(right, 2 ** w if dyadic else w + 1) for w in range(W)], 2)
    if 'rel' in b:
        new = new + b['rel'][None, None]
    z = new if z is None else z + new
    n = lm.ln(b['znorm'], z)
    a, e, g = lm.mm(n, b['a']), lm.mm(n, b['b']), jax.nn.sigmoid(lm.mm(n, b['gate']))
    total = j.zeros_like(z)
    if dyadic:
        # Doubling only: t -> t - 2**(w-1) -> t - 2**w, both legs from row w-1.
        for w in range(1, W):
            total = total.at[:, :, w].add(a[:, :, w - 1] * shifted(e[:, :, w - 1], 2 ** (w - 1)))
        z = z + g * total
    else:
        # Path t -> t-s -> t-d, for d = s + r: the first leg is z[t, s-1], the second is z[t-s, r-1].
        for s in range(1, W):
            second = shifted(e, s)[:, :, :W - s]
            total = total.at[:, :, s:].add(a[:, :, s - 1:s] * second)
        z = z + g * total / math.sqrt(W)
    q = lm.mm(h, b['q'])
    w = jax.nn.softmax(j.einsum('btc,btwc->btw', q, n) / math.sqrt(P), -1)
    return lm.mm(j.einsum('btw,btwc->btc', w, z), b['o']), z
