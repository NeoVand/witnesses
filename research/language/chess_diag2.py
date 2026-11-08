"""Diagnostics of the hybrid block (chess_train2.py).

For saved hybrid weights, on validation positions, computed in float32 on CPU:
- accuracy in float32 and with the training path's bfloat16 rounding emulated (inputs and output of the pair attention
  cast to bfloat16), to check that the connected subtraction is not lost to rounding;
- per hybrid block: the drop in first-move accuracy when only the cubic branch, only the attention branch, or both are
  switched off; branch output size against the residual stream;
- for the cubic branch: effective number of pairs, top weight, the connected part's share of the raw term
  ||sum p v1 v2 - (sum p v1)(sum p v2)|| / ||sum p v1 v2||, the score's interaction share over the 64 x 64 partner
  grid relative to random scores (interaction_relative), and the relative bfloat16 error of the connected output;
- on positions with a pin: rank of (pinner, king) among the 4,096 ordered pairs for the pinned square, best head.
python chess_diag2.py <params.npz> <data dir> <n positions> ['{"connected": false, "center": true}']
"""
import json, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chess_train2 as ct
from chess_diag import pins

CFG = dict(d=256, H=8, layers=['sa', 'sa', 'hyb', 'sa', 'sa', 'hyb', 'sa', 'sa'], connected=True, diag=True)


def load(path, cfg):
    p = ct.init(jax.random.PRNGKey(0), cfg)
    P = dict(np.load(path))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    return jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])


def bf(t):
    return t.astype(j.bfloat16).astype(j.float32)


def cubic_branch(c, x, H, rounding, connected=True, center=False):
    """The cubic branch, explicitly, with its pair weights. rounding emulates the bf16 fused path."""
    B, n, d = x.shape
    dh = d // H
    sp = lambda t: t.reshape(B, t.shape[1], H, dh)
    q = sp(x @ c['q'])
    k1, k2, v1, v2 = (sp(x[:, :64] @ c[w]) for w in ('k1', 'k2', 'v1', 'v2'))
    if center:
        k1, k2 = k1 - k1.mean(1, keepdims=True), k2 - k2.mean(1, keepdims=True)
    K = (k1[:, :, None] * k2[:, None, :]).reshape(B, 4096, H, dh)
    V12 = (v1[:, :, None] * v2[:, None, :]).reshape(B, 4096, H, dh)
    V1 = j.broadcast_to(v1[:, :, None], (B, 64, 64, H, dh)).reshape(B, 4096, H, dh)
    V2 = j.broadcast_to(v2[:, None, :], (B, 64, 64, H, dh)).reshape(B, 4096, H, dh)
    if rounding:
        q, K, V12, V1, V2 = map(bf, (q, K, V12, V1, V2))
    sc = j.einsum('bihc,bphc->bhip', q, K) / dh ** .5
    a = jax.nn.softmax(sc, -1)  # [B, H, n, 4096]
    raw, m1, m2 = (j.einsum('bhip,bphc->bihc', a, V) for V in (V12, V1, V2))
    if rounding:
        raw, m1, m2 = map(bf, (raw, m1, m2))
    y = raw - m1 * m2 if connected else raw
    return y.reshape(B, n, d) @ c['o'], a, raw, y, sc


def interaction(s):
    """Score interaction share over the 64 x 64 partner grid, relative to a random score's (63^2 / 4095)."""
    r = s - s.mean(-1, keepdims=True) - s.mean(-2, keepdims=True) + s.mean((-1, -2), keepdims=True)
    t = s - s.mean((-1, -2), keepdims=True)
    return float((r ** 2).sum() / (t ** 2).sum()) / (63 ** 2 / 4095)


def run(p, cfg, sq, extra, off=None, rounding=False, keep=False):
    """off: dict block -> 'cubic' | 'sa' | 'both' switched off."""
    off = off or {}
    x = p['piece'][sq.astype(j.int32)] + p['square'][None]
    ex = extra.astype(j.int32)
    e = ex[:, :4].astype(j.float32) @ p['castle'] + p['ep'][ex[:, 4]] + p['extra']
    h = j.concatenate([x, e[:, None]], 1)
    stats, maps = {}, {}
    H = cfg['H']
    for li, (kind, b) in enumerate(zip(cfg['layers'], p['blocks'])):
        xn = ct.norm(h, b['g1'])
        B, n, d = xn.shape
        dh = d // H
        sp = lambda t: t.reshape(B, n, H, dh)
        qq, kk, vv = (sp(t) for t in j.split(xn @ b['qkv'], 3, -1))
        att = jax.nn.softmax(j.einsum('bihc,bjhc->bhij', qq, kk) / dh ** .5, -1)
        sa = j.einsum('bhij,bjhc->bihc', att, vv).reshape(B, n, d) @ b['o']
        out = sa * (off.get(li) not in ('sa', 'both'))
        if kind == 'hyb':
            c = {k[2:]: v for k, v in b.items() if k.startswith('c_')}
            cb, a, raw, y, sc = cubic_branch(c, xn, H, rounding, cfg.get('connected', False), cfg.get('center', False))
            out = out + cb * (off.get(li) not in ('cubic', 'both'))
            rn = j.linalg.norm(h, axis=-1).mean()
            ent = -(a * j.log(a + 1e-12)).sum(-1)
            stats[li] = dict(effective_pairs=float(j.exp(ent).mean()), top=float(a.max(-1).mean()),
                             cubic_over_resid=float(j.linalg.norm(cb, axis=-1).mean() / rn), sa_over_resid=float(j.linalg.norm(sa, axis=-1).mean() / rn),
                             connected_over_raw=float(j.linalg.norm(y) / j.linalg.norm(raw)), interaction_relative=interaction(sc.reshape(B, H, n, 64, 64)))
            if keep:
                maps[li] = (a, y)
        h = h + out
        h = h + jax.nn.gelu(ct.norm(h, b['g2']) @ b['f1']) @ b['f2']
    h = ct.norm(h, p['g'])[:, :64]
    logits = j.einsum('bfc,btc->bft', h @ p['wf'], h @ p['wt']).reshape(h.shape[0], 4096) / h.shape[-1] ** .5
    return logits, stats, maps


def main(path, data, n, opts):
    cfg = dict(CFG, **opts)
    p = load(path, cfg)
    D = Path(data)
    sq, ex, mk, lab = (np.load(D / f'val_{k}.npy')[:n] for k in ('sq', 'extra', 'mask', 'label'))
    bs = 64
    batches = range(0, n, bs)
    hyb = [i for i, k in enumerate(cfg['layers']) if k == 'hyb']
    res = dict(path=path, n=n)

    def accuracy(off=None, rounding=False):
        good, st = 0, []
        for i in batches:
            lg, s, _ = run(p, cfg, j.array(sq[i:i + bs]), j.array(ex[i:i + bs]), off, rounding)
            good += int((np.asarray(ct.masked(lg, j.array(mk[i:i + bs])).argmax(-1)) == lab[i:i + bs]).sum())
            st.append(s)
        return good / n, st
    base, st = accuracy()
    res['accuracy_f32'] = base
    res['accuracy_bf16_emulated'] = accuracy(rounding=True)[0]
    res['hybrid_stats'] = {str(l): {k: float(np.mean([s[l][k] for s in st])) for k in st[0][l]} for l in hyb}
    res['drop'] = {f'{l}_{what}': base - accuracy({l: what})[0] for l in hyb for what in ('cubic', 'sa', 'both')}
    res['drop']['all_cubic'] = base - accuracy({l: 'cubic' for l in hyb})[0]
    sa_blocks = [i for i, k in enumerate(cfg['layers']) if k == 'sa']
    res['drop_sa_blocks'] = {str(l): base - accuracy({l: 'sa'})[0] for l in sa_blocks}
    # rounding error of the connected output, and pins
    found = pins(sq[:n])
    err, hits = {l: [] for l in hyb}, {l: [] for l in hyb}
    for i in batches:
        _, _, m32 = run(p, cfg, j.array(sq[i:i + bs]), j.array(ex[i:i + bs]), keep=True)
        _, _, m16 = run(p, cfg, j.array(sq[i:i + bs]), j.array(ex[i:i + bs]), rounding=True, keep=True)
        for l in hyb:
            err[l].append(float(j.linalg.norm(m16[l][1] - m32[l][1]) / j.linalg.norm(m32[l][1])))
        for (b_i, s, att, king) in [f for f in found if i <= f[0] < i + bs]:
            for l in hyb:
                w = np.asarray(m32[l][0][b_i - i, :, s]).reshape(-1, 64, 64)  # [H, j, k]
                best = max(w[:, att, king].max(), w[:, king, att].max())
                hw = w.max(0)
                hits[l].append(dict(weight=float(best), rank=int((hw > best).sum())))
    res['bf16_relative_error_connected'] = {str(l): float(np.mean(v)) for l, v in err.items()}
    res['pins'] = dict(count=len(found), chance=2 / 4096, by_layer={str(l): dict(mean_weight=float(np.mean([h['weight'] for h in v])), median_rank=float(np.median([h['rank'] for h in v])),
                                                                                 top1=float(np.mean([h['rank'] == 0 for h in v]))) for l, v in hits.items() if v})
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
