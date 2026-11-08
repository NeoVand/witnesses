"""Diagnostics of trained chess models with cubic layers (chess_train.py).

For saved weights: per layer, how spread the attention is (effective number of pairs or squares), how large the mixer's
output is against the residual stream, and the drop in first-move accuracy when the mixer is switched off. On positions
with a pin, whether the cubic layer at the pinned piece attends to the pinner and the king together.
python chess_diag.py <params.npz> <layers comma list> <data dir> <n positions>
"""
import json, math, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chess_train as ct


def load(path, cfg):
    p = ct.init(jax.random.PRNGKey(0), cfg)
    P = dict(np.load(path))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    return jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])


def run(p, cfg, sq, extra, off=None):
    """forward with per-layer statistics; off = index of a block whose mixer is switched off."""
    x = p['piece'][sq.astype(j.int32)] + p['square'][None]
    ex = extra.astype(j.int32)
    e = ex[:, :4].astype(j.float32) @ p['castle'] + p['ep'][ex[:, 4]] + p['extra']
    h = j.concatenate([x, e[:, None]], 1)
    stats, maps = [], {}
    for li, (kind, b) in enumerate(zip(cfg['layers'], p['blocks'])):
        xn = ct.norm(h, b['g1'])
        B, n, d = xn.shape
        H = cfg['H']; dh = d // H
        sp = lambda t: t.reshape(B, n, H, dh)
        if kind == 'sa':
            q, k, v = (sp(t) for t in j.split(xn @ b['qkv'], 3, -1))
            a = jax.nn.softmax(j.einsum('bihc,bjhc->bhij', q, k) / dh ** .5, -1)
            y = j.einsum('bhij,bjhc->bihc', a, v)
            flat = a
        else:
            q, k1, k2, v1, v2 = (sp(xn @ b[w]) for w in ('q', 'k1', 'k2', 'v1', 'v2'))
            s = j.einsum('bihc,bjhc,bkhc->bhijk', q, k1, k2) / dh ** .5
            s = j.where(j.eye(n, dtype=bool)[None, None, None], -1e9, s)
            a = jax.nn.softmax(s.reshape(B, H, n, n * n), -1)
            y = j.einsum('bhijk,bjhc,bkhc->bihc', a.reshape(B, H, n, n, n), v1, v2)
            flat = a
            maps[li] = a.reshape(B, H, n, n, n)
            # Share of the output a product of two separate averages would give (the factorised part).
            pj, pk = a.reshape(B, H, n, n, n).sum(-1), a.reshape(B, H, n, n, n).sum(-2)
            fac = j.einsum('bhij,bjhc->bihc', pj, v1) * j.einsum('bhik,bkhc->bihc', pk, v2)
        out = y.reshape(B, n, d) @ b['o']
        ent = -(flat * j.log(flat + 1e-12)).sum(-1)
        st = dict(layer=li, kind=kind, effective=float(j.exp(ent).mean()), top=float(flat.max(-1).mean()),
                  out_over_resid=float(j.linalg.norm(out, axis=-1).mean() / j.linalg.norm(h, axis=-1).mean()))
        if kind == 'cubic':
            st['factorised_share'] = float(((fac - y) ** 2).sum() / (y ** 2).sum())
        stats.append(st)
        h = h + (0 if off == li else out)
        h = h + jax.nn.gelu(ct.norm(h, b['g2']) @ b['f1']) @ b['f2']
    h = ct.norm(h, p['g'])[:, :64]
    logits = j.einsum('bfc,btc->bft', h @ p['wf'], h @ p['wt']).reshape(h.shape[0], 4096) / h.shape[-1] ** .5
    return logits, stats, maps


def pins(sq):
    """(pinned square, pinner square, king square) for every absolute pin, both colours, from the encoded board."""
    import chess
    out = []
    for b_i, row in enumerate(sq):
        board = chess.Board(None)
        for s, c in enumerate(row):
            if c:
                board.set_piece_at(s, chess.Piece((c - 1) % 6 + 1, chess.WHITE if c <= 6 else chess.BLACK))
        for color in (chess.WHITE, chess.BLACK):
            king = board.king(color)
            if king is None:
                continue
            for s in chess.SquareSet(board.occupied_co[color]):
                if s != king and board.is_pinned(color, s):
                    ray = chess.SquareSet(chess.ray(king, s))
                    attackers = [a for a in chess.SquareSet(board.occupied_co[not color]) if a in ray and board.piece_type_at(a) in (chess.BISHOP, chess.ROOK, chess.QUEEN)
                                 and chess.SquareSet(chess.between(king, a)) & board.occupied == chess.SquareSet([s])]
                    for a in attackers:
                        out.append((b_i, s, a, king))
    return out


def main(path, layers, data, n):
    cfg = dict(d=256, H=8, layers=layers)
    p = load(path, cfg)
    D = Path(data)
    sq, ex, mk, lab = (np.load(D / f'val_{k}.npy')[:n] for k in ('sq', 'extra', 'mask', 'label'))
    res = dict(path=path, layers=layers, n=n)
    batches = range(0, n, 128)

    def accuracy(off=None):
        good, stats = 0, []
        for i in batches:
            lg, st, _ = run(p, cfg, j.array(sq[i:i + 128]), j.array(ex[i:i + 128]), off)
            said = np.asarray(ct.masked(lg, j.array(mk[i:i + 128])).argmax(-1))
            good += int((said == lab[i:i + 128]).sum())
            stats.append(st)
        return good / n, stats
    base, stats = accuracy()
    res['accuracy'] = base
    res['layers_stats'] = [{k: float(np.mean([s[l][k] for s in stats])) if not isinstance(stats[0][l][k], str) else stats[0][l][k] for k in stats[0][l]} for l in range(len(layers))]
    res['ablation_drop'] = [base - accuracy(off=l)[0] for l in range(len(layers))]
    if 'cubic' in layers:
        found = pins(sq[:n])
        hits = {l: [] for l, k in enumerate(layers) if k == 'cubic'}
        chance = []
        for i in batches:
            _, _, maps = run(p, cfg, j.array(sq[i:i + 128]), j.array(ex[i:i + 128]))
            for (b_i, s, a, king) in [f for f in found if i <= f[0] < i + 128]:
                for l, m in maps.items():
                    w = np.asarray(m[b_i - i, :, s])  # [H, j, k] from the pinned square
                    pair = w[:, a, king] + w[:, king, a]
                    hits[l].append(dict(best_head_pair=float(pair.max()), rank_of_pair=int((w.max(0) > max(w.max(0)[a, king], w.max(0)[king, a])).sum())))
                chance.append(2 / (65 * 64))
        res['pins'] = dict(count=len(found), chance_weight=float(np.mean(chance)) if chance else None,
                           by_layer={str(l): dict(mean_best_head_weight=float(np.mean([h['best_head_pair'] for h in v])), median_rank=float(np.median([h['rank_of_pair'] for h in v])),
                                                  top1=float(np.mean([h['rank_of_pair'] == 0 for h in v]))) for l, v in hits.items() if v})
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(','), sys.argv[3], int(sys.argv[4]))
