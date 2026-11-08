"""Move prediction on Lichess puzzles: Transformers with or without cubic (2-simplicial) layers.

Board as 65 tokens: 64 squares (piece and square embeddings, side to move always white) and one token for castling
rights and en passant. Pre-norm blocks; each block's mixer is softmax attention ('sa') or cubic attention ('cubic'):
A_ijk = sum_c q_ic k1_jc k2_kc / sqrt(dh) over ordered pairs j != k, Y_i = sum_jk softmax_jk(A) v1_j * v2_k.
Move head: logits(f, t) = (h_f Wf) . (h_t Wt) / sqrt(d) over legal moves (as Chessformer's source-destination head).
Scored on first-move and every-move accuracy on held-out puzzles, by theme and rating.
python chess_train.py '<json>' prints one JSON line.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
DEST = Path(os.environ.get('LAB_DEST', '.'))
LINE = ('pin', 'skewer', 'xRayAttack', 'discoveredAttack')


def mats(key, **shapes):
    keys = iter(jax.random.split(key, len(shapes) + 1))
    return {k: jax.random.normal(next(keys), s) * (s[0] ** -.5) for k, s in shapes.items()}


def init(key, cfg):
    d, L = cfg['d'], len(cfg['layers'])
    ks = jax.random.split(key, L + 2)
    p = mats(ks[0], piece=(13, d), square=(64, d), castle=(4, d), ep=(9, d), wf=(d, d), wt=(d, d))
    p['extra'] = j.zeros(d)
    p['g'] = j.ones(d)
    p['blocks'] = []
    for i, kind in enumerate(cfg['layers']):
        b = mats(ks[i + 1], f1=(d, 4 * d), f2=(4 * d, d)) | dict(g1=j.ones(d), g2=j.ones(d))
        b |= mats(jax.random.fold_in(ks[i + 1], 7), qkv=(d, 3 * d), o=(d, d)) if kind == 'sa' else mats(jax.random.fold_in(ks[i + 1], 7), q=(d, d), k1=(d, d), k2=(d, d), v1=(d, d), v2=(d, d), o=(d, d))
        p['blocks'].append(b)
    return p


def norm(x, g):
    return (x - x.mean(-1, keepdims=True)) * jax.lax.rsqrt(x.var(-1, keepdims=True) + 1e-6) * g


def mixer(kind, b, x, H):
    B, n, d = x.shape
    dh = d // H
    sp = lambda t: t.reshape(B, n, H, dh)
    if kind == 'sa':
        q, k, v = (sp(t) for t in j.split(x @ b['qkv'], 3, -1))
        a = jax.nn.softmax(j.einsum('bihc,bjhc->bhij', q, k) / dh ** .5, -1)
        y = j.einsum('bhij,bjhc->bihc', a, v)
    else:
        q, k1, k2, v1, v2 = (sp(x @ b[w]) for w in ('q', 'k1', 'k2', 'v1', 'v2'))
        s = j.einsum('bihc,bjhc,bkhc->bhijk', q, k1, k2) / dh ** .5
        s = j.where(j.eye(n, dtype=bool)[None, None, None], -1e9, s)
        a = jax.nn.softmax(s.reshape(B, H, n, n * n), -1).reshape(B, H, n, n, n)
        y = j.einsum('bhijk,bjhc,bkhc->bihc', a, v1, v2)
    return y.reshape(B, n, d) @ b['o']


def forward(p, cfg, sq, extra):
    sq, extra = sq.astype(j.int32), extra.astype(j.int32)
    x = p['piece'][sq] + p['square'][None]
    e = extra[:, :4].astype(j.float32) @ p['castle'] + p['ep'][extra[:, 4]] + p['extra']
    h = j.concatenate([x, e[:, None]], 1)
    for kind, b in zip(cfg['layers'], p['blocks']):
        block = lambda h, b=b, kind=kind: h + mixer(kind, b, norm(h, b['g1']), cfg['H'])
        h = jax.checkpoint(block)(h) if cfg.get('remat') and kind == 'cubic' else block(h)
        h = h + jax.nn.gelu(norm(h, b['g2']) @ b['f1']) @ b['f2']
    h = norm(h, p['g'])[:, :64]
    return j.einsum('bfc,btc->bft', h @ p['wf'], h @ p['wt']).reshape(h.shape[0], 4096) / h.shape[-1] ** .5


def masked(logits, mask):
    return j.where(j.unpackbits(mask, axis=-1).astype(bool), logits, -1e9)


def flops_per_position(cfg):
    d, n, H = cfg['d'], 65, cfg['H']
    f = 0
    for kind in cfg['layers']:
        f += n * 8 * d * d  # MLP
        f += n * 4 * d * d + 2 * n * n * d if kind == 'sa' else n * 6 * d * d + 2 * n ** 3 * d
    return 2 * (f + 64 * 2 * d * d + 64 * 64 * d)  # multiply-adds to FLOPs, plus the move head


def main(job):
    src = Path('/vol/results') / job['data']
    load = lambda split, k: np.load(src / f'{split}_{k}.npy')
    cfg = job['model']
    N = job.get('train_examples')
    tr = {k: j.array(load('train', k)[:N]) for k in ('sq', 'extra', 'mask', 'label')}
    n = int(tr['label'].shape[0])
    tests = {s: {k: load(s, k) for k in ('sq', 'extra', 'mask', 'label', 'ply', 'puzzle')} for s in ('val', 'test')}
    meta = {s: json.loads((src / f'{s}_meta.json').read_text()) for s in ('val', 'test')}
    steps, B = job['steps'], job['batch']
    sched = optax.warmup_cosine_decay_schedule(0., job['lr'], min(1000, steps // 10), steps, job['lr'] * .05)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=job.get('wd', .01)))
    rows = []
    for seed in job['seeds']:
        p = init(jax.random.PRNGKey(seed), cfg)
        st = tx.init(p)
        params = int(sum(w.size for w in jax.tree.leaves(p)))

        @jax.jit
        def step(p, st, key, tr):  # data passed in, not closed over, so it is not baked into the compiled graph
            ix = jax.random.randint(key, (B,), 0, n)
            def loss(p):
                lg = masked(forward(p, cfg, tr['sq'][ix], tr['extra'][ix]), tr['mask'][ix])
                return optax.softmax_cross_entropy_with_integer_labels(lg, tr['label'][ix].astype(j.int32)).mean()
            v, g = jax.value_and_grad(loss)(p)
            u, st = tx.update(g, st, p)
            return optax.apply_updates(p, u), st, v
        said_fn = jax.jit(lambda p, sq, ex, mk: masked(forward(p, cfg, sq, ex), mk).argmax(-1))

        def evaluate(p, s):
            t = tests[s]
            said = np.concatenate([np.asarray(said_fn(p, t['sq'][i:i + 1024], t['extra'][i:i + 1024], t['mask'][i:i + 1024])) for i in range(0, len(t['label']), 1024)])
            ok = said == t['label'].astype(np.int64)
            first = t['ply'] == 0
            per = {}
            for pz, good in zip(t['puzzle'], ok):
                per[int(pz)] = per.get(int(pz), True) and bool(good)
            out = dict(first=float(ok[first].mean()), every=float(ok.mean()), solved=float(np.mean(list(per.values()))))
            if s == 'test':
                themes = {}
                for pz, good in zip(t['puzzle'][first], ok[first]):
                    for th in meta[s][str(pz)]['themes']:
                        themes.setdefault(th, []).append(good)
                    line = any(th in LINE for th in meta[s][str(pz)]['themes'])
                    themes.setdefault('_line' if line else '_other', []).append(good)
                    r = meta[s][str(pz)]['rating']
                    themes.setdefault(f'_rating_{min(r // 500, 5) * 500}', []).append(good)
                out['by_theme'] = {k: [float(np.mean(v)), len(v)] for k, v in themes.items()}
            return out
        key = jax.random.PRNGKey(seed + 100)
        t0, curve = time.time(), []
        for it in range(steps):
            key, sub = jax.random.split(key)
            p, st, v = step(p, st, sub, tr)
            if (it + 1) % job.get('log', 1000) == 0:
                curve.append((it + 1, float(v), evaluate(p, 'val')['first']))
                print(json.dumps(dict(seed=seed, step=it + 1, loss=round(float(v), 4), val=round(curve[-1][2], 4), sec=round(time.time() - t0))), file=sys.stderr, flush=True)
        secs = time.time() - t0
        rows.append(dict(cfg=cfg, seed=seed, params=params, flops=flops_per_position(cfg), steps=steps, batch=B, lr=job['lr'], train_examples=n, seconds=secs,
                         curve=curve, val=evaluate(p, 'val'), test=evaluate(p, 'test')))
        (DEST / 'rows.json').write_text(json.dumps(rows))
        if job.get('save'):
            flat, _ = jax.tree_util.tree_flatten_with_path(p)
            np.savez_compressed(DEST / f'params_s{seed}.npz', **{jax.tree_util.keystr(k): np.asarray(w) for k, w in flat})
    return dict(rows=[{k: v for k, v in r.items() if k != 'test'} | dict(test={k: v for k, v in r['test'].items() if k != 'by_theme'}) for r in rows])


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
