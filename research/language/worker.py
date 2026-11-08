"""One language-model training run with an optional gradient trace. python worker.py '<json>' prints one JSON line.

Reads the GPT-2 token stream /vol/fineweb_edu/train.bin and the held-out sets /vol/data-v1/{dev,test}.npz written by
prepare.py; writes per-document test losses to LAB_DEST/test.npz."""
import hashlib, json, math, os, pickle, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
import model
DEST = Path(os.environ['LAB_DEST'])


def tree_norm(t):
    return j.sqrt(sum(j.sum(a.astype(j.float32) ** 2) for a in jax.tree.leaves(t)) + 1e-20)


def tree_dot(a, b):
    return sum(j.sum(x.astype(j.float32) * y.astype(j.float32)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def main(job):
    c = model.config(job['arm'], **job.get('config', {}))
    model.lm.MM = j.bfloat16
    train = np.memmap('/vol/fineweb_edu/train.bin', dtype=np.uint16, mode='r')
    dev = dict(np.load('/vol/data-v1/dev.npz'))
    def decay_mask(p):
        # Optionally exempt the branch's factor matrices from weight decay. A product decays k times over.
        keep = {'q', 'k1', 'k2', 'v1', 'v2', 'left', 'right', 'a', 'b', 'sq', 'sk'}
        def rule(path, _):
            names = [getattr(k, 'key', None) for k in path]
            return not ('branch' in names and keep & set(n for n in names if isinstance(n, str)))
        return jax.tree_util.tree_map_with_path(rule, p)
    decay = optax.add_decayed_weights(.1, mask=decay_mask) if getattr(c, 'wd_exempt', 0) else optax.add_decayed_weights(.1)
    def branch_scale(updates, params=None):
        factor = getattr(c, 'branch_lr', 1.)
        def rule(path, u):
            return u * factor if 'branch' in [getattr(k, 'key', None) for k in path] else u
        return jax.tree_util.tree_map_with_path(rule, updates)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam(), decay, optax.stateless(branch_scale))
    ce = lambda logits, y: optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    @jax.jit
    def step(p, state, x, y, rate):
        loss, g = jax.value_and_grad(lambda p: ce(model.forward(c, p, x), y))(p)
        u, state = tx.update(g, state, p)
        return jax.tree.map(lambda p, u: p - rate * u, p, u), state, loss

    @jax.jit
    def ev(p, x, y, mask):
        return j.sum(optax.softmax_cross_entropy_with_integer_labels(model.forward(c, p, x), y) * mask, -1)

    @jax.jit
    def diag(p, x, y, x2, y2):
        ga = jax.grad(lambda p: ce(model.forward(c, p, x), y))(p)
        gb = jax.grad(lambda p: ce(model.forward(c, p, x2), y2))(p)
        pg, a, b = model.groups(p), model.groups(ga), model.groups(gb)
        out = {k: dict(grad=tree_norm(a[k]), param=tree_norm(pg[k]), snr=tree_dot(a[k], b[k]) / (tree_norm(a[k]) * tree_norm(b[k]))) for k in pg}
        zero = lambda: j.zeros((*x.shape, c.d), j.float32)
        sites = [{'attn': zero(), 'branch': zero(), 'mlp': zero()} for _ in p['layers']]

        def site_loss(sites):
            trace = []
            logits = model.forward(c, p, x, sites, trace)
            return ce(logits, y), trace
        (loss, trace), gs = jax.value_and_grad(site_loss, has_aux=True)(sites)
        layers = []
        for layer, g in zip(trace, gs):
            row = dict(stream=j.sqrt((layer['stream'] ** 2).mean()))
            for name in ('attn', 'branch', 'mlp'):
                if name in layer:
                    y_ = layer[name].astype(j.float32)
                    row[name + '_rms'] = j.sqrt((y_ ** 2).mean())
                    row[name + '_cos'] = -(y_ * g[name]).sum() / (j.sqrt((y_ ** 2).sum()) * j.sqrt((g[name] ** 2).sum()) + 1e-12)
                    row[name + '_nats'] = -(y_ * g[name]).sum()
            row |= {k: v for k, v in layer.get('stats', {}).items()}
            layers.append(row)
        return dict(loss=loss, groups=out, layers=layers)

    def evaluate(p, data):
        out = []
        for start in range(0, len(data['tokens']), 16):
            tok, mask = data['tokens'][start:start + 16], data['mask'][start:start + 16]
            out.extend(np.asarray(ev(p, tok[:, :-1], tok[:, 1:], mask)))
        losses, counts = np.array(out), data['mask'].sum(-1)
        return float(losses.sum() / counts.sum()), losses, counts

    seed, steps, rate = job['seed'], job['steps'], job.get('lr', .001)
    p = model.init(c, seed)
    state = tx.init(p)
    rng = np.random.default_rng(seed)
    diag_rng = np.random.default_rng(seed + 55555)
    warm = min(200, max(1, steps // 10))
    trace_hash = hashlib.sha256()
    begin = time.monotonic()
    active, first, curve, diags = 0., None, [], []
    every = job.get('diag_every', 0)
    grab = lambda g: (lambda ix: (np.stack([train[t:t + c.ctx + 1] for t in ix]).astype(np.int32)))(g.integers(0, len(train) - c.ctx - 1, c.batch))
    for it in range(steps):
        ix = rng.integers(0, len(train) - c.ctx - 1, c.batch)
        trace_hash.update(ix.astype('<i8').tobytes())
        tok = np.stack([train[t:t + c.ctx + 1] for t in ix]).astype(np.int32)
        factor = (it + 1) / warm if it < warm else .1 + .9 * .5 * (1 + math.cos(math.pi * (it - warm) / max(1, steps - warm)))
        if every and (it == 0 or (it + 1) % every == 0 or (it < 1000 and (it + 1) % (every // 5 or 1) == 0)):
            a, b = grab(diag_rng), grab(diag_rng)
            d = jax.tree.map(lambda v: float(v), diag(p, a[:, :-1], a[:, 1:], b[:, :-1], b[:, 1:]))
            d['step'] = it + 1
            before = model.groups(p)
            d['_before'] = True
            diags.append(d)
        now = time.monotonic()
        new_p, state, loss = step(p, state, tok[:, :-1], tok[:, 1:], j.asarray(rate * factor, j.float32))
        loss = float(loss)
        dt = time.monotonic() - now
        if diags and diags[-1].get('_before'):
            after = model.groups(new_p)
            for k in after:
                diags[-1]['groups'][k]['update_ratio'] = float(tree_norm(jax.tree.map(lambda a, b: a - b, after[k], before[k])) / tree_norm(before[k]))
            del diags[-1]['_before']
        p = new_p
        if it == 0:
            first = dt
        else:
            active += dt
        if not math.isfinite(loss):
            return dict(job=job, finite=False, steps=it + 1, loss=loss, diags=diags)
        if (it + 1) % max(1, steps // 8) == 0 or it == 0:
            curve.append(dict(step=it + 1, loss=loss, active=active))
            (DEST / 'progress.json').write_text(json.dumps(curve[-1]))
    loop = time.monotonic() - begin
    dev_loss = evaluate(p, dev)[0]
    test_loss, losses, counts = evaluate(p, dict(np.load('/vol/data-v1/test.npz')))
    np.savez_compressed(DEST / 'test.npz', loss_sum=losses, token_count=counts)
    out = dict(seed=seed, steps=steps, lr=rate, tokens=steps * c.batch * c.ctx, finite=math.isfinite(test_loss), dev_loss=dev_loss, test_loss=test_loss, first_step_seconds=first,
               active_seconds=active, loop_seconds=loop, seconds_per_step=active / max(steps - 1, 1), counts=model.counts(c, p), curve=curve, diags=diags, data_order_sha256=trace_hash.hexdigest())
    taus = [np.asarray(l['branch']['tau']).tolist() for l in p['layers'] if 'branch' in l and 'tau' in l['branch']]
    if taus:
        out['final_tau'] = taus
    if job.get('save'):
        with (DEST / 'params.pkl').open('wb') as f:
            pickle.dump(dict(config=vars(c), params=jax.tree.map(np.asarray, p), job=job), f)
    out['total_seconds'] = time.monotonic() - begin
    return out


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
