"""SET with a restricted set of cards in hand: does a layer learn the rule or a table of its cards in hand?

(a) Data per question: fits with m cards in hand (the first m of a fixed shuffle ORDER) at several data sizes.
(b) Rule or table: every fit with m <= 27 is also tested on the 54 cards in hand it never held (HELD), and every fit
    on deals with 16 table cards instead of 8.
(c) A fit may run in stages [[m, T, N, updates], ...], parameters carried from one stage to the next.
(d) Saved weights of the join and small-Transformer fits, for the mechanism analysis (mechanism.py).

The table size T is a parameter; the paper's deals have T = 8.
"""
import json, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
import bench2
from bench2 import ARMS, init, apply

ORDER = np.random.default_rng(7).permutation(81)
HELD = ORDER[27:]  # never a query in any training set with m <= 27


def deals(seed, count, S, T=8):
    rng = np.random.default_rng(seed)
    xs, ys, need = [], [], [count // 2, count - count // 2]
    while sum(need):
        B = max(256, sum(need) * 2)
        q = S[rng.integers(0, len(S), B)]
        r = rng.random((B, 81))
        r[np.arange(B), q] = 2.
        ids = np.concatenate([np.argsort(r, axis=1)[:, :T], q[:, None]], 1)
        attrs = np.stack([(ids // 3 ** a) % 3 for a in range(4)], -1)
        x = np.zeros((B, T + 2, 64), np.float32)
        x[:, 0, 63] = 1
        x[:, T + 1, 62] = 1
        for t in range(T + 1):
            for a in range(4):
                x[np.arange(B), t + 1, 3 * a + attrs[:, t, a]] = 1
        witness = ((attrs[:, T, None, None, :] + attrs[:, :T, None, :] + attrs[:, None, :T, :]) % 3 == 0).all(-1) & ~np.eye(T, dtype=bool)[None]
        y = witness.any((1, 2)).astype(int)
        for label in (0, 1):
            ix = np.flatnonzero(y == label)[:need[label]]
            xs.append(x[ix]); ys.append(y[ix]); need[label] -= len(ix)
    x, y = np.concatenate(xs), np.concatenate(ys).astype(np.int32)
    order = rng.permutation(count)
    return x[order], y[order]


def trainer(a, updates, N):
    """bench2's trainer, with parameters carried per rate so stages can follow one another."""
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())

    def train(p, x, y, lr, seed):
        def step(state, t):
            p, st = state
            ix = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed), t), (bench2.BATCH,), 0, N)
            loss = lambda p: optax.softmax_cross_entropy_with_integer_labels(apply(p, x[ix], a), y[ix]).mean()
            value, g = jax.value_and_grad(loss)(p)
            u, st = tx.update(g, st, p)
            return (jax.tree.map(lambda w, c: w - lr * c, p, u), st), value
        (p, _), losses = jax.lax.scan(step, (p, tx.init(p)), j.arange(updates))
        return p, losses[-200:].mean()
    return jax.jit(jax.vmap(jax.vmap(train, (0, 0, 0, None, 0)), (0, None, None, 0, None)))


def query_set(m):
    return ORDER[:m]


def fit(spec):
    """spec: arm, lrs, stages [[m, T, N, updates], ...], save. Returns one row with every evaluation."""
    a = ARMS[spec["arm"]]
    lrs = spec["lrs"]
    acc = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, None, None)), (0, None, None)))
    params = jax.tree.map(lambda *w: j.stack(w), *[init(s, a) for s in (0, 1, 2)])
    params = jax.tree.map(lambda w: j.broadcast_to(w, (len(lrs), *w.shape)), params)
    t0, losses, train_acc = time.time(), [], []
    for k, (m, T, N, updates) in enumerate(spec["stages"]):
        S = query_set(m)
        data = [deals(10000 + s + 1000 * k, N, S, T) for s in (0, 1, 2)]
        x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
        params, loss = trainer(a, updates, N)(params, x, y, j.array(lrs), j.arange(3))
        jax.block_until_ready(loss)
        losses.append(np.asarray(loss).tolist())
        sub = lambda t: t[:, :1024]
        train_acc.append(np.asarray(jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, 0, 0)), (0, None, None)))(params, sub(x), sub(y))).tolist())
        del x, y, data
    m, T = spec["stages"][-1][:2]
    S = query_set(m)
    evals = dict(val=(9200, S, 8), in8=(20000, S, 8), in16=(20002, S, 16))
    if m <= 27:
        evals |= dict(out8=(20001, HELD, 8), out16=(20003, HELD, 16))
    if m < 81:
        evals |= dict(full8=(20004, ORDER, 8))
    res = {}
    for name, (seed, S_, T_) in evals.items():
        tx, ty = deals(seed, 1024, S_, T_)
        res[name] = np.asarray(acc(params, j.array(tx), j.array(ty))).tolist()
    row = dict(arm=spec["arm"], tag=spec.get("tag", ""), stages=spec["stages"], lrs=lrs, params=int(sum(w[0, 0].size for w in jax.tree.leaves(params))),
               seconds=time.time() - t0, final_loss=losses, train=train_acc, **res)
    return row, params


def main():
    job = json.loads(sys.argv[1])
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(job["fits"]):
        row, params = fit(spec)
        with (out / "fits.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(dict(arm=row["arm"], tag=row["tag"], stages=row["stages"], sec=round(row["seconds"], 1))), flush=True)
        if spec.get("save"):
            flat, _ = jax.tree_util.tree_flatten_with_path(params)
            np.savez_compressed(out / f"params_{i}_{spec['arm']}.npz", **{jax.tree_util.keystr(k): np.asarray(w) for k, w in flat})
        del params
        jax.clear_caches()


if __name__ == "__main__":
    main()
