"""Learning curves: bench2's arms at several training-set sizes (cells [arm, deals, updates]), plus Transformers with
dropout and weight decay."""
import json, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
import bench2
from bench2 import ARMS, init, layernorm
from tasks import generate

ARMS = ARMS | {"sa4_reg": dict(ARMS["sa4"], reg=True), "sa_big_reg": dict(ARMS["sa_big"], reg=True)}


def dropout(t, key, rate=.1):
    keep = jax.random.bernoulli(key, 1 - rate, t.shape)
    return j.where(keep, t / (1 - rate), 0.)


def transformer_reg(p, x, key):
    h = j.tanh(x @ p["input"])
    b, n, d = h.shape
    mask = (j.arange(n)[None] > 0) & ~j.eye(n, dtype=bool)
    for block in p["blocks"]:
        key, k1, k2 = (jax.random.split(key, 3) if key is not None else (None, None, None))
        q, k, v = j.split(layernorm(h, block["g1"]) @ block["qkv"], 3, -1)
        split = lambda t: t.reshape(b, n, 4, d // 4).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        s = q @ k.swapaxes(-1, -2) / (d / 4) ** .5
        y = (jax.nn.softmax(j.where(mask, s, -j.inf), -1) @ v).transpose(0, 2, 1, 3).reshape(b, n, d) @ block["o"]
        h = h + (dropout(y, k1) if key is not None else y)
        f = jax.nn.gelu(layernorm(h, block["g2"]) @ block["f1"] + block["fb"]) @ block["f2"]
        h = h + (dropout(f, k2) if key is not None else f)
    return layernorm(h, p["g"])[:, -1] @ p["head"] + p["bias"]


def apply(p, x, a, key=None):
    return transformer_reg(p, x, key) if a.get("reg") else bench2.apply(p, x, a)


def trainer(a, updates, n):
    parts = [optax.clip_by_global_norm(1.), optax.scale_by_adam()]
    if a.get("reg"):
        parts.append(optax.add_decayed_weights(.1))
    tx = optax.chain(*parts)

    def train(p, x, y, lr, seed):
        def step(state, t):
            p, st = state
            key = jax.random.fold_in(jax.random.PRNGKey(seed), t)
            ix = jax.random.randint(key, (bench2.BATCH,), 0, n)
            loss = lambda p: optax.softmax_cross_entropy_with_integer_labels(apply(p, x[ix], a, jax.random.fold_in(key, 1)), y[ix]).mean()
            value, g = jax.value_and_grad(loss)(p)
            u, st = tx.update(g, st, p)
            return (jax.tree.map(lambda w, c: w - lr * c, p, u), st), value
        (p, _), losses = jax.lax.scan(step, (p, tx.init(p)), j.arange(updates))
        return p, losses[-200:].mean()
    return jax.jit(jax.vmap(jax.vmap(train, (0, 0, 0, None, 0)), (None, None, None, 0, None)))


def main():
    job = json.loads(sys.argv[1])
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    for name, n, updates in job["cells"]:
        a = ARMS[name]
        lrs = job["sa_lrs"] if a["kind"] == "sa" else job["lrs"]
        acc = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, 0, 0)), (0, None, None)))
        train = trainer(a, updates, n)
        params = jax.tree.map(lambda *w: j.stack(w), *[init(s, a) for s in (0, 1, 2)])
        count = sum(w[0].size for w in jax.tree.leaves(params))
        for task in job["tasks"]:
            data = [generate(task, 10000 + s, n) for s in (0, 1, 2)]
            x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
            del data
            vx, vy = generate(task, 9200, 1024)
            tx, ty = generate(task, 20000, 1024)
            three = lambda t: j.broadcast_to(j.array(t), (3, *t.shape))
            t0 = time.time()
            fitted, loss = train(params, x, y, j.array(lrs), j.arange(3))
            jax.block_until_ready(loss)
            seconds = time.time() - t0
            row = dict(arm=name, task=task, N=n, lrs=lrs, params=int(count), updates=updates, seconds=seconds, final_loss=np.asarray(loss).tolist(),
                       train=np.asarray(acc(fitted, x[:, :1024], y[:, :1024])).tolist(), val=np.asarray(acc(fitted, three(vx), three(vy))).tolist(), test=np.asarray(acc(fitted, three(tx), three(ty))).tolist())
            with (out / f"{name}_{n}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(dict(arm=name, task=task, N=n, sec=round(seconds, 1))), flush=True)
            del x, y, fitted
        jax.clear_caches()


if __name__ == "__main__":
    main()
