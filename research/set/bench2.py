"""The join (one layer of 2-simplicial attention, `c1`), Transformers, Newton and Feynman arms at matched size on the
bench tasks, SET among them. One cell of N training deals per task (32,768 by default), three seeds per rate; the
rate is picked on validation accuracy summed over seeds, and test accuracy is reported per seed."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
from operators import pair_mask, moments, feynman
from tasks import generate
import hoa_polybench as legacy

D, R, N, BATCH, UPDATES = 96, 64, 32768, 128, 12000
ARMS = {
    "c1": dict(kind="cubic"), "c_h4": dict(kind="cubic", heads=4), "c_h8": dict(kind="cubic", heads=8), "c_temp": dict(kind="cubic", temp=True),
    "c_warm": dict(kind="cubic", warm=True), "c_mass": dict(kind="cubic", mass=True), "c_scale": dict(kind="cubic", scale=True), "c_conn": dict(kind="cubic", conn=True),
    "tie_k": dict(kind="cubic", tie_k=True), "tie_v": dict(kind="cubic", tie_v=True), "tie_both": dict(kind="cubic", tie_k=True, tie_v=True), "sym": dict(kind="cubic", sym=True),
    "sa1": dict(kind="sa", L=1, d=84), "sa2": dict(kind="sa", L=2, d=60), "sa4": dict(kind="sa", L=4, d=44), "sa8": dict(kind="sa", L=8, d=32),
    "sa4_long": dict(kind="sa", L=4, d=44, updates=4 * UPDATES), "sa_big": dict(kind="sa", L=4, d=128),
    "newton_mlp": dict(kind="newton"), "on_mlp": dict(kind="on"), "fey_mlp": dict(kind="feynman"), "fey_tied": dict(kind="feynman", tie=True), "fey_mass": dict(kind="feynman", mass=True),
}


def init(seed, a):
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 256))
    mat = lambda r, c: jax.random.normal(next(keys), (r, c)) / r ** .5
    if a["kind"] == "sa":
        d = a["d"]
        p = dict(input=mat(64, d), head=mat(d, 2), bias=j.zeros(2), blocks=[dict(qkv=mat(d, 3 * d), o=mat(d, d), f1=mat(d, 2 * d), fb=j.zeros(2 * d), f2=mat(2 * d, d), g1=j.ones(d), g2=j.ones(d)) for _ in range(a["L"])], g=j.ones(d))
        return p
    p = dict(input=mat(64, D), m1=mat(D, 2 * D), mb=j.zeros(2 * D), m2=mat(2 * D, 2), bias=j.zeros(2))
    if a["kind"] == "newton":
        p["legacy"] = dict(th=dict(w=mat(D, R), b=j.zeros(R)), g=dict(w=mat(D, R), b=j.zeros(R)), o=dict(w=mat(6 * R, D), b=j.zeros(D)))
        return p
    p |= {k: mat(D, R) for k in ("q", "k1", "k2", "v1", "v2")} | dict(o=mat(R, D))
    if a["kind"] == "feynman":
        p |= dict(L=mat(R, R), Rm=mat(R, R))
    if a.get("mass"):
        p["w"] = j.zeros(D)
    if a.get("temp"):
        p["tau"] = j.log(4.) * j.ones(())
    return p


def layernorm(x, g):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6) * g


def transformer(p, x, a):
    h = j.tanh(x @ p["input"])
    b, n, d = h.shape
    heads = 4
    mask = (j.arange(n)[None] > 0) & ~j.eye(n, dtype=bool)
    for block in p["blocks"]:
        q, k, v = j.split(layernorm(h, block["g1"]) @ block["qkv"], 3, -1)
        split = lambda t: t.reshape(b, n, heads, d // heads).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        s = q @ k.swapaxes(-1, -2) / (d / heads) ** .5
        y = (jax.nn.softmax(j.where(mask, s, -j.inf), -1) @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        h = h + y @ block["o"]
        h = h + jax.nn.gelu(layernorm(h, block["g2"]) @ block["f1"] + block["fb"]) @ block["f2"]
    return layernorm(h, p["g"])[:, -1] @ p["head"] + p["bias"]


def cubic(p, h, a):
    b, n = h.shape[:2]
    m = pair_mask(n, bos=True)
    H = a.get("heads", 1)
    split = lambda t: t.reshape(b, n, H, R // H)
    q, k1, k2 = split(h @ p["q"]), split(h @ p["k1"]), split(h @ p["k1" if a.get("tie_k") else "k2"])
    v1, v2 = split(h @ p["v1"]), split(h @ p["v1" if a.get("tie_v") else "v2"])
    s = j.einsum("bihc,bjhc,bkhc->bhijk", q, k1, k2) / (R / H) ** .5
    if a.get("temp"):
        s = s * j.exp(p["tau"])
    valid = m.any((-2, -1))
    masked = j.where(valid[None, None, :, None, None], j.where(m, s, -j.inf), 0.)
    flat = masked.reshape(b, H, n, -1)
    prob = j.where(m, jax.nn.softmax(flat, -1).reshape(masked.shape), 0.)
    if a.get("sym"):
        prob = (prob + prob.swapaxes(-1, -2)) / 2
    y = j.einsum("bhijk,bjhc,bkhc->bihc", prob, v1, v2)
    if a.get("conn"):
        y = y - j.einsum("bhij,bjhc->bihc", prob.sum(4), v1) * j.einsum("bhik,bkhc->bihc", prob.sum(3), v2)
    y = y.reshape(b, n, R)
    if a.get("scale"):
        y = y / j.sqrt((jax.lax.stop_gradient(y) ** 2).mean() + 1e-8)
    y = y @ p["o"]
    if a.get("mass"):
        mass = jax.scipy.special.logsumexp(flat[:, 0], -1) - j.log(j.maximum(m.sum((-2, -1)), 1))
        y = y + j.where(valid[None], mass, 0.)[..., None] * p["w"]
    return y


def apply(p, x, a):
    if a["kind"] == "sa":
        return transformer(p, x, a)
    h = j.tanh(x @ p["input"])
    if a["kind"] == "cubic":
        y = cubic(p, h, a)
    elif a["kind"] == "newton":
        c = argparse.Namespace(model="newton", heads=1, d=D, rank=R, order=3, phase=2., bosfix=1, fnorm=1, nfloor=1e-6, readout="full")
        y = legacy.attention(c, p["legacy"], h)
    elif a["kind"] == "on":
        pos = jax.nn.softplus
        y = moments(pos(h @ p["q"]), pos(h @ p["k1"]), pos(h @ p["k2"]), h @ p["v1"], h @ p["v2"], bos=True)[0] @ p["o"]
    else:
        m = pair_mask(h.shape[1], bos=True)
        H, F = feynman(h @ p["q"], h @ p["k1"], h @ p["k2"], h @ p["v1"], p["L"], p["L"] if a.get("tie") else p["Rm"], m)
        y = H @ p["o"]
        if a.get("mass"):
            y = y + (F - F.mean())[..., None] * p["w"]
    z = j.tanh(h[:, -1] + y[:, -1])
    return jax.nn.gelu(z @ p["m1"] + p["mb"]) @ p["m2"] + p["bias"]


def trainer(a, updates):
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())

    def train(p, x, y, lr, seed):
        def step(state, t):
            p, st = state
            ix = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed), t), (BATCH,), 0, N)
            loss = lambda p: optax.softmax_cross_entropy_with_integer_labels(apply(p, x[ix], a), y[ix]).mean()
            value, g = jax.value_and_grad(loss)(p)
            u, st = tx.update(g, st, p)
            rate = lr * j.minimum(1., (t + 1) / (.1 * updates)) if a.get("warm") else lr
            return (jax.tree.map(lambda w, c: w - rate * c, p, u), st), value
        (p, _), losses = jax.lax.scan(step, (p, tx.init(p)), j.arange(updates))
        return p, losses[-200:].mean()
    return jax.jit(jax.vmap(jax.vmap(train, (0, 0, 0, None, 0)), (None, None, None, 0, None)))


def main():
    job = json.loads(sys.argv[1])
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    lrs = j.array(job["lrs"])
    global N
    N = job.get("N", N)
    updates_default = job.get("updates", UPDATES)
    for name in job["arms"]:
        a = ARMS[name]
        updates = a.get("updates", updates_default)
        train = trainer(a, updates)
        acc = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, 0, 0)), (0, None, None)))
        params = jax.tree.map(lambda *w: j.stack(w), *[init(s, a) for s in (0, 1, 2)])
        count = sum(w[0].size for w in jax.tree.leaves(params))
        for task in job["tasks"]:
            data = [generate(task, 10000 + s, N) for s in (0, 1, 2)]
            x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
            vx, vy = generate(task, 9200, 1024)
            tx, ty = generate(task, 20000, 1024)
            three = lambda t: j.broadcast_to(j.array(t), (3, *t.shape))
            t0 = time.time()
            fitted, loss = train(params, x, y, lrs, j.arange(3))
            jax.block_until_ready(loss)
            seconds = time.time() - t0
            row = dict(arm=name, task=task, lrs=job["lrs"], params=int(count), updates=updates, seconds=seconds, final_loss=np.asarray(loss).tolist(),
                       train=np.asarray(acc(fitted, x[:, :1024], y[:, :1024])).tolist(), val=np.asarray(acc(fitted, three(vx), three(vy))).tolist(), test=np.asarray(acc(fitted, three(tx), three(ty))).tolist())
            with (out / f"{name}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(dict(arm=name, task=task, sec=round(seconds, 1))), flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
