"""Newton with a linear read-out (`newton_bare`, the base of bench_nset.py's arms), oriented Newton and Feynman arms,
on bench2's input map and head."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
import bench2
from bench2 import D, R, BATCH
from operators import pair_mask, moments, contract
from tasks import generate
import hoa_polybench as legacy

ARMS = {
    "on_mass": dict(kind="on", mass=True), "on_h4": dict(kind="on", heads=4), "on_h4_mass": dict(kind="on", heads=4, mass=True), "on_exp_mass": dict(kind="on", mass=True, pos="exp"),
    "newton_bare": dict(kind="newton", head="linear"), "newton_skip": dict(kind="newton", head="skip"),
    "fey_r1": dict(kind="feynman", rounds=1), "fey_r1_mass": dict(kind="feynman", rounds=1, mass=True), "fey_r3": dict(kind="feynman", rounds=3), "fey_lin": dict(kind="feynman", rounds=2, nofeed=True),
}


def init(seed, a):
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 64))
    mat = lambda r, c: jax.random.normal(next(keys), (r, c)) / r ** .5
    p = dict(input=mat(64, D), m1=mat(D, 2 * D), mb=j.zeros(2 * D), m2=mat(2 * D, 2), bias=j.zeros(2), head=mat(D, 2))
    if a["kind"] == "newton":
        p["legacy"] = dict(th=dict(w=mat(D, R), b=j.zeros(R)), g=dict(w=mat(D, R), b=j.zeros(R)), o=dict(w=mat(6 * R, D), b=j.zeros(D)))
        return p
    p |= {k: mat(D, R) for k in ("q", "k1", "k2", "v1", "v2")} | dict(o=mat(R, D))
    if a["kind"] == "feynman":
        p |= dict(L=mat(R, R), Rm=mat(R, R))
    if a.get("mass"):
        p["w"] = j.zeros((a.get("heads", 1), D))
    return p


def feynman(q, u, w, v, L, Rm, mask, rounds, nofeed):
    n = q.shape[1]
    b = j.where(mask.any(-1), j.einsum("bic,bjc->bij", q, u), -j.inf)
    c = j.where(mask, j.einsum("bic,bjc,bkc->bijk", q, u, w), -j.inf)
    H, F = v, j.zeros(v.shape[:2], v.dtype)
    for _ in range(rounds):
        f = 0. * F if nofeed else F
        energies = j.concatenate([j.zeros((*F.shape, 1), v.dtype), b + f[:, None, :], (c + f[:, None, :, None] + f[:, None, None, :]).reshape(*F.shape, -1)], -1)
        F = jax.scipy.special.logsumexp(energies, -1)
        p = jax.nn.softmax(energies, -1)
        H = p[..., :1] * v + j.einsum("bij,bjv->biv", p[..., 1:n + 1], H) + contract(p[..., n + 1:].reshape(q.shape[0], n, n, n), H @ L, H @ Rm)
    return H, F


def apply(p, x, a):
    h = j.tanh(x @ p["input"])
    b, n = h.shape[:2]
    if a["kind"] == "newton":
        c = argparse.Namespace(model="newton", heads=1, d=D, rank=R, order=3, phase=2., bosfix=1, fnorm=1, nfloor=1e-6, readout="full")
        y = legacy.attention(c, p["legacy"], h)
    elif a["kind"] == "on":
        pos = j.exp if a.get("pos") == "exp" else jax.nn.softplus
        H = a.get("heads", 1)
        split = lambda t: t.reshape(b, n, H, R // H)
        q, k1, k2, v1, v2 = (split(h @ p[k]) for k in ("q", "k1", "k2", "v1", "v2"))
        ys, zs = [], []
        for i in range(H):
            yi, zi = moments(pos(q[:, :, i]), pos(k1[:, :, i]), pos(k2[:, :, i]), v1[:, :, i], v2[:, :, i], bos=True)
            ys.append(yi); zs.append(zi)
        y = j.concatenate(ys, -1) @ p["o"]
        if a.get("mass"):
            count = max((n - 2) * (n - 3), 1)
            y = y + j.log(j.maximum(j.stack(zs, -1), 1e-9) / count) @ p["w"]
    else:
        m = pair_mask(n, bos=True)
        Y, F = feynman(h @ p["q"], h @ p["k1"], h @ p["k2"], h @ p["v1"], p["L"], p["Rm"], m, a["rounds"], a.get("nofeed", False))
        y = Y @ p["o"]
        if a.get("mass"):
            y = y + (F - j.log(1. + n + n * n))[..., None] @ p["w"]
    z = j.tanh(h[:, -1] + y[:, -1])
    linear = z @ p["head"] + p["bias"]
    if a.get("head") == "linear":
        return linear
    mlp = jax.nn.gelu(z @ p["m1"] + p["mb"]) @ p["m2"]
    return linear + mlp if a.get("head") == "skip" else mlp + p["bias"]


def trainer(a, updates, n):
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())

    def train(p, x, y, lr, seed):
        def step(state, t):
            p, st = state
            ix = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed), t), (BATCH,), 0, n)
            value, g = jax.value_and_grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(apply(p, x[ix], a), y[ix]).mean())(p)
            u, st = tx.update(g, st, p)
            return (jax.tree.map(lambda w, c: w - lr * c, p, u), st), value
        (p, _), losses = jax.lax.scan(step, (p, tx.init(p)), j.arange(updates))
        return p, losses[-200:].mean()
    return jax.jit(jax.vmap(jax.vmap(train, (0, 0, 0, None, 0)), (None, None, None, 0, None)))


def main():
    job = json.loads(sys.argv[1])
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    n, lrs = 32768, job["lrs"]
    for name in job["arms"]:
        a = ARMS[name]
        train = trainer(a, 12000, n)
        acc = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, 0, 0)), (0, None, None)))
        params = jax.tree.map(lambda *w: j.stack(w), *[init(s, a) for s in (0, 1, 2)])
        count = sum(w[0].size for w in jax.tree.leaves(params))
        for task in job["tasks"]:
            data = [generate(task, 10000 + s, n) for s in (0, 1, 2)]
            x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
            vx, vy = generate(task, 9200, 1024)
            tx, ty = generate(task, 20000, 1024)
            three = lambda t: j.broadcast_to(j.array(t), (3, *t.shape))
            t0 = time.time()
            fitted, loss = train(params, x, y, j.array(lrs), j.arange(3))
            jax.block_until_ready(loss)
            row = dict(arm=name, task=task, lrs=lrs, params=int(count), seconds=time.time() - t0, final_loss=np.asarray(loss).tolist(), train=np.asarray(acc(fitted, x[:, :1024], y[:, :1024])).tolist(), val=np.asarray(acc(fitted, three(vx), three(vy))).tolist(), test=np.asarray(acc(fitted, three(tx), three(ty))).tolist())
            with (out / f"{name}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(dict(arm=name, task=task, sec=round(row["seconds"], 1))), flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
