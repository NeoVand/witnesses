"""Newton attention on SET.

Learning-curve cells are bench3's loop (same data, seeds, batch, optimizer, updates and evaluation) with bench4's Newton
arms. The rule fit is bench8.fit with bench4's arms in place of bench2's. Every fit also stores newton_chars.analyse for
each rate and seed, trained and untrained, and its weights.

Arms: newton_bare (rank 64), newton_r256 (rank 256), newton_oscale (rank 64 with the read-out step correction
64 / (6 r), the arm of the paper), newton_count (count form for orders 2 and 3); mixed() trains on several table sizes.

job: {"cells": [[arm, N, updates, lrs], ...], "rule": [bench8 spec, ...], "mixed": [spec, ...]}
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j
import bench4, bench8
from bench2 import D
from tasks import generate
import hoa_polybench as legacy
import newton_chars

# Rank 256: bench4's newton_bare at a larger rank. Same key order as bench4.init, so the
# input map and head draw the same weights; the MLP weights the linear head never reads are drawn and dropped, so the
# parameter count is the count in use.
_init4, _apply4 = bench4.init, bench4.apply
ARMS = bench4.ARMS | {"newton_r256": dict(kind="newton", head="linear", rank=256)}
# Read-out step correction: the output map stored as W / c and used as c * W, so the function at initialisation is unchanged and an
# Adam step moves the read-out c times less; c = 64 / (6 r), the join's output fan-in over Newton's.
ARMS["newton_oscale"] = dict(kind="newton", head="linear", rank=64, oscale=64 / 384)


# Count form for orders 2 and 3 (newton_theory/table16.py shows why the mean form fails on larger tables). f_m = phi_q e_m / C(8, m) with the
# constant of the 8-card training table, and no per-deal RMS normaliser on those two blocks; order 1 unchanged; the
# read-out step correction kept.
ARMS["newton_count"] = dict(kind="newton", head="linear", rank=64, oscale=64 / 384, count=8)


def newton_layer(L, h, n0=None, rank=64):
    """hoa_polybench.attention's Newton path (heads 1, order 3, phase 2, bosfix, fnorm, floor 1e-6, readout full),
    written out; with n0 set, blocks m = 1, 2 use the count form."""
    th = 2. * (h @ L["th"]["w"] + L["th"]["b"])
    g = jax.nn.sigmoid(h @ L["g"]["w"] + L["g"]["b"])
    phi = g * j.exp(1j * th)
    partner = phi.at[:, 0].set(0)
    pl = [partner.sum(1, keepdims=True) - partner, (partner * partner).sum(1, keepdims=True) - partner * partner]
    e = [j.ones_like(phi), pl[0], (pl[0] * pl[0] - pl[1]) / 2]
    T = h.shape[1]
    W = L["o"]["w"].reshape(3, 2 * rank, -1)
    y = L["o"]["b"]
    for m in range(3):
        f = phi * e[m] / math.comb(n0 if (n0 and m) else max(T - 2, m), m)
        F = j.concatenate([f.real, f.imag], -1)
        if not (n0 and m):
            F = F / j.sqrt((F ** 2).mean(-1, keepdims=True) + 1e-6)
        y = y + F @ W[m]
    return y


def effective(L, a):
    c = a.get("oscale")
    return L if c is None else dict(L, o=dict(w=c * L["o"]["w"], b=L["o"]["b"]))


def init(seed, a):
    if "rank" not in a:
        return _init4(seed, a)
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 64))
    mat = lambda r, c: jax.random.normal(next(keys), (r, c)) / r ** .5
    p = dict(input=mat(64, D))
    mat(D, 2 * D); mat(2 * D, 2)                                # bench4's m1, m2: drawn, not kept
    p |= dict(bias=j.zeros(2), head=mat(D, 2))
    R = a["rank"]
    p["legacy"] = dict(th=dict(w=mat(D, R), b=j.zeros(R)), g=dict(w=mat(D, R), b=j.zeros(R)), o=dict(w=mat(6 * R, D) / a.get("oscale", 1.), b=j.zeros(D)))
    return p


def apply(p, x, a):
    if "rank" not in a:
        return _apply4(p, x, a)
    h = j.tanh(x @ p["input"])
    c = argparse.Namespace(model="newton", heads=1, d=D, rank=a["rank"], order=3, phase=2., bosfix=1, fnorm=1, nfloor=1e-6, readout="full")
    if a.get("count"):
        y = newton_layer(effective(p["legacy"], a), h, a["count"], a["rank"])
    else:
        y = legacy.attention(c, effective(p["legacy"], a), h)
    return j.tanh(h[:, -1] + y[:, -1]) @ p["head"] + p["bias"]


bench4.ARMS, bench4.init, bench4.apply = ARMS, init, apply      # bench4.trainer reads bench4.apply
bench8.ARMS, bench8.init, bench8.apply = ARMS, init, apply


def flat(params):
    return {jax.tree_util.keystr(k): np.asarray(w) for k, w in jax.tree_util.tree_flatten_with_path(params)[0]}


def chars(trained, untrained):
    """trained: flat dict with leading axes [rate, seed]; untrained: [seed]."""
    rng = np.random.default_rng(0)
    L, S = trained["['input']"].shape[:2]
    return dict(trained=[[newton_chars.analyse(newton_chars.from_legacy(lambda k: trained[k][i, s].astype(np.float64)), rng) for s in range(S)] for i in range(L)],
                init=[newton_chars.analyse(newton_chars.from_legacy(lambda k: untrained[k][s].astype(np.float64)), rng) for s in range(S)])


def curve(name, n, updates, lrs, out):
    a = bench4.ARMS[name]
    train = bench4.trainer(a, updates, n)
    acc = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (bench4.apply(p, x, a).argmax(-1) == y).mean(), (0, 0, 0)), (0, None, None)))
    params = jax.tree.map(lambda *w: j.stack(w), *[bench4.init(s, a) for s in (0, 1, 2)])
    count = sum(w[0].size for w in jax.tree.leaves(params))
    data = [generate("set", 10000 + s, n) for s in (0, 1, 2)]
    x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
    del data
    vx, vy = generate("set", 9200, 1024)
    tx, ty = generate("set", 20000, 1024)
    three = lambda t: j.broadcast_to(j.array(t), (3, *t.shape))
    t0 = time.time()
    fitted, loss = train(params, x, y, j.array(lrs), j.arange(3))
    jax.block_until_ready(loss)
    seconds = time.time() - t0
    row = dict(arm=name, task="set", N=n, lrs=lrs, params=int(count), updates=updates, seconds=seconds, final_loss=np.asarray(loss).tolist(),
               train=np.asarray(acc(fitted, x[:, :1024], y[:, :1024])).tolist(), val=np.asarray(acc(fitted, three(vx), three(vy))).tolist(),
               test=np.asarray(acc(fitted, three(tx), three(ty))).tolist())
    F, F0 = flat(fitted), flat(params)
    row["chars"] = chars(F, F0)
    np.savez_compressed(out / f"params_{name}_{n}.npz", **F)
    with (out / "cells.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(dict(arm=name, N=n, sec=round(seconds, 1))), flush=True)
    del x, y, fitted


def rule(i, spec, out):
    row, params = bench8.fit(spec)
    a = bench4.ARMS[spec["arm"]]
    F0 = flat(jax.tree.map(lambda *w: j.stack(w), *[bench4.init(s, a) for s in (0, 1, 2)]))
    F = flat(params)
    row["chars"] = chars(F, F0)
    np.savez_compressed(out / f"params_rule{i}_{spec['arm']}.npz", **F)
    with (out / "fits.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(dict(arm=row["arm"], tag=row["tag"], stages=row["stages"], sec=round(row["seconds"], 1))), flush=True)


def mixed(i, spec, out):
    """bench8's rule fit on a mix of table sizes (8 and 12 cards in the paper). Each update draws BATCH / len(sizes) deals from each
    size's training set (equal weight in the loss); validation is the same mix (own cards); tests at every size in
    spec["test_sizes"], own cards and held-out cards."""
    import optax
    from bench2 import BATCH
    a = ARMS[spec["arm"]]
    lrs, m, sizes, n_each, updates = spec["lrs"], spec["m"], spec["sizes"], spec["n_each"], spec["updates"]
    S = bench8.query_set(m)
    k = BATCH // len(sizes)
    data = [[bench8.deals(10000 + s + 500 * z, n_each, S, T) for s in (0, 1, 2)] for z, T in enumerate(sizes)]
    X = [j.array(np.stack([d[0] for d in ds])) for ds in data]
    Y = [j.array(np.stack([d[1] for d in ds])) for ds in data]
    del data
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())

    def train(p, X, Y, lr, seed):
        def step(state, t):
            p, st = state
            key = jax.random.fold_in(jax.random.PRNGKey(seed), t)
            ixs = [jax.random.randint(jax.random.fold_in(key, z), (k,), 0, n_each) for z in range(len(sizes))]
            loss = lambda p: sum(optax.softmax_cross_entropy_with_integer_labels(apply(p, x[ix], a), y[ix]).mean() for x, y, ix in zip(X, Y, ixs)) / len(sizes)
            value, g = jax.value_and_grad(loss)(p)
            u, st = tx.update(g, st, p)
            return (jax.tree.map(lambda w, c: w - lr * c, p, u), st), value
        (p, _), losses = jax.lax.scan(step, (p, tx.init(p)), j.arange(updates))
        return p, losses[-200:].mean()
    fit_ = jax.jit(jax.vmap(jax.vmap(train, (0, 0, 0, None, 0)), (0, None, None, 0, None)))
    params0 = jax.tree.map(lambda *w: j.stack(w), *[init(s, a) for s in (0, 1, 2)])
    params = jax.tree.map(lambda w: j.broadcast_to(w, (len(lrs), *w.shape)), params0)
    t0 = time.time()
    params, loss = fit_(params, X, Y, j.array(lrs), j.arange(3))
    jax.block_until_ready(loss)
    acc = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, None, None)), (0, None, None)))
    acc_own = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, a).argmax(-1) == y).mean(), (0, 0, 0)), (0, None, None)))
    row = dict(arm=spec["arm"], tag=spec.get("tag", ""), m=m, sizes=sizes, n_each=n_each, updates=updates, lrs=lrs,
               params=int(sum(w[0, 0].size for w in jax.tree.leaves(params))), seconds=time.time() - t0, final_loss=np.asarray(loss).tolist(),
               train={str(T): np.asarray(acc_own(params, x[:, :1024], y[:, :1024])).tolist() for T, x, y in zip(sizes, X, Y)})
    vals = [np.asarray(acc(params, *map(j.array, bench8.deals(9200 + z, 1024, S, T)))) for z, T in enumerate(sizes)]
    row["val"] = (sum(vals) / len(vals)).tolist()
    seeds = {8: (20000, 20001), 16: (20002, 20003)}                 # bench8's own test deals at 8 and 16 cards
    for T in spec["test_sizes"]:
        s_in, s_out = seeds.get(T, (21000 + 10 * T, 21001 + 10 * T))
        row[f"in{T}"] = np.asarray(acc(params, *map(j.array, bench8.deals(s_in, 1024, S, T)))).tolist()
        row[f"out{T}"] = np.asarray(acc(params, *map(j.array, bench8.deals(s_out, 1024, bench8.HELD, T)))).tolist()
    F, F0 = flat(params), flat(params0)
    row["chars"] = chars(F, F0)
    np.savez_compressed(out / f"params_mixed{i}_{spec['arm']}.npz", **F)
    with (out / "fits.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(dict(arm=row["arm"], tag=row["tag"], sec=round(row["seconds"], 1))), flush=True)


def main():
    job = json.loads(sys.argv[1])
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    for name, n, updates, lrs in job.get("cells", []):
        curve(name, n, updates, lrs, out)
        jax.clear_caches()
    for i, spec in enumerate(job.get("rule", [])):
        rule(i, spec, out)
        jax.clear_caches()
    for i, spec in enumerate(job.get("mixed", [])):
        mixed(i, spec, out)
        jax.clear_caches()


if __name__ == "__main__":
    main()
