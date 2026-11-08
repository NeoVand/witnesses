"""Exponential Newton on SET. The join's own read-out with its softmax over distinct unordered table pairs computed
at linear cost through 3^d Jacobi-Anger channels of d learned Z_3 clocks (d = 4: 81 channels; d = 6: 729).

Model (63,183 parameters with four clocks, 63,765 with six; the join c1 has 62,018):
  h = tanh(x W_in)                                   x: 64 one-hot channels, W_in 64 x 96, as the join
  clocks: p_i(x) = softmax(h W_c + b_c)_i            d categorical clocks over Z_3 (W_c 96 x 3d)
  beta = softplus(b) * scale                         one learned temperature; scale = 1 in training
  weight of the pair {j, k} for the card in hand q: E prod_i exp(beta (cos(2 pi (a_i + b_i + c_i) / 3) - 1)),
          a ~ p(q), b ~ p(x_j), c ~ p(x_k) independent (exactly the join's exp(beta * sum of 4 clock cosines) when the
          clocks are one-hot; positive always)
  value of the pair: (v1_j * v2_k + v1_k * v2_j) / 2, v1 = h W_v1, v2 = h W_v2 (96 x 128 each)
  y = sum_{j<k} w_jk value_jk / sum_{j<k} w_jk, computed as 81 channels n in {-1, 0, 1}^4 with weight
      W_n = prod_i a_{n_i}, a_0 = (1 + 2 eps) / 3, a_{+-1} = (1 - eps) / 3, eps = exp(-1.5 beta) (the aliased, exact,
      Jacobi-Anger series on Z_3) and per-token factor C^n(x) = prod_i u_i(x)_{n_i}, u = characteristic function of p_i;
      each channel's pair sums are Newton's e_2 of C^n and of C^n v (repeated partners removed). Linear in table size.
  z = tanh(h_q + y W_o) (W_o 128 x 96); logits = gelu(z W_1 + b_1) W_2 + b      the join's MLP head, as bench2.apply
Numerics: float32; Z is floored at 1e-6 * C(n, 2) (the float32 resolution of the channel sum; inactive unless Z is
below it). Every test set is also scored with a float64 brute-force forward pass over all pairs (numpy).

job: {"clocks": 4 or 6, "train_form": "channels" or "pairs", "rule": [{lrs, m, T, N, updates, test_sizes, tag,
      seeds, split}], "cells": [[N, updates, lrs], ...]}
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))   # research/set (bench8, bench2, tasks); flat on Modal
import bench8
from bench2 import BATCH
from tasks import generate

D, RV, NC, M = 96, 128, 4, 3
FUSED = False                             # an alternative order of the linear-form sums (see layer); unused in the paper's runs
TRAIN_FORM = "channels"                  # the six-clock runs train with "pairs" (same function, see layer)
BETA0 = 2.0
OMEGA = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))


def init(seed):
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 64))
    mat = lambda r, c: jax.random.normal(next(keys), (r, c)) / r ** .5
    return dict(input=mat(64, D), clock=mat(D, NC * M), clock_b=j.zeros(NC * M), beta=j.full((), math.log(math.expm1(BETA0))),
                v1=mat(D, RV), v2=mat(D, RV), o=mat(RV, D), m1=mat(D, 2 * D), mb=j.zeros(2 * D), m2=mat(2 * D, 2), bias=j.zeros(2))


def channel_weights(beta):
    eps = j.exp(-1.5 * beta)
    a1, a0 = (1 - eps) / 3, (1 + 2 * eps) / 3
    a = j.stack([a1, a0, a1])
    W = a
    for _ in range(NC - 1):
        W = (W[:, None] * a[None, :]).reshape(-1)
    return W


def layer(p, h, beta, form="channels"):
    """Output y (B, RV) and partition function Z (B,) at the card in hand (last token); partners are tokens 1..T-2."""
    B, T, _ = h.shape
    prob = jax.nn.softmax((h @ p["clock"] + p["clock_b"]).reshape(B, T, NC, M), -1)
    u1 = prob[..., 0] + prob[..., 1] * OMEGA + prob[..., 2] * (OMEGA * OMEGA)
    U = j.stack([j.conj(u1), j.ones_like(u1), u1], -1)                     # n = -1, 0, 1
    C = U[:, :, 0]
    for i in range(1, NC):
        C = (C[..., :, None] * U[:, :, i, None, :]).reshape(B, T, -1)       # (B, T, 81)
    Ct, Cq = C[:, 1:-1], C[:, -1]
    W = channel_weights(beta)
    s0 = Ct.sum(1)
    e2 = (s0 * s0 - (Ct * Ct).sum(1)) / 2
    Z = (W * Cq * e2).sum(-1).real
    ht = h[:, 1:-1]
    v1, v2 = (ht @ p["v1"]).astype(Ct.dtype), (ht @ p["v2"]).astype(Ct.dtype)
    pairs = (T - 2) * (T - 3) / 2
    if form == "pairs":
        # six-clock training: the same function through the pair-weight matrix G_jk = sum_n W_n C^n(q) C^n_j C^n_k
        # (the 81 or 729 channels give each pair's weight; the pair sum is explicit). Identical in exact arithmetic;
        # cost n^2 (K + RV) per query instead of n K RV, which is cheaper than the linear form below about RV partners.
        G = j.einsum("bn,bjn,bkn->bjk", W * Cq, Ct, Ct).real * (1 - j.eye(T - 2))
        Z = G.sum((1, 2)) / 2
        num = j.einsum("bjk,bjr,bkr->br", G, v1.real, v2.real) / 2
        return num / j.maximum(Z, 1e-6 * pairs)[:, None], Z
    s1, s2 = j.einsum("bjn,bjr->bnr", Ct, v1), j.einsum("bjn,bjr->bnr", Ct, v2)
    if FUSED:
        # the same linear-cost sums in a cheaper order. The repeated-partner term contracts the channels
        # first, sum_n a_n sum_j C_jn^2 (v1 v2)_j = sum_j (sum_n a_n C_jn^2) (v1 v2)_j, and the product term is one
        # fused contraction, so only s1 and s2 are (B, channels, RV) tensors. Identical in exact arithmetic.
        a = W * Cq
        num = (j.einsum("bn,bnr,bnr->br", a, s1, s2) - j.einsum("bj,bjr->br", j.einsum("bn,bjn->bj", a, Ct * Ct), v1 * v2)).real / 2
    else:
        p12 = j.einsum("bjn,bjr->bnr", Ct * Ct, v1 * v2)
        num = j.einsum("bn,bnr->br", W * Cq, s1 * s2 - p12).real / 2
    return num / j.maximum(Z, 1e-6 * pairs)[:, None], Z


def apply(p, x, scale=1., form=None):
    h = j.tanh(x @ p["input"])
    y, _ = layer(p, h, jax.nn.softplus(p["beta"]) * scale, form or TRAIN_FORM)
    z = j.tanh(h[:, -1] + y @ p["o"])
    return jax.nn.gelu(z @ p["m1"] + p["mb"]) @ p["m2"] + p["bias"]


def scale_for(T):
    """Reading B: beta grows like log of the number of pairs, relative to 8-card training tables."""
    return math.log(T * (T - 1) / 2) / math.log(28)


# ---------------- float64 brute force over pairs (numpy) ----------------
def np_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def np_forward(P, x, scale=1., chunk=128):
    """P: dict of float64 arrays for one model. Brute force over all distinct unordered table pairs."""
    out = []
    for s in range(0, len(x), chunk):
        xb = x[s:s + chunk].astype(np.float64)
        h = np.tanh(xb @ P["input"])
        B, T, _ = h.shape
        lg = (h @ P["clock"] + P["clock_b"]).reshape(B, T, NC, M)
        prob = np.exp(lg - lg.max(-1, keepdims=True)); prob /= prob.sum(-1, keepdims=True)
        beta = np.logaddexp(0., P["beta"]) * scale
        g = np.exp(beta * (np.cos(2 * np.pi * np.arange(M) / M) - 1))
        jj, kk = np.triu_indices(T - 2, 1)
        pt, pq = prob[:, 1:-1], prob[:, -1]
        pj, pk = pt[:, jj], pt[:, kk]                                       # (B, P, 4, 3)
        conv = lambda a, b: np.stack([sum(a[..., u] * b[..., (z - u) % M] for u in range(M)) for z in range(M)], -1)
        tot = conv(conv(pj, pk), pq[:, None])
        w = (tot @ g).prod(-1)                                              # (B, P)
        ht = h[:, 1:-1]
        v1, v2 = ht @ P["v1"], ht @ P["v2"]
        val = (v1[:, jj] * v2[:, kk] + v1[:, kk] * v2[:, jj]) / 2
        y = np.einsum("bp,bpr->br", w, val) / w.sum(1, keepdims=True)
        z = np.tanh(h[:, -1] + y @ P["o"])
        out.append(np_gelu(z @ P["m1"] + P["mb"]) @ P["m2"] + P["bias"])
    return np.concatenate(out)


# ---------------- clocks: do they sit on characters? ----------------
CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
SVEC = np.array([[(s // 3 ** a) % 3 for a in range(4)] for s in range(1, 81)])


def rank3(V):
    V = [list(v % 3) for v in V]
    r, rows, cols = 0, len(V), 4
    for c in range(cols):
        piv = next((i for i in range(r, rows) if V[i][c] % 3), None)
        if piv is None:
            continue
        V[r], V[piv] = V[piv], V[r]
        inv = 1 if V[r][c] == 1 else 2
        V[r] = [(e * inv) % 3 for e in V[r]]
        for i in range(rows):
            if i != r and V[i][c]:
                f = V[i][c]
                V[i] = [(a - f * b) % 3 for a, b in zip(V[i], V[r])]
        r += 1
    return r


def clocks(P):
    out = {}
    for query in (False, True):
        x = np.zeros((81, 64))
        for a in range(4):
            x[np.arange(81), 3 * a + CARDS[:, a]] = 1
        if query:
            x[:, 62] = 1
        h = np.tanh(x @ P["input"])
        lg = (h @ P["clock"] + P["clock_b"]).reshape(81, NC, M)
        prob = np.exp(lg - lg.max(-1, keepdims=True)); prob /= prob.sum(-1, keepdims=True)
        hard = prob.argmax(-1)                                              # (81, 4)
        fits, ss = [], []
        for i in range(NC):
            best = (0, None)
            for s in SVEC:
                for t in range(3):
                    f = float(((CARDS @ s + t) % 3 == hard[:, i]).mean())
                    if f > best[0]:
                        best = (f, s)
            fits.append(best[0]); ss.append(best[1].tolist())
        out["query" if query else "table"] = dict(fit=fits, s=ss, rank=rank3(np.array(ss)), confidence=prob.max(-1).mean(0).tolist())
    out["beta"] = float(np.logaddexp(0., P["beta"]))
    return out


# ---------------- training (bench8.trainer, this model) ----------------
def trainer(updates, N):
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())

    def train(p, x, y, lr, seed):
        def step(state, t):
            p, st = state
            ix = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed), t), (BATCH,), 0, N)
            value, g = jax.value_and_grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(apply(p, x[ix]), y[ix]).mean())(p)
            u, st = tx.update(g, st, p)
            return (jax.tree.map(lambda w, c: w - lr * c, p, u), st), value
        (p, _), losses = jax.lax.scan(step, (p, tx.init(p)), j.arange(updates))
        return p, losses[-200:].mean(), losses[::max(1, updates // 48)]
    return jax.jit(jax.vmap(jax.vmap(train, (0, 0, 0, None, 0)), (0, None, None, 0, None)))


def accuracy_fn(scale, chunk=128, form="channels"):
    f = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x, scale, form).argmax(-1) == y).sum(), (0, None, None)), (0, None, None)))
    return lambda p, x, y: sum(f(p, x[i:i + chunk], y[i:i + chunk]) for i in range(0, x.shape[0], chunk)) / x.shape[0]


def to_np(params, i, s):
    return {k: np.asarray(v[i, s], np.float64) for k, v in params.items()}


def score(params, x, y, T, lrs):
    """Accuracy per rate and seed: readings A (beta as learned) and B (beta * scale_for(T)), float32 channel form and
    float64 brute force."""
    r = {}
    for name, sc in (("A", 1.), ("B", scale_for(T))):
        r[name] = np.asarray(accuracy_fn(sc)(params, j.array(x), j.array(y))).tolist()
        r[name + "64"] = [[float((np_forward(to_np(params, i, s), x, sc).argmax(-1) == y).mean()) for s in range(3)] for i in range(len(lrs))]
        if TRAIN_FORM != "channels":
            r[name + "pairs"] = np.asarray(accuracy_fn(sc, form=TRAIN_FORM)(params, j.array(x), j.array(y))).tolist()
    return r


def flat(params):
    return {k: np.asarray(v) for k, v in params.items()}


def rule(i, spec, out):
    lrs, m, T, N, updates = spec["lrs"], spec["m"], spec["T"], spec["N"], spec["updates"]
    S = bench8.query_set(m)
    SEEDS = spec.get("seeds", [0, 1, 2])                                           # held-out-card rule: 3, 4, 5
    data = [bench8.deals(10000 + s, N, S, T) for s in SEEDS]                        # bench8.fit's stage-0 seeds
    x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
    del data
    params = jax.tree.map(lambda *w: j.stack(w), *[init(s) for s in SEEDS])
    params0 = params
    params = jax.tree.map(lambda w: j.broadcast_to(w, (len(lrs), *w.shape)), params)
    t0 = time.time()
    fit_ = trainer(updates, N)
    params_in = params
    params, loss, curve = fit_(params_in, x, y, j.array(lrs), j.array(SEEDS))
    jax.block_until_ready(loss)
    seconds = time.time() - t0
    if spec.get("smoke"):                                                            # second call: no compile
        t1 = time.time(); jax.block_until_ready(fit_(params_in, x, y, j.array(lrs), j.arange(3))[1]); spec["step_seconds"] = (time.time() - t1) / updates
    own_ = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x).argmax(-1) == y).sum(), (0, 0, 0)), (0, None, None)))
    own = lambda p, x, y: sum(own_(p, x[:, i:i + 128], y[:, i:i + 128]) for i in range(0, x.shape[1], 128)) / x.shape[1]
    row = dict(arm="expnewton", tag=spec.get("tag", ""), m=m, T=T, N=N, updates=updates, lrs=lrs,
               params=int(sum(w[0, 0].size for w in jax.tree.leaves(params))), seconds=seconds,
               final_loss=np.asarray(loss).tolist(), loss_curve=np.asarray(curve).tolist(),
               train=np.asarray(own(params, x[:, :1024], y[:, :1024])).tolist(),
               beta=np.asarray(jax.nn.softplus(params["beta"])).tolist(), step_seconds=spec.get("step_seconds"))
    del x, y
    if spec.get("smoke"):                                                            # training side only
        with (out / "smoke.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return
    vx, vy = bench8.deals(9200, 1024, S, 8)
    row["val"] = np.asarray(accuracy_fn(1.)(params, j.array(vx), j.array(vy))).tolist()
    row["seeds"] = SEEDS
    if spec.get("split"):
        # held-out-card rule: the 54 never-trained cards in hand split once, before running: validation cards HELD[:27],
        # test cards HELD[27:]. The rate is picked on validation-card deals; only test-card deals are scored.
        VALC, TESTC = bench8.HELD[:27], bench8.HELD[27:]
        row["val_own"] = row["val"]                                                  # the old rule, for reference
        vx, vy = bench8.deals(9300, 1024, VALC, 8)
        row["val"] = np.asarray(accuracy_fn(1.)(params, j.array(vx), j.array(vy))).tolist()
        for Tt in spec["test_sizes"]:
            row[f"test{Tt}"] = score(params, *bench8.deals(22000 + Tt, 1024, TESTC, Tt), Tt, lrs)
            print(json.dumps(dict(T=Tt, sec=round(time.time() - t0, 1))), flush=True)
        spec = dict(spec, test_sizes=[])
    seeds = {8: (20000, 20001), 16: (20002, 20003)}                                   # bench8's own test deals
    for Tt in spec["test_sizes"]:
        s_in, s_out = seeds.get(Tt, (21000 + 10 * Tt, 21001 + 10 * Tt))              # 20 cards: bench_nset.mixed's deals
        row[f"in{Tt}"] = score(params, *bench8.deals(s_in, 1024, S, Tt), Tt, lrs)
        row[f"out{Tt}"] = score(params, *bench8.deals(s_out, 1024, bench8.HELD, Tt), Tt, lrs)
        print(json.dumps(dict(T=Tt, sec=round(time.time() - t0, 1))), flush=True)
    row["clocks"] = [[clocks(to_np(params, i, s)) for s in range(3)] for i in range(len(lrs))]
    row["clocks_init"] = [clocks({k: np.asarray(v[s], np.float64) for k, v in params0.items()}) for s in range(3)]
    np.savez_compressed(out / f"params_rule{i}.npz", **flat(params))
    with (out / "fits.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(dict(kind="rule", tag=row["tag"], sec=round(time.time() - t0, 1))), flush=True)


def cell(n, updates, lrs, out, smoke=False):
    data = [generate("set", 10000 + s, n) for s in (0, 1, 2)]
    x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
    del data
    params = jax.tree.map(lambda *w: j.stack(w), *[init(s) for s in (0, 1, 2)])
    params = jax.tree.map(lambda w: j.broadcast_to(w, (len(lrs), *w.shape)), params)
    t0 = time.time()
    fit_ = trainer(updates, n)
    params_in = params
    params, loss, curve = fit_(params_in, x, y, j.array(lrs), j.arange(3))
    jax.block_until_ready(loss)
    seconds = time.time() - t0
    step_seconds = None
    if smoke:                                                                        # second call: no compile
        t1 = time.time(); jax.block_until_ready(fit_(params_in, x, y, j.array(lrs), j.arange(3))[1]); step_seconds = (time.time() - t1) / updates
    own_ = jax.jit(jax.vmap(jax.vmap(lambda p, x, y: (apply(p, x).argmax(-1) == y).sum(), (0, 0, 0)), (0, None, None)))
    own = lambda p, x, y: sum(own_(p, x[:, i:i + 128], y[:, i:i + 128]) for i in range(0, x.shape[1], 128)) / x.shape[1]
    if smoke:                                                                        # training side only
        with (out / "smoke.jsonl").open("a") as f:
            f.write(json.dumps(dict(kind="cell", N=n, lrs=lrs, updates=updates, seconds=seconds, final_loss=np.asarray(loss).tolist(),
                                    loss_curve=np.asarray(curve).tolist(), train=np.asarray(own(params, x[:, :1024], y[:, :1024])).tolist(),
                                    beta=np.asarray(jax.nn.softplus(params["beta"])).tolist(), step_seconds=step_seconds)) + "\n")
        return
    vx, vy = generate("set", 9200, 1024)
    tx_, ty_ = generate("set", 20000, 1024)
    acc = accuracy_fn(1.)
    row = dict(arm="expnewton", task="set", N=n, lrs=lrs, updates=updates, seconds=seconds,
               params=int(sum(w[0, 0].size for w in jax.tree.leaves(params))),
               final_loss=np.asarray(loss).tolist(), loss_curve=np.asarray(curve).tolist(),
               train=np.asarray(own(params, x[:, :1024], y[:, :1024])).tolist(),
               val=np.asarray(acc(params, j.array(vx), j.array(vy))).tolist(), test=np.asarray(acc(params, j.array(tx_), j.array(ty_))).tolist(),
               test_trainform=np.asarray(accuracy_fn(1., form=TRAIN_FORM)(params, j.array(tx_), j.array(ty_))).tolist(),
               test64=[[float((np_forward(to_np(params, i, s), tx_).argmax(-1) == ty_).mean()) for s in range(3)] for i in range(len(lrs))],
               beta=np.asarray(jax.nn.softplus(params["beta"])).tolist())
    np.savez_compressed(out / f"params_cell_{n}.npz", **flat(params))
    with (out / "cells.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(dict(kind="cell", N=n, sec=round(seconds, 1))), flush=True)


def main():
    global NC, FUSED, TRAIN_FORM
    job = json.loads(sys.argv[1])
    TRAIN_FORM = job.get("train_form", TRAIN_FORM)
    NC = job.get("clocks", NC)            # 4 clocks (81 channels) or 6 (729 channels)
    FUSED = job.get("fused", FUSED)
    out = Path(job["dest"])
    out.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(job.get("rule", [])):
        rule(i, spec, out)
        jax.clear_caches()
    for n, updates, lrs in job.get("cells", []):
        cell(n, updates, lrs, out, job.get("smoke", False))
        jax.clear_caches()


if __name__ == "__main__":
    main()
