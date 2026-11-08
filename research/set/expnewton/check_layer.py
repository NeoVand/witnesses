"""CPU check: the trained layer as coded (bench_exp.apply, JAX float32 channel form) against the float64 brute force over
all distinct table pairs (bench_exp.np_forward), with random weights made sharp, at 8, 16, 32 and 64 table cards and
beta 1, 4, 8. Also times one CPU training step. Writes check_layer.json.

    python -I check_layer.py [smoke]      (jax, optax, numpy)
"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, jax, jax.numpy as j
import bench_exp as E
import bench8

res = []
for T in (8, 16, 32, 64):
    S = bench8.ORDER if T > 40 else bench8.query_set(27)
    if T <= 40:
        x, y = bench8.deals(31000 + T, 256, S, T)
    else:                                              # every 64-card table has a set: build inputs directly
        rng = np.random.default_rng(T)
        x = np.zeros((256, T + 2, 64), np.float32); x[:, 0, 63] = 1; x[:, T + 1, 62] = 1
        for b in range(256):
            ids = rng.choice(81, T + 1, replace=False)
            for t, c in enumerate(ids):
                for a in range(4):
                    x[b, t + 1, 3 * a + E.CARDS[c, a]] = 1
    for beta in (1., 4., 8.):
        for sharp in (1., 8.):
            p = E.init(0)
            p = dict(p, clock=p["clock"] * sharp * 4, beta=j.array(np.log(np.expm1(beta)), j.float32))
            P = {k: np.asarray(v, np.float64) for k, v in p.items()}
            lj = np.asarray(E.apply(p, j.array(x)))
            ln = E.np_forward(P, x)
            res.append(dict(T=T, beta=beta, clock_scale=sharp * 4, max_logit_diff=float(np.abs(lj - ln).max()),
                            logit_scale=float(np.abs(ln).max()), argmax_agree=float((lj.argmax(-1) == ln.argmax(-1)).mean())))
            print(json.dumps(res[-1]), flush=True)

out = dict(layer=res)
if len(sys.argv) > 1 and sys.argv[1] == "smoke":
    # training side only: rule data (m = 27, 8-card tables), 3 seeds, rate 3e-3, 1,000 updates; loss and beta only
    S = bench8.query_set(27)
    data = [bench8.deals(10000 + s, 4096, S, 8) for s in (0, 1, 2)]
    x, y = j.array(np.stack([d[0] for d in data])), j.array(np.stack([d[1] for d in data]))
    params = jax.tree.map(lambda *w: j.stack(w), *[E.init(s) for s in (0, 1, 2)])
    params = jax.tree.map(lambda w: j.broadcast_to(w, (1, *w.shape)), params)
    tr = E.trainer(1000, 4096)
    t0 = time.time()
    fitted, loss, curve = tr(params, x, y, j.array([3e-3]), j.arange(3))
    jax.block_until_ready(loss)
    out["smoke"] = dict(seconds=time.time() - t0, final_loss=np.asarray(loss).tolist(), curve=np.asarray(curve).round(3).tolist(),
                        beta=np.asarray(jax.nn.softplus(fitted["beta"])).tolist())
    print(json.dumps(out["smoke"]))
(HERE / "check_layer.json").write_text(json.dumps(out, indent=1))
