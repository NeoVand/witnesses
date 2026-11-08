"""Checks for the count form (bench_nset.newton_layer, arm newton_count), CPU.

1. With the count form off, newton_layer equals hoa_polybench.attention (the bench's layer) on 8- and 16-card deals.
2. A hand-set count-form layer (40 characters + trivial channel, threshold N > 1/2 in the output bias), run through
   bench_nset.apply, is exact on tables of 8, 16, 20 and 26 cards with one weight setting.
3. Training-side only: 1,000 updates of newton_count at rate 3e-3 on the m = 27 rule data; training loss and
   training accuracy (no validation or test number is computed).
Run from research/set:  python3 -I newton_paper/count_check.py newton_paper/count_check.json
"""
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # research/set
import numpy as np, jax, jax.numpy as j, optax
import bench_nset, bench4, bench8
from bench2 import D, R, BATCH
import hoa_polybench as legacy

CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
NEG = ((-CARDS) % 3) @ np.array([1, 3, 9, 27])
REPS = [s for s in range(1, 81) if s < NEG[s]]
A = bench_nset.ARMS["newton_count"]
UNIT, KAPPA = 95, 10.0


def equivalence():
    p = bench_nset._init4(0, bench4.ARMS["newton_bare"])
    c = argparse.Namespace(model="newton", heads=1, d=D, rank=R, order=3, phase=2., bosfix=1, fnorm=1, nfloor=1e-6, readout="full")
    out = {}
    for T in (8, 16):
        x, _ = bench8.deals(777, 256, bench8.ORDER, T)
        h = j.tanh(j.asarray(x) @ p["input"])
        out[T] = float(j.abs(bench_nset.newton_layer(p["legacy"], h) - legacy.attention(c, p["legacy"], h)).max())
    return out


def hand():
    p = jax.tree.map(np.asarray, bench_nset.init(0, A))
    W_in = np.zeros((64, D), np.float32)
    for a_ in range(4):
        for v in range(3):
            W_in[3 * a_ + v, 3 * a_ + v] = 1.0
    S = np.zeros((R, 4)); S[1:41] = CARDS[REPS]
    th_w = np.zeros((D, R), np.float32)
    for a_ in range(4):
        for v in range(3):
            th_w[3 * a_ + v] = (2 * np.pi / 3) * S[:, a_] * v / (2 * np.tanh(1.0))
    g_b = np.where(np.arange(R) <= 40, 40.0, -40.0).astype(np.float32)
    C8 = math.comb(8, 2)
    o_w = np.zeros((6 * R, D), np.float32)
    o_w[4 * R + 0, UNIT] = KAPPA * C8 / 81                  # trivial channel: C(n,2)/C(8,2) in count form
    o_w[4 * R + 1:4 * R + 41, UNIT] = KAPPA * 2 * C8 / 81
    o_b = np.zeros(D, np.float32); o_b[UNIT] = -KAPPA / 2   # threshold N > 1/2, after no normaliser
    head = np.zeros((D, 2), np.float32); head[UNIT] = [-5., 5.]
    p.update(input=W_in, head=head, bias=np.zeros(2, np.float32))
    p["legacy"] = dict(th=dict(w=th_w, b=np.zeros(R, np.float32)), g=dict(w=np.zeros((D, R), np.float32), b=g_b),
                       o=dict(w=o_w / A["oscale"], b=o_b))      # stored as W / c, as the arm trains it
    p = jax.tree.map(j.asarray, p)
    res = {}
    for T in (8, 16, 20, 26):
        x, y = bench8.deals(4242 + T, 2048, bench8.ORDER, T)
        pred = np.asarray(bench_nset.apply(p, j.asarray(x), A).argmax(-1))
        res[T] = dict(n=int(len(y)), correct=int((pred == y).sum()))
    return res


def train_side(updates=1000, lr=3e-3, seed=0):
    x, y = bench8.deals(10000 + seed, 55296, bench8.ORDER[:27], 8)
    x, y = j.asarray(x), j.asarray(y)
    p = bench_nset.init(seed, A)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())
    st = tx.init(p)

    @jax.jit
    def step(p, st, t):
        ix = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(seed), t), (BATCH,), 0, len(x))
        v, g = jax.value_and_grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(bench_nset.apply(p, x[ix], A), y[ix]).mean())(p)
        u, st = tx.update(g, st, p)
        return jax.tree.map(lambda w, c: w - lr * c, p, u), st, v
    log = []
    for t in range(updates + 1):
        if t % 200 == 0:
            lg = bench_nset.apply(p, x[:1024], A)
            log.append(dict(step=t, train_loss=float(optax.softmax_cross_entropy_with_integer_labels(lg, y[:1024]).mean()),
                            train_acc=float((lg.argmax(-1) == y[:1024]).mean())))
        if t < updates:
            p, st, _ = step(p, st, t)
    return log


if __name__ == "__main__":
    out = dict(equivalence_max_abs=equivalence(), hand_set=hand(), train_side_seed0=train_side())
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out))
