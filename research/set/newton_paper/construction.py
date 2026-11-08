"""Hand-set Newton layer that solves SET exactly, built in the bench's own code (bench4.newton_bare, hoa_polybench).

Channels 1-40 hold one character each of (Z_3)^4 (one per conjugate pair {s, -s}), channel 0 the trivial character,
channels 41-63 are switched off by their gates. Phases theta_c = 2 pi <s_c, x> / 3 come from the bench's own maps:
h = tanh(x W_in) puts tanh(1) on one hidden unit per attribute value, theta = 2 (h W_th + b_th). The arity-3 block of
the output map reads Re(phi_q e_2 / C(n-1, 2)) into one free unit of y, so that, before the RMS normalisation,
    y* = kappa (count / C(n-1, 2) - eps),   count = number of table pairs completing a set with the card in hand,
and z* = tanh(h* + y*/rho) with h* = 0 at that unit and rho > 0 the RMS normaliser. The head reads sign(z*).
eps = 1/240 puts the threshold at count = 0.117 for 8 table cards and 0.5 for 16, so one weight setting is exact on both.

Checks: accuracy on the bench's test sets; count recovered from the layer to float precision; two variants
(threshold in the output bias instead of the trivial channel; no RMS normalisation).
Run from research/set:  python3 -I newton_paper/construction.py newton_paper/construction.json
"""
import argparse, itertools, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # research/set
import numpy as np, jax, jax.numpy as j
import bench4, bench8
from bench2 import D, R
from tasks import generate
import hoa_polybench as legacy

CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
NEG = ((-CARDS) % 3) @ np.array([1, 3, 9, 27])
REPS = [s for s in range(1, 81) if s < NEG[s]]          # one representative per conjugate pair: 40
assert len(REPS) == 40
UNIT, KAPPA, EPS = 95, 10.0, 1 / 240


def hand_params(threshold="trivial"):
    a = bench4.ARMS["newton_bare"]
    p = jax.tree.map(np.asarray, bench4.init(0, a))     # the bench's pytree; every layer weight is overwritten below
    W_in = np.zeros((64, D), np.float32)
    for a_ in range(4):
        for v in range(3):
            W_in[3 * a_ + v, 3 * a_ + v] = 1.0           # unit 3a+v = tanh(1) * [attribute a has value v]
    S = np.zeros((R, 4))
    S[1:41] = CARDS[REPS]                                # channel 0: trivial character; 41-63 unused
    th_w = np.zeros((D, R), np.float32)
    for a_ in range(4):
        for v in range(3):
            th_w[3 * a_ + v] = (2 * np.pi / 3) * S[:, a_] * v / (2 * np.tanh(1.0))
    g_b = np.where(np.arange(R) <= 40, 40.0, -40.0).astype(np.float32)   # gates 1 on channels 0-40, 0 elsewhere
    o_w = np.zeros((6 * R, D), np.float32)              # rows: arity k, then [Re of r channels, Im of r channels]
    base = 2 * 2 * R                                     # arity index 2 = the card in hand with an unordered pair
    o_w[base + 0, UNIT] = KAPPA * (1 / 81 - (EPS if threshold == "trivial" else 0.0))
    o_w[base + 1:base + 41, UNIT] = KAPPA * 2 / 81
    o_b = np.zeros(D, np.float32)
    if threshold == "bias":
        o_b[UNIT] = -KAPPA * EPS                          # threshold added after the normalisation
    head = np.zeros((D, 2), np.float32)
    head[UNIT] = [-5.0, 5.0]
    p.update(input=W_in, head=head, bias=np.zeros(2, np.float32))
    p["legacy"] = dict(th=dict(w=th_w, b=np.zeros(R, np.float32)), g=dict(w=np.zeros((D, R), np.float32), b=g_b),
                       o=dict(w=o_w, b=o_b))
    return jax.tree.map(j.asarray, p)


def apply(p, x, fnorm=1):
    """bench4.apply's newton_bare path, with the normalisation switchable for the variant."""
    if fnorm:
        return bench4.apply(p, x, bench4.ARMS["newton_bare"])
    h = j.tanh(x @ p["input"])
    c = argparse.Namespace(model="newton", heads=1, d=D, rank=R, order=3, phase=2., bosfix=1, fnorm=0, nfloor=1e-6, readout="full")
    y = legacy.attention(c, p["legacy"], h)
    return j.tanh(h[:, -1] + y[:, -1]) @ p["head"] + p["bias"]


def counts(x):
    """Number of table pairs completing a set with the card in hand, from the one-hot tokens."""
    attrs = np.argmax(np.asarray(x)[:, :, :12].reshape(len(x), -1, 4, 3), -1)    # [B, T, 4]
    q, table = attrs[:, -1], attrs[:, 1:-1]
    n = table.shape[1]
    out = np.zeros(len(x), int)
    for j_, k_ in itertools.combinations(range(n), 2):
        out += ((q + table[:, j_] + table[:, k_]) % 3 == 0).all(-1)
    return out


def recovered_count(p, x):
    """count read back from the layer: C(n-1,2) * (y* rho / kappa + eps), rho recomputed from the raw features."""
    h = j.tanh(x @ p["input"])
    c = argparse.Namespace(model="newton", heads=1, d=D, rank=R, order=3, phase=2., bosfix=1, fnorm=1, nfloor=1e-6, readout="full", want_parts=True)
    parts = legacy.attention(c, p["legacy"], h)[:, -1, 2, UNIT]                     # arity-3 part at the free unit
    L = p["legacy"]
    th = 2. * (h @ L["th"]["w"] + L["th"]["b"]); g = jax.nn.sigmoid(h @ L["g"]["w"] + L["g"]["b"])
    phi = g * j.exp(1j * th); P = phi.at[:, 0].set(0)
    p1, p2 = P.sum(1, keepdims=True) - P, (P * P).sum(1, keepdims=True) - P * P
    T = x.shape[1]
    C = math.comb(T - 2, 2)
    out = (phi * (p1 * p1 - p2) / 2 / C)[:, -1]
    rho = j.sqrt(((out.real ** 2).sum(-1) + (out.imag ** 2).sum(-1)) / (2 * R) + 1e-6)
    return np.asarray(C * (parts * rho / KAPPA + EPS))


def main(dest):
    sets = {"test (Fig. 5a, seed 20000)": generate("set", 20000, 1024), "validation (seed 9200)": generate("set", 9200, 1024),
            "fresh 100k (seed 31337)": generate("set", 31337, 100000)}
    ORDER = bench8.ORDER
    sets |= {"in8, m=27 (seed 20000)": bench8.deals(20000, 1024, ORDER[:27], 8), "out8, held-out cards (seed 20001)": bench8.deals(20001, 1024, bench8.HELD, 8),
             "in16 (seed 20002)": bench8.deals(20002, 1024, ORDER[:27], 16), "out16 (seed 20003)": bench8.deals(20003, 1024, bench8.HELD, 16),
             "full8, all 81 cards (seed 20004)": bench8.deals(20004, 1024, ORDER, 8)}
    rows = {}
    for name, threshold, fnorm in (("trivial-channel threshold (main)", "trivial", 1), ("threshold in the output bias", "bias", 1), ("no RMS normalisation", "trivial", 0)):
        p = hand_params(threshold)
        res = {}
        for sname, (x, y) in sets.items():
            x = j.asarray(x)
            pred = np.concatenate([np.asarray(apply(p, x[i:i + 4096], fnorm).argmax(-1)) for i in range(0, len(x), 4096)])
            res[sname] = dict(n=int(len(y)), correct=int((pred == y).sum()), accuracy=float((pred == y).mean()))
            if name.startswith("trivial") and fnorm:
                c = counts(x)
                assert ((c > 0).astype(int) == y).all(), "label is not [count >= 1]"
                rc = np.concatenate([recovered_count(p, x[i:i + 4096]) for i in range(0, len(x), 4096)])
                res[sname] |= dict(max_count_error=float(np.abs(rc - c).max()), max_count=int(c.max()), positives=int((c > 0).sum()))
        rows[name] = res
        print(name, {k: (v["correct"], v["n"]) for k, v in res.items()}, flush=True)
    used = {"input map": 12, "phase map": 12 * 41, "gate biases": 64, "output map (arity-3 block, one unit)": 41, "head": 2}
    json.dump(dict(rows=rows, kappa=KAPPA, eps=EPS, unit=UNIT, channels=dict(trivial=[0], characters=list(range(1, 41)), off=list(range(41, 64))),
                   representatives=[CARDS[s].tolist() for s in REPS], nonzero_weights=used), open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
