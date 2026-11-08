"""Why trained Newton fails on 16-card tables, from its normalisations (Methods, "Newton's 16-card threshold").

Theory. The layer reads the order-3 feature f_2 = phi_q e_2 / C(n, 2): a mean over the C(n, 2)
table pairs. One set among the pairs moves the mean by 1/C(n, 2), and the read-out's offset is a constant learned at
n = 8 cards. So a read-out that crosses zero at N0 sets on 8-card tables (0 < N0 < 1) crosses at
    N*_16 = N0 * C(16, 2) / C(8, 2) = 4.29 N0
on 16-card tables, if the decision depends on the table only through the pair mean (the "mean model"). Positives
with fewer than N*_16 sets are then called negative.

Protocol, in this order, per trained layer (rates 1e-3 and 3e-3, seeds 0-2):
  P. prediction from 8-card deals only: the mean margin at N = 0, 1, 2 sets, a straight line through them, its zero
     N0, then N*_16 and the predicted 16-card accuracy 1/2 + 1/2 P(N > N*_16 | positive, 16 cards), with the exact
     distribution of N on the bench's 16-card positives (charlaw_geometry / combinatorics). Written to the output
     before any 16-card deal is scored.
  M. measurement on the bench's 16-card test sets (in16: seed 20002, out16: seed 20003, 1,024 each) and on 20,000
     fresh 16-card deals: accuracy, and accuracy by N.
  D. diagnostics: the best constant shift of the margin on 16-card deals (an oracle threshold; if it restores the
     accuracy, the failure is the threshold, not the signal), the order-3 RMS normaliser at 8 and 16 cards, and the
     part of the margin's dependence on N that runs through the order-3 block.

python -I table16.py params/rule_newton_oscale_cpu_24000.npz table16.json   (jax, optax, numpy)
"""
import json, math, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np, jax, jax.numpy as j, argparse
import bench_nset, bench8, bench4
import hoa_polybench as legacy
from bench2 import D

ARM = bench4.ARMS["newton_oscale"]


def unflatten(F, ri, si):
    p = {}
    for key, w in F.items():
        path = re.findall(r"\['([^']+)'\]", key)
        d = p
        for part in path[:-1]:
            d = d.setdefault(part, {})
        d[path[-1]] = j.asarray(w[ri, si])
    return p


def set_counts(x, T):
    attrs = np.argmax(np.asarray(x)[:, 1:T + 2, :12].reshape(len(x), T + 1, 4, 3), -1)
    q, tab = attrs[:, T], attrs[:, :T]
    N = np.zeros(len(x), int)
    for a in range(T):
        for b in range(a + 1, T):
            N += ((q + tab[:, a] + tab[:, b]) % 3 == 0).all(-1)
    return N


def margin(p, x):
    out = np.concatenate([np.asarray(bench_nset.apply(p, j.asarray(x[i:i + 2048]), ARM)) for i in range(0, len(x), 2048)])
    return out[:, 1] - out[:, 0]


def internals(p, x):
    """RMS normalisers of the three orders at the card in hand, and the parts of y (before the bias)."""
    h = j.tanh(j.asarray(x) @ p["input"])
    L = bench_nset.effective(p["legacy"], ARM)
    c = argparse.Namespace(model="newton", heads=1, d=D, rank=64, order=3, phase=2., bosfix=1, fnorm=0, nfloor=1e-6,
                           readout="full", want_parts=True)
    raw = legacy.attention(c, L, h)[:, -1]                           # parts without normalisation, [B, 3, D]
    th = 2. * (h @ L["th"]["w"] + L["th"]["b"]); g = jax.nn.sigmoid(h @ L["g"]["w"] + L["g"]["b"])
    phi = g * j.exp(1j * th); P = phi.at[:, 0].set(0)
    p1, p2 = P.sum(1, keepdims=True) - P, (P * P).sum(1, keepdims=True) - P * P
    T = x.shape[1]
    f = [phi, phi * p1 / (T - 2), phi * (p1 * p1 - p2) / 2 / math.comb(T - 2, 2)]
    rho = [np.asarray(j.sqrt(((v[:, -1].real ** 2).sum(-1) + (v[:, -1].imag ** 2).sum(-1)) / 128 + 1e-6)) for v in f]
    return rho


def positives_given_n(T):
    tot = math.comb(80, T)
    P = [math.comb(40, N) * math.comb(40 - N, T - 2 * N) * 2 ** (T - 2 * N) / tot for N in range(T // 2 + 1)]
    pos = 1 - P[0]
    return [v / pos for v in P]                                       # index N; entry 0 is P(N=0)/pos, unused


def main(npz, dest):
    F = dict(np.load(npz))
    S = bench8.query_set(27)
    x8, y8 = bench8.deals(41000, 30000, S, 8)
    N8 = set_counts(x8, 8)
    PN16 = positives_given_n(16)
    rows = []
    for ri, lr in enumerate((1e-3, 3e-3)):
        for si in range(3):
            p = unflatten(F, ri, si)
            m8 = margin(p, x8)
            acc8 = float(((m8 > 0) == (y8 == 1)).mean())
            means = {int(N): float(m8[N8 == N].mean()) for N in (0, 1, 2)}
            counts = {int(N): int((N8 == N).sum()) for N in (0, 1, 2, 3)}
            A = np.array([[1, 0], [1, 1], [1, 2]], float)
            w = np.array([counts[0], counts[1], counts[2]], float)
            coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], np.array([means[0], means[1], means[2]]) * np.sqrt(w), rcond=None)[0]
            N0 = float(-coef[0] / coef[1])
            Nstar = N0 * math.comb(16, 2) / math.comb(8, 2)
            pred = 0.5 + 0.5 * sum(PN16[N] for N in range(1, len(PN16)) if N > Nstar)
            row = dict(lr=lr, seed=si, acc8=acc8, mean_margin8=means, n8=counts, line=coef.tolist(), N0=N0,
                       Nstar16=Nstar, predicted16=pred)
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(dict(prediction=rows), open(dest, "w"), indent=1)          # predictions saved before any 16-card score

    tests = {"in16 (seed 20002)": bench8.deals(20002, 1024, S, 16), "out16 (seed 20003)": bench8.deals(20003, 1024, bench8.HELD, 16),
             "fresh 16 (seed 41001)": bench8.deals(41001, 20000, S, 16)}
    x16f, y16f = tests["fresh 16 (seed 41001)"]
    N16 = set_counts(x16f, 16)
    for row in rows:
        p = unflatten(F, [1e-3, 3e-3].index(row["lr"]), row["seed"])
        meas = {}
        for name, (x, y) in tests.items():
            m = margin(p, x)
            meas[name] = float(((m > 0) == (y == 1)).mean())
        m16 = margin(p, x16f)
        byN = {int(N): dict(n=int((N16 == N).sum()), acc=float(((m16[N16 == N] > 0) == (N > 0)).mean()),
                            mean_margin=float(m16[N16 == N].mean())) for N in range(0, 5) if (N16 == N).sum() > 20}
        # oracle shift of the threshold on 16-card deals
        shifts = np.linspace(-20, 20, 801)
        accs = [float(((m16 > s) == (y16f == 1)).mean()) for s in shifts]
        best = int(np.argmax(accs))
        rho8 = internals(p, x8[:4000]); rho16 = internals(p, x16f[:4000])
        row.update(measured16=meas, by_N16=byN, oracle_shift=float(shifts[best]), oracle_acc16=accs[best],
                   rms8=[float(r.mean()) for r in rho8], rms16=[float(r.mean()) for r in rho16])
        print(json.dumps({k: row[k] for k in ("lr", "seed", "predicted16", "measured16", "oracle_acc16", "rms8", "rms16")}), flush=True)
        print("   by N:", json.dumps(byN), flush=True)
    json.dump(dict(prediction=rows), open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
