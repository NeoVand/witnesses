"""Second model: the 16-card threshold from the layer's two normalisations, without 16-card labels.

table16.py's first model (the decision depends on the table only through the pair mean, so the threshold count scales
with C(n, 2)) predicted N*_16 = 2.3-2.5 sets and 59% accuracy; the layers measure 74%. That model ignores the RMS
normaliser, which divides the order-3 block by a per-deal rho_2 that shrinks when the table grows. This model keeps
both normalisations and nothing else:

  * On 8-card deals, the order-3 drive before normalisation, Z3 = F_2 W_o^(2) (a vector over the 96 units; F_2 is the
    pair mean), is fitted as Z3 = B3 + (N / C(8,2)) U: a baseline plus a response per set, by least squares.
  * On 16-card deals, WITHOUT their labels or set counts, the layer's own rho_2 and the rest of the drive
    (Yr = h_q + order-1 and order-2 parts + bias) are computed from the inputs.
  * The predicted margin of a 16-card deal with N sets is m(N) = sum_u w_u tanh(Yr_u + (B3 + (N / C(16,2)) U)_u / rho_2),
    with w the head's margin direction, averaged over the unlabeled deals. Its zero N*_16 is the predicted threshold,
    and 1/2 + 1/2 P(N > N*_16 | positive) the predicted accuracy (deterministic threshold; noise ignored).
  * The 16-card normalisers rho_2 and the drive Yr are label-free; the only 8-card inputs are B3 and U.

The measured crossing is read from table16.json (mean margin by N at 16 cards, linear interpolation).

python -I table16_model.py params/rule_newton_oscale_cpu_24000.npz table16.json table16_model.json
"""
import json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, jax, jax.numpy as j
import bench_nset, bench8, bench4
from table16 import unflatten, set_counts, margin, positives_given_n, ARM


def drives(p, x):
    """Per deal at the card in hand: Yr (h_q + order-1, order-2 parts + bias), Z3 = F_2 W2 (before RMS), rho_2."""
    L = bench_nset.effective(p["legacy"], ARM)
    h = j.tanh(j.asarray(x) @ p["input"])
    th = 2. * (h @ L["th"]["w"] + L["th"]["b"]); g = jax.nn.sigmoid(h @ L["g"]["w"] + L["g"]["b"])
    phi = g * j.exp(1j * th); P = phi.at[:, 0].set(0)
    p1, p2 = P.sum(1, keepdims=True) - P, (P * P).sum(1, keepdims=True) - P * P
    T = x.shape[1]
    f = [phi[:, -1], (phi * p1)[:, -1] / (T - 2), (phi * (p1 * p1 - p2) / 2)[:, -1] / math.comb(T - 2, 2)]
    W = L["o"]["w"]
    r = 64
    out = []
    for m, fm in enumerate(f):
        F = j.concatenate([fm.real, fm.imag], -1)
        rho = j.sqrt((F ** 2).mean(-1, keepdims=True) + 1e-6)
        out.append((F, rho, W[m * 2 * r:(m + 1) * 2 * r]))
    Yr = h[:, -1] + L["o"]["b"] + sum((F / rho) @ Wm for F, rho, Wm in out[:2])
    F2, rho2, W2 = out[2]
    Z3 = F2 @ W2
    w = p["head"][:, 1] - p["head"][:, 0]
    b = p["bias"][1] - p["bias"][0]
    m = (j.tanh(Yr + Z3 / rho2) @ w) + b
    return np.asarray(Yr), np.asarray(Z3), np.asarray(rho2[:, 0]), np.asarray(w), float(b), np.asarray(m)


def main(npz, t16, dest):
    F = dict(np.load(npz))
    S = bench8.query_set(27)
    x8, _ = bench8.deals(41000, 30000, S, 8)
    N8 = set_counts(x8, 8)
    x16, _ = bench8.deals(41001, 20000, S, 16)                 # labels and set counts not used below
    prev = {(r["lr"], r["seed"]): r for r in json.load(open(t16))["prediction"]}
    PN16 = positives_given_n(16)
    rows = []
    for ri, lr in enumerate((1e-3, 3e-3)):
        for si in range(3):
            p = unflatten(F, ri, si)
            Yr8, Z8, rho8, w, b, m8 = drives(p, x8)
            assert np.allclose(m8, margin(p, x8), atol=1e-3), "replica of the layer"
            A = np.c_[np.ones(len(N8)), N8 / 28.0]
            coef, *_ = np.linalg.lstsq(A, Z8, rcond=None)              # B3 = coef[0], U = coef[1]
            B3, U = coef
            resid = Z8 - A @ coef
            r2 = 1 - (resid ** 2).sum() / ((Z8 - Z8.mean(0)) ** 2).sum()
            Yr16, _, rho16, _, _, _ = drives(p, x16)
            def mbar(N, C, Yr, rho):
                return float((np.tanh(Yr + (B3 + (N / C) * U)[None] / rho[:, None]) @ w + b).mean())
            # check on 8 cards: the fitted model's mean margin by N against the measured one
            fit8 = {N: (mbar(N, 28, Yr8, rho8), float(m8[N8 == N].mean())) for N in (0, 1, 2)}
            grid = np.linspace(0, 6, 601)
            vals = np.array([mbar(N, 120, Yr16, rho16) for N in grid])
            cross = float(grid[np.argmax(vals > 0)]) if (vals > 0).any() else float("inf")
            pred = 0.5 + 0.5 * sum(PN16[N] for N in range(1, len(PN16)) if N > cross)
            meas = prev[(lr, si)]["by_N16"]
            mm = {int(k): v["mean_margin"] for k, v in meas.items()}
            Ncross = next(N + (-mm[N]) / (mm[N + 1] - mm[N]) for N in range(0, 4) if mm[N] < 0 <= mm[N + 1])
            row = dict(lr=lr, seed=si, r2_order3_fit8=float(r2), fit8_model_vs_measured=fit8,
                       rho2_8=float(rho8.mean()), rho2_16=float(rho16.mean()),
                       per_set_signal_ratio=float((28 * rho8.mean()) / (120 * rho16.mean())),
                       Nstar16_model2=cross, predicted16_model2=pred,
                       Nstar16_model1=prev[(lr, si)]["Nstar16"], predicted16_model1=prev[(lr, si)]["predicted16"],
                       Nstar16_measured=float(Ncross), measured16_fresh=prev[(lr, si)]["measured16"]["fresh 16 (seed 41001)"],
                       oracle16=prev[(lr, si)]["oracle_acc16"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
