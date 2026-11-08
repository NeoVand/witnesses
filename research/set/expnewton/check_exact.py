"""Check 1 (CPU): exponential Newton against brute-force softmax over distinct unordered pairs (numpy, float64 unless
stated). Writes check_exact.json.

A. Continuous random phases, truncated Jacobi-Anger (a_n = I_n(beta) e^{-beta}, |n| <= F): relative error of Z and of
   the output against F, d and beta; the smallest F reaching 1e-3 and 1e-6.
B. SET deals (Z_3 phases of the four single-attribute characters) at 8, 16, 32 and 64 table cards: the aliased F = 1
   series (exact on Z_3) and the truncated Bessel series at F = 1, 2, 4, 8.
C. Categorical clocks (random distributions over Z_3, the trained layer's form) against explicit convolution.
D. float32 channel form against the float64 brute force, on SET deals with and without a set, against beta and n.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from ref import (coef_bessel, coef_alias, freqs, channel_weights, channel_factors_phase, channel_factors_cat,
                 newton_join, brute_phase, brute_cat, CARDS, set_deal, set_count, complement)

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
out = {}


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


def relv(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)))


# ---------------- A. continuous phases ----------------
t0 = time.time()
A = []
BETAS = [0.5, 1, 2, 4, 8]
for d in (1, 2, 3, 4):
    Fmax = {1: 30, 2: 24, 3: 14, 4: 9}[d]
    for beta in BETAS:
        for F in range(1, Fmax + 1):
            K = (2 * F + 1) ** d
            if K > 140000:
                break
            n_ = np.arange(-F, F + 1)
            W = channel_weights([coef_bessel(beta, F)] * d)
            ez, ey = 0., 0.
            for inst in range(6):
                n = 16
                th_q, th_t = rng.uniform(0, 2 * np.pi, (3, d)), rng.uniform(0, 2 * np.pi, (n, d))
                v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
                Zb, Yb = brute_phase(th_q, th_t, beta, v1, v2)
                Zc, Yc = newton_join(channel_factors_phase(th_q, n_), channel_factors_phase(th_t, n_), W, v1, v2)
                ez, ey = max(ez, rel(Zc, Zb)), max(ey, relv(Yc, Yb))
            A.append(dict(d=d, beta=beta, F=F, channels=K, errZ=ez, errY=ey))
            if max(ez, ey) < 1e-9:
                break
need = {}
for d in (1, 2, 3, 4):
    for beta in BETAS:
        rows = [r for r in A if r["d"] == d and r["beta"] == beta]
        f3 = next((r["F"] for r in rows if max(r["errZ"], r["errY"]) <= 1e-3), None)
        f6 = next((r["F"] for r in rows if max(r["errZ"], r["errY"]) <= 1e-6), None)
        need[f"d{d}_b{beta}"] = dict(d=d, beta=beta, F_1e3=f3, F_1e6=f6,
                                    channels_1e3=None if f3 is None else (2 * f3 + 1) ** d,
                                    channels_1e6=None if f6 is None else (2 * f6 + 1) ** d)
# n-independence of the relative error (d = 2, beta = 2, F = 6)
nind = []
for n in (8, 16, 32, 64):
    F, d, beta = 6, 2, 2.
    n_ = np.arange(-F, F + 1)
    W = channel_weights([coef_bessel(beta, F)] * d)
    th_q, th_t = rng.uniform(0, 2 * np.pi, (3, d)), rng.uniform(0, 2 * np.pi, (n, d))
    v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
    Zb, Yb = brute_phase(th_q, th_t, beta, v1, v2)
    Zc, Yc = newton_join(channel_factors_phase(th_q, n_), channel_factors_phase(th_t, n_), W, v1, v2)
    nind.append(dict(n=n, errZ=rel(Zc, Zb), errY=relv(Yc, Yb)))
# per-clock truncation tail, the quantity the F rule is about: tau_F = 2 sum_{n > F} I_n(beta) e^{-beta}
tail = {str(b): [float(1 - coef_bessel(b, F).sum()) for F in range(0, 31)] for b in BETAS}
out["A"] = dict(rows=A, need=need, n_independence=nind, tail=tail, seconds=time.time() - t0)

# ---------------- B. SET deals, Z_3 phases ----------------
t0 = time.time()
B = []
n3 = freqs(3)
for n in (8, 16, 32, 64):
    for beta in (1., 2., 4., 8.):
        rows = dict(n=n, beta=beta)
        e_alias, e_bes = [0., 0.], {F: [0., 0.] for F in (1, 2, 4, 8)}
        for inst in range(40):
            q, table = set_deal(rng, n)
            th_q = 2 * np.pi * CARDS[[q]] / 3
            th_t = 2 * np.pi * CARDS[table] / 3
            v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
            Zb, Yb = brute_phase(th_q, th_t, beta, v1, v2)
            W = channel_weights([coef_alias(beta, 3)] * 4)
            Zc, Yc = newton_join(channel_factors_phase(th_q, n3), channel_factors_phase(th_t, n3), W, v1, v2)
            e_alias = [max(e_alias[0], rel(Zc, Zb)), max(e_alias[1], relv(Yc, Yb))]
            for F in e_bes:
                n_ = np.arange(-F, F + 1)
                Wb = channel_weights([coef_bessel(beta, F)] * 4)
                Zc, Yc = newton_join(channel_factors_phase(th_q, n_), channel_factors_phase(th_t, n_), Wb, v1, v2)
                e_bes[F] = [max(e_bes[F][0], rel(Zc, Zb)), max(e_bes[F][1], relv(Yc, Yb))]
        rows["alias_F1"] = dict(errZ=e_alias[0], errY=e_alias[1])
        rows["bessel"] = {str(F): dict(errZ=v[0], errY=v[1]) for F, v in e_bes.items()}
        B.append(rows)
out["B"] = dict(rows=B, seconds=time.time() - t0)

# ---------------- C. categorical clocks ----------------
t0 = time.time()
C = []
for n in (8, 16, 32, 64):
    for beta in (1., 4., 8.):
        ez, ey = 0., 0.
        for inst in range(10):
            conc = rng.choice([0.1, 1., 10.])
            pq = rng.dirichlet(conc * np.ones(3), (3, 4))
            pt = rng.dirichlet(conc * np.ones(3), (n, 4))
            v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
            Zb, Yb = brute_cat(pq, pt, beta, 3, v1, v2)
            W = channel_weights([coef_alias(beta, 3)] * 4)
            Zc, Yc = newton_join(channel_factors_cat(pq, 3), channel_factors_cat(pt, 3), W, v1, v2)
            ez, ey = max(ez, rel(Zc, Zb)), max(ey, relv(Yc, Yb))
        C.append(dict(n=n, beta=beta, errZ=ez, errY=ey))
# one-hot distributions reproduce the hard Z_3 phases
q, table = set_deal(rng, 16)
oh = lambda ids: np.eye(3)[CARDS[ids]]
Zh, _ = brute_phase(2 * np.pi * CARDS[[q]] / 3, 2 * np.pi * CARDS[table] / 3, 4.)
Zo, _ = newton_join(channel_factors_cat(oh([q]), 3), channel_factors_cat(oh(table), 3), channel_weights([coef_alias(4., 3)] * 4))
out["C"] = dict(rows=C, onehot_vs_phase=rel(Zo, Zh), seconds=time.time() - t0)


# ---------------- D. float32 ----------------
def negatives(rng, n):
    """A card in hand and n table cards with no set: one card from each of n of the 40 complementary pairs."""
    q = rng.integers(81)
    pairs, seen = [], set()
    for c in range(81):
        if c == q or c in seen:
            continue
        m = complement(q, c)
        seen |= {c, m}
        pairs.append((c, m))
    pick = rng.choice(len(pairs), n, replace=False)
    return q, np.array([pairs[i][rng.integers(2)] for i in pick])


def f32_join(Cq, Ct, W, v1, v2):
    Cq, Ct, W = Cq.astype(np.complex64), Ct.astype(np.complex64), W.astype(np.float32)
    v1, v2 = v1.astype(np.float32), v2.astype(np.float32)
    s0 = Ct.sum(0)
    e2 = (s0 * s0 - (Ct * Ct).sum(0)) / 2
    Z = ((W * e2)[None] * Cq).sum(1).real
    s1, s2 = Ct.T @ v1, Ct.T @ v2
    num = (Cq * W[None]) @ (s1 * s2 - (Ct * Ct).T @ (v1 * v2)) / 2
    return Z.astype(np.float64), (num.real / Z[:, None]).astype(np.float64)


t0 = time.time()
Dr = []
for n in (8, 16, 20, 32, 40):
    for beta in (2., 4., 6., 8., 10., 12.):
        W = channel_weights([coef_alias(beta, 3)] * 4)
        res = {}
        for kind in ("neg", "pos"):
            ez, ey, neg_Z = [], [], 0
            for inst in range(100):
                q, table = negatives(rng, n) if kind == "neg" else set_deal(rng, n)
                if kind == "pos" and set_count(q, table) == 0:
                    continue
                th_q, th_t = 2 * np.pi * CARDS[[q]] / 3, 2 * np.pi * CARDS[table] / 3
                v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
                Zb, Yb = brute_phase(th_q, th_t, beta, v1, v2)
                Zc, Yc = f32_join(channel_factors_phase(th_q, n3), channel_factors_phase(th_t, n3), W, v1, v2)
                ez.append(rel(Zc, Zb)); ey.append(relv(Yc, Yb)); neg_Z += int(Zc[0] <= 0)
            res[kind] = dict(deals=len(ez), errZ_median=float(np.median(ez)), errZ_max=float(np.max(ez)),
                             errY_median=float(np.median(ey)), errY_max=float(np.max(ey)), Z_nonpositive=neg_Z)
        Dr.append(dict(n=n, beta=beta, **res))
out["D"] = dict(rows=Dr, seconds=time.time() - t0)

(HERE / "check_exact.json").write_text(json.dumps(out, indent=1))

# ---------------- summary ----------------
print("A. F needed (max of Z and output relative error), continuous phases, n = 16")
for k, v in need.items():
    print(f"  d={v['d']} beta={v['beta']}: F(1e-3)={v['F_1e3']} [{v['channels_1e3']} ch], F(1e-6)={v['F_1e6']} [{v['channels_1e6']} ch]")
print("  n-independence d=2 beta=2 F=6:", [(r["n"], f"{max(r['errZ'], r['errY']):.1e}") for r in nind])
print("B. SET deals, Z_3 phases")
for r in B:
    print(f"  n={r['n']} beta={r['beta']}: alias F=1 {max(r['alias_F1'].values()):.1e}; Bessel F=1 {max(r['bessel']['1'].values()):.1e}, "
          f"F=2 {max(r['bessel']['2'].values()):.1e}, F=4 {max(r['bessel']['4'].values()):.1e}, F=8 {max(r['bessel']['8'].values()):.1e}")
print("C. categorical clocks:", [(r["n"], r["beta"], f"{max(r['errZ'], r['errY']):.1e}") for r in C], "one-hot vs phase", f"{out['C']['onehot_vs_phase']:.1e}")
print("D. float32, SET deals (median / max relative error of Z on negatives, of the output on negatives; nonpositive Z)")
for r in Dr:
    print(f"  n={r['n']} beta={r['beta']}: neg Z {r['neg']['errZ_median']:.1e}/{r['neg']['errZ_max']:.1e}, Y {r['neg']['errY_median']:.1e}/{r['neg']['errY_max']:.1e}, "
          f"Z<=0 {r['neg']['Z_nonpositive']}; pos Z {r['pos']['errZ_max']:.1e}, Y {r['pos']['errY_max']:.1e}")
