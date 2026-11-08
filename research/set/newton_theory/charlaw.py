"""The characters law: how well SET can be read from k exact characters, against the best read-out of them.

On the bench's SET deals (8 table cards, card in hand, balanced; see charlaw_geometry.py), keep the channels of a set
K of k character classes and ask how well the label can be read from them.

Read-outs (all thresholds and weights are fitted on training deals and scored on separate test deals):
  sum       the equal-weight sum of the order-3 real parts, S_K = sum_{s in K} M_s, with the best threshold;
            for K = all 40 this is the construction of Proposition prop-count.
  linear    logistic regression on the order-3 features of K (real parts M_s and imaginary parts c1 - c2),
            then the best threshold on its score: the read-out a Newton layer of order 3 has.
  bayes     (k <= 3, exact) the best possible read-out of everything the K channels carry at orders 1-3,
            together with the card in hand: any function of the histograms (n0, n1, n2) of s . b, s in K.

Exact values for k <= 3 come from a dynamic programme over the 40 opposite pairs; they also check the sampler.

Run:  python -I charlaw.py charlaw.json   (from newton_theory/, numpy and scipy, about 30 min on one CPU core)
"""
import itertools, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charlaw_geometry as geo

DOT = geo.DOT


# ---------------------------------------------------------------------------------------------- exact, k <= 3
def exact(K, n=8):
    """Exact joint counts of (histograms of the K channels, label) over all 8-subsets of the 80 vectors.

    State axes: t (cards so far), then (n0, n1) for each channel, then y. Returns dict info -> (count0, count1)
    for t = n, where info = tuple of (n0, n1, n2) per channel."""
    k = len(K)
    shape = (n + 1,) + (n + 1, n + 1) * k + (2,)
    A = np.zeros(shape)
    A[(0,) * (1 + 2 * k) + (0,)] = 1.0

    def shift(B, dt, dn, y_to_one):
        out = np.zeros_like(B)
        src = [slice(None)] * B.ndim
        dst = [slice(None)] * B.ndim
        src[0], dst[0] = slice(0, n + 1 - dt), slice(dt, n + 1)
        for ax, d in enumerate(dn):
            src[1 + ax], dst[1 + ax] = slice(0, n + 1 - d), slice(d, n + 1)
        if y_to_one:
            for yy in (0, 1):
                s2, d2 = list(src), list(dst)
                s2[-1], d2[-1] = yy, 1
                out[tuple(d2)] += B[tuple(s2)]
        else:
            out[tuple(dst)] += B[tuple(src)]
        return out

    for i in range(40):
        vals = [DOT[s, i] for s in K]          # s . v_i ; for -v_i it is -s . v_i
        new = A.copy()                          # choose neither
        for sign in (1, -1):                    # choose exactly one of +-v_i
            dn = []
            for c in vals:
                c = (sign * c) % 3
                dn += [1 if c == 0 else 0, 1 if c == 1 else 0]
            new += shift(A, 1, dn, False)
        dn = []                                 # choose both: an opposite pair, label 1
        for c in vals:
            dn += [2, 0] if c == 0 else [0, 1]
        new += shift(A, 2, dn, True)
        A = new
    A = A[n]
    out = {}
    for idx in zip(*np.nonzero(A.sum(-1))):
        info = tuple((idx[2 * a], idx[2 * a + 1], n - idx[2 * a] - idx[2 * a + 1]) for a in range(k))
        out[info] = (A[idx + (0,)], A[idx + (1,)])
    return out


def exact_scores(K):
    table = exact(K)
    T0 = sum(v[0] for v in table.values()); T1 = sum(v[1] for v in table.values())
    assert abs(T0 - math.comb(40, 8) * 2 ** 8) < 1 and abs(T0 + T1 - math.comb(80, 8)) < 1
    bayes = 0.5 * sum(max(c0 / T0, c1 / T1) for c0, c1 in table.values())
    S = {}
    for info, (c0, c1) in table.items():
        s = sum(h[0] * (h[0] - 1) // 2 + h[1] * h[2] for h in info)
        a, b = S.get(s, (0., 0.)); S[s] = (a + c0 / T0, b + c1 / T1)
    keys = sorted(S)
    p0 = np.array([S[s][0] for s in keys]); p1 = np.array([S[s][1] for s in keys])
    # predict positive iff S > threshold; best over thresholds between support points
    best = max(0.5 * (p0[:j].sum() + p1[j:].sum()) for j in range(len(keys) + 1))
    return dict(bayes=bayes, sum=best, info_values=len(table))


# ---------------------------------------------------------------------------------------------- sampled
def best_threshold(score_tr, y_tr):
    order = np.argsort(score_tr, kind="stable")
    s, y = score_tr[order], y_tr[order]
    n1 = y.sum(); n0 = len(y) - n1
    # predict positive for score > s[j]; accuracy (balanced) for every cut
    c0 = np.cumsum(1 - y); c1 = np.cumsum(y)
    acc = 0.5 * (c0 / n0 + (n1 - c1) / n1)
    valid = np.r_[s[1:] != s[:-1], True]
    j = np.argmax(np.where(valid, acc, -1))
    if acc[j] <= 0.5:                      # no cut beats predicting one class: put everything positive
        return s[0] - 1.0
    hi = s[j + 1] if j + 1 < len(s) else s[j] + 1
    return 0.5 * (s[j] + hi)


def balanced_acc(pred, y):
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def logistic(X, y, Xte):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X, Xte = (X - mu) / sd, (Xte - mu) / sd
    X1 = np.c_[X, np.ones(len(X))]
    def f(w):
        z = X1 @ w
        loss = np.logaddexp(0, z) - y * z
        g = X1.T @ (1 / (1 + np.exp(-z)) - y) / len(y)
        return loss.mean(), g
    w = minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x
    return X1 @ w, np.c_[Xte, np.ones(len(Xte))] @ w, w


SUB = None   # a fixed random subset of the training deals for the logistic fits (set in main)


class Data:
    def __init__(self, seed, per_label):
        rng = np.random.default_rng(seed)
        M, I, Y = [], [], []
        for label in (0, 1):
            for _ in range(per_label // 100000):
                i, e = geo.sample(rng, 100000, label)
                m, im = geo.arity3(geo.histograms(i, e))
                M.append(m.astype(np.int16)); I.append(im.astype(np.int16)); Y.append(np.full(len(m), label))
        self.M, self.I, self.y = np.concatenate(M), np.concatenate(I), np.concatenate(Y)


def sampled_scores(K, tr, te, fit_linear=True):
    K = list(K)
    out = {}
    s_tr, s_te = tr.M[:, K].sum(1).astype(float), te.M[:, K].sum(1).astype(float)
    t = best_threshold(s_tr, tr.y)
    out["sum"] = balanced_acc((s_te > t).astype(int), te.y)
    if fit_linear and K:
        X = np.c_[tr.M[:, K], tr.I[:, K]].astype(float); Xte = np.c_[te.M[:, K], te.I[:, K]].astype(float)
        sub = SUB if SUB is not None else slice(None)
        z_tr, z_te, w = logistic(X[sub], tr.y[sub].astype(float), Xte)
        t = best_threshold(z_tr, tr.y[sub])
        out["linear"] = balanced_acc((z_te > t).astype(int), te.y)
    return out


# ---------------------------------------------------------------------------------------------- families of K
def classes_by_weight():
    w = (geo.VEC[geo.REPS] != 0).sum(1)
    return {d: [s for s in range(40) if w[s] == d] for d in (1, 2, 3, 4)}


def span(K):
    """Dimension of the span of the representatives of K over Z_3 (rank of the projective configuration + 1)."""
    if not K:
        return 0
    A = geo.VEC[geo.REPS[list(K)]].copy() % 3
    r = 0
    for c in range(4):
        piv = next((i for i in range(r, len(A)) if A[i, c] % 3), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * (1 if A[r, c] == 1 else 2)) % 3
        for i in range(len(A)):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % 3
        r += 1
    return r


def line_through(s, t):
    vs, vt = geo.VEC[geo.REPS[s]], geo.VEC[geo.REPS[t]]
    pts = {s, t}
    for a, b in ((1, 1), (1, 2)):
        v = (a * vs + b * vt) % 3
        pts.add(int(geo.PAIR[v @ geo.CODE]))
    return sorted(pts)


def plane_perp(z):
    """The 13 classes s with s . z = 0 (a plane of the dual space)."""
    return [s for s in range(40) if (geo.VEC[geo.REPS[s]] @ z) % 3 == 0]


def main(dest):
    t0 = time.time()
    res = dict(exact={}, sampled=[], notes={})
    # exact, k <= 3, every orbit (k = 3: collinear or not)
    col = line_through(0, 1)[:3]
    noncol = [0, 1, next(s for s in range(2, 40) if span([0, 1, s]) == 3)]
    for name, K in (("k=1", [0]), ("k=2", [0, 1]), ("k=3 collinear", col), ("k=3 independent", noncol)):
        r = exact_scores(K)
        res["exact"][name] = dict(K=K, span=span(K), **r)
        print(name, r, round(time.time() - t0, 1), flush=True)

    tr, te = Data(1, 1000000), Data(2, 1000000)
    global SUB
    SUB = np.random.default_rng(4).permutation(len(tr.y))[:400000]
    print("data", round(time.time() - t0, 1), flush=True)
    rng = np.random.default_rng(3)
    W = classes_by_weight()
    families = []
    for k in range(0, 41):
        for r in range(12 if 0 < k < 40 else 1):
            families.append(("random", sorted(rng.choice(40, k, replace=False).tolist())))
    nested = W[1] + W[2] + W[3] + W[4]
    for k in (4, 16, 32, 40):
        families.append((f"by attributes, first {k}", sorted(nested[:k])))
    L = line_through(0, 1)
    families.append(("line (4 collinear)", L))
    frame = [0]
    for s in range(1, 40):
        if span(frame + [s]) == len(frame) + 1:
            frame.append(s)
        if len(frame) == 4:
            break
    families.append(("frame (4 independent)", sorted(frame)))
    P = plane_perp(np.array([1, 0, 0, 0]))
    families.append(("plane (13)", P))
    families.append(("plane complement (27)", sorted(set(range(40)) - set(P))))
    families.append(("line complement (36)", sorted(set(range(40)) - set(L))))
    families.append(("single attributes (the join's 4 clocks)", sorted(W[1])))
    for name, K in families:
        r = sampled_scores(K, tr, te)
        res["sampled"].append(dict(family=name, k=len(K), span=span(K), K=K, **r))
    print("sampled", round(time.time() - t0, 1), flush=True)
    # sampler check against the exact values
    for name, e in res["exact"].items():
        r = sampled_scores(e["K"], tr, te, fit_linear=True)
        e["sampled_sum"] = r["sum"]; e["sampled_linear"] = r["linear"]
    # moments of M_s given the label: per-channel variance, pair covariance, and the k-only law
    for y in (0, 1):
        M = te.M[te.y == y].astype(float)
        C = np.cov(M.T)
        off = C[~np.eye(40, dtype=bool)]
        res["notes"][f"label{y}"] = dict(mean_per_channel=float(M.mean()), mean_spread=float(M.mean(0).std()),
                                         var=float(np.diag(C).mean()), var_spread=float(np.diag(C).std()),
                                         cov=float(off.mean()), cov_spread=float(off.std()),
                                         var_S40=float(M.sum(1).var()))
    res["seconds"] = time.time() - t0
    json.dump(res, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
