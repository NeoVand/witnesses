"""Background-gate margin of the hand-built capped propagator on one question (Sec. 3, Methods): the largest
background gate eps on non-edges at which it still answers, before and after the row cap, against 1/W.
python -I bg250.py paper/figures/data/q250.json"""
import json, sys
import numpy as np
q = json.load(open(sys.argv[1]))
N = max(max(u, v) for u, v in q["edges"]) + 1
N = max(N, q["goal"] + 1, q["start"] + 1)
A = np.zeros((N, N))
for u, v in q["edges"]:
    A[u, v] = 1
pres = np.zeros(N); pres[[int(x) for x in q["names"]] if isinstance(q["names"][0], int) else list(range(N))] = 1
pres = np.maximum(pres, (A.sum(0) + A.sum(1)) > 0)
cur, goal, ans = q["path"][-1], q["goal"], q["answer"]
kids = [int(v) for v in np.flatnonzero(A[cur])]
t0 = 0.99
def layer(eps, cap):
    pm = np.outer(pres, pres)
    G = (t0 * A.T + eps * (1 - A.T)) * pm
    np.fill_diagonal(G, 0.0)
    if cap:
        G = G * (t0 / np.maximum(G.sum(1, keepdims=True), t0))
    rho = np.abs(np.linalg.eigvals(G)).max()
    R = np.linalg.solve(np.eye(N) - G, np.eye(N))
    return R[goal], G, rho
def right(eps, cap):
    x, G, rho = layer(eps, cap)
    if rho >= 1: return False
    return x[ans] > max(x[c] for c in kids if c != ans)
for cap in (True, False):
    lo, hi = -12.0, 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if right(10 ** mid, cap) else (lo, mid)
    x, G, rho = layer(10 ** lo, cap)
    non = (1 - A.T) * np.outer(pres, pres); np.fill_diagonal(non, 0)
    print("cap" if cap else "nocap", "eps* log10 %.3f" % lo, "post-cap background max %.2e median %.2e" % (G[non > 0].max(), np.median(G[non > 0])), "rho %.3f" % rho)
x0, G0, _ = layer(0.0, False)
R0 = np.linalg.solve(np.eye(N) - G0 * np.outer(pres, pres), np.eye(N))
P = pres > 0
S = R0[np.ix_(P, P)].sum()
print("n present", int(P.sum()), "S", round(S, 1), "log10(1/S) %.3f" % np.log10(1 / S))
