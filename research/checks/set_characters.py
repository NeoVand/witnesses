"""Proposition 3: the 81-channel form of exponential Newton against brute force over all pairs (numpy, float64).

On SET deals (the card in hand and n table cards as Z_3 phases of the four single-attribute clocks), the pair weight
exp(beta * sum_i (cos(2 pi x_i / 3) - 1)), x = x(q) + x(a_j) + x(a_k), is summed over all unordered pairs {j, k}
directly and through the 3^4 = 81 channels of equation (ja), whose coefficients are the Jacobi-Anger series folded
modulo 3. Both the partition function Z and the output for the symmetric value (v1_j * v2_k + v1_k * v2_j) / 2 are
compared, as the largest relative error over 40 deals at each table size and beta. The same comparison is made for
categorical clocks (random distributions over Z_3 for every card) against explicit convolution.

python -I set_characters.py [out.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from set_ref import (CARDS, brute_cat, brute_phase, channel_factors_cat, channel_factors_phase, channel_weights,
                     coef_alias, freqs, newton_join, set_deal)

rng = np.random.default_rng(0)


def rel(a, b):
    return float(np.max(np.abs(a - b) / np.abs(b)))


def relv(a, b):
    return float(np.max(np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)))


n3 = freqs(3)
phase = []
for n in (8, 16, 32, 64):
    for beta in (1., 2., 4., 8.):
        W = channel_weights([coef_alias(beta, 3)] * 4)
        ez = ey = 0.
        for _ in range(40):
            q, table = set_deal(rng, n)
            th_q, th_t = 2 * np.pi * CARDS[[q]] / 3, 2 * np.pi * CARDS[table] / 3
            v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
            Zb, Yb = brute_phase(th_q, th_t, beta, v1, v2)
            Zc, Yc = newton_join(channel_factors_phase(th_q, n3), channel_factors_phase(th_t, n3), W, v1, v2)
            ez, ey = max(ez, rel(Zc, Zb)), max(ey, relv(Yc, Yb))
        phase.append(dict(n=n, beta=beta, errZ=ez, errY=ey))

categorical = []
for n in (8, 16, 32, 64):
    for beta in (1., 4., 8.):
        W = channel_weights([coef_alias(beta, 3)] * 4)
        ez = ey = 0.
        for _ in range(10):
            conc = rng.choice([0.1, 1., 10.])
            pq = rng.dirichlet(conc * np.ones(3), (1, 4))
            pt = rng.dirichlet(conc * np.ones(3), (n, 4))
            v1, v2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
            Zb, Yb = brute_cat(pq, pt, beta, 3, v1, v2)
            Zc, Yc = newton_join(channel_factors_cat(pq, 3), channel_factors_cat(pt, 3), W, v1, v2)
            ez, ey = max(ez, rel(Zc, Zb)), max(ey, relv(Yc, Yb))
        categorical.append(dict(n=n, beta=beta, errZ=ez, errY=ey))

print("SET deals, Z_3 phases, 81 channels against brute force (largest relative error of Z and of the output):")
for r in phase:
    print(f"  n={r['n']:2d} beta={r['beta']:.0f}: Z {r['errZ']:.1e}  output {r['errY']:.1e}")
low = [max(r["errZ"], r["errY"]) for r in phase if r["beta"] <= 4]
print(f"  largest over n = 8..64, beta <= 4: {max(low):.1e}")
print("Categorical clocks against explicit convolution:")
for r in categorical:
    print(f"  n={r['n']:2d} beta={r['beta']:.0f}: {max(r['errZ'], r['errY']):.1e}")
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(dict(phase=phase, categorical=categorical), indent=1))
