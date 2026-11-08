"""Figure data for the four clocks: the sorted Fourier spectrum of a trained cubic's score tensor, the score of every
pair for one query, and, per attribute, the mean centred score by the attribute's sum mod 3.
python3 -I clocks_data.py <params.npz> <rate index> <seed> <out.json>
Fig. 3b (clocks_m81.json): python3 -I clocks_data.py results8/q-m81b/params_0_c1.npz 0 0 clocks_m81.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism import features, R, CARDS, CHI, S, WEIGHT

P = dict(np.load(sys.argv[1]))
w = lambda n: P[f"['{n}']"][int(sys.argv[2]), int(sys.argv[3])]
hq, ht = np.tanh(features(True) @ w("input")), np.tanh(features(False) @ w("input"))
A = np.einsum("qc,jc,kc->qjk", hq @ w("q"), ht @ w("k1"), ht @ w("k2")) / R ** .5
A = A - A.mean((1, 2), keepdims=True)
F = np.einsum("qjk,qs,jt,ku->stu", A, CHI.conj(), CHI.conj(), CHI.conj(), optimize=True) / 81 ** 3
power = (np.abs(F) ** 2).ravel()
order = np.argsort(power)[::-1]
share = power[order] / power.sum()
labels = []
for i in order[:12]:
    s, t, u = S[i // 6561], S[(i // 81) % 81], S[i % 81]
    labels.append(dict(s=s.tolist(), t=t.tolist(), u=u.tolist()))
sums = (CARDS[:, None, None, :] + CARDS[None, :, None, :] + CARDS[None, None, :, :]) % 3  # [q, j, k, attribute]
by_attr = [[float(A[sums[..., a] == r].mean()) for r in range(3)] for a in range(4)]
q = 0
partner = ((-(CARDS[:, None, :] + CARDS[None, :, :])) % 3 * np.array([1, 3, 9, 27])).sum(-1)
json.dump(dict(source=sys.argv[1], seed=int(sys.argv[3]), share_sorted_top=share[:200].tolist(), share_cumulative_8=float(share[:8].sum()), n_characters=int(power.size),
               top=labels, by_attribute=by_attr, query=q, pair_scores=A[q].round(4).tolist(), partner=partner[q].tolist()), open(sys.argv[4], "w"))
print("top-8 share", round(float(share[:8].sum()), 3), "by attribute", np.round(by_attr, 2).tolist())
