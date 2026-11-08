"""Read SET's arithmetic out of a trained join (one layer of 2-simplicial attention, the "cubic" arm c1).

The score of query card q for the pair (j, k) is A(q, j, k) = sum_c Q(q)_c K1(j)_c K2(k)_c / sqrt(R), with the token
features the model sees (query flag on q). Terms that depend on q alone do not change the softmax over pairs, so we
subtract the mean over pairs for each q. Then we expand the centred tensor in the characters of (Z_3)^4,
chi_s(x) = w^(s . x) with w = exp(2 pi i / 3), on each of the three axes, and report where its variance lies.

python3 -I mechanism.py <params.npz> <rate index> <out.json>   (needs numpy only)
Clock share of the paper (joins with all 81 cards in hand, the picked rate, index 0):
  python3 -I mechanism.py results8/q-m81a/params_0_c1.npz 0 results8/q-m81a/mech_0_c1.json   (and q-m81b)
single_attribute_diagonal is the share on the eight single-attribute characters.
python3 -I mechanism.py --selftest
"""
import json, sys
import numpy as np

R = 64
CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
S = CARDS.copy()  # characters are indexed by the same 81 vectors
CHI = np.exp(2j * np.pi / 3 * ((CARDS @ S.T) % 3))  # CHI[x, s] = w^(s . x)
WEIGHT = (S != 0).sum(1)


def features(query):
    x = np.zeros((81, 64), np.float32)
    for a in range(4):
        x[np.arange(81), 3 * a + CARDS[:, a]] = 1
    if query:
        x[:, 62] = 1
    return x


def analyse(A):
    A = A - A.mean((1, 2), keepdims=True)
    F = np.einsum("qjk,qs,jt,ku->stu", A, CHI.conj(), CHI.conj(), CHI.conj(), optimize=True) / 81 ** 3
    power = np.abs(F) ** 2
    total = power.sum()
    diag = np.einsum("sss->s", power)
    top = np.argsort(power.ravel())[::-1][:12]
    named = [dict(s=S[i // 6561].tolist(), t=S[(i // 81) % 81].tolist(), u=S[i % 81].tolist(), share=float(power.ravel()[i] / total)) for i in top]
    # With all 80 other cards on the table, is the top-scoring ordered pair one that completes the set with q?
    partner = ((-(CARDS[:, None, :] + CARDS[None, :, :])) % 3 * np.array([1, 3, 9, 27])).sum(-1)  # partner[q, j]
    eye = np.eye(81, dtype=bool)
    hit = []
    for q in range(81):
        M = np.where(eye | eye[q][:, None] | eye[q][None, :], -np.inf, A[q])
        j, k = np.unravel_index(M.argmax(), M.shape)
        hit.append(partner[q, j] == k)
    return dict(variance=float(total), single_attribute_diagonal=float(diag[WEIGHT == 1].sum() / total), all_diagonal=float(diag[WEIGHT > 0].sum() / total),
                by_weight={int(k): float(diag[WEIGHT == k].sum() / total) for k in range(1, 5)}, top=named, top_pair_completes=float(np.mean(hit)))


def main(path, lr_index, out):
    P = dict(np.load(path))
    get = lambda name: P[f"['{name}']"]
    rows = []
    for seed in range(get("input").shape[1]):
        w = lambda name: get(name)[lr_index, seed]
        hq, ht = np.tanh(features(True) @ w("input")), np.tanh(features(False) @ w("input"))
        A = np.einsum("qc,jc,kc->qjk", hq @ w("q"), ht @ w("k1"), ht @ w("k2")) / R ** .5
        rows.append(dict(seed=seed, **analyse(A)))
        print(json.dumps(dict(seed=seed, single=round(rows[-1]["single_attribute_diagonal"], 3), all_diagonal=round(rows[-1]["all_diagonal"], 3))), flush=True)
    json.dump(dict(path=str(path), lr_index=lr_index, rows=rows), open(out, "w"), indent=1)


def selftest():
    clocks = sum(np.cos(2 * np.pi / 3 * (CARDS[:, None, None, a] + CARDS[None, :, None, a] + CARDS[None, None, :, a])) for a in range(4))
    r = analyse(clocks)
    assert abs(r["single_attribute_diagonal"] - 1) < 1e-9, r
    noise = np.random.default_rng(0).normal(size=(81, 81, 81))
    r2 = analyse(noise)
    print("four clocks:", round(r["single_attribute_diagonal"], 6), "noise:", round(r2["single_attribute_diagonal"], 6), "expected for noise:", round(8 / 81 ** 3, 6))


if __name__ == "__main__":
    selftest() if sys.argv[1] == "--selftest" else main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
