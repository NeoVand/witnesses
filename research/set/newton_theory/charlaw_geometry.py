"""Shared geometry of SET in the coordinates of the card in hand (charlaw.py, the characters law).

Translate every deal by the card in hand q: b_j = a_j - q. Then q + a_j + a_k = b_j + b_k (3q = 0), the table is
a set of 8 distinct non-zero vectors of (Z_3)^4, and the deal is positive iff two table vectors are opposite,
b_k = -b_j. The 80 non-zero vectors form 40 opposite pairs {v, -v}: the points of the projective space PG(3, 3).
The 40 conjugate pairs of non-trivial characters {chi_s, chi_-s} are the points of the dual PG(3, 3).

Deal distribution of the bench (tasks.generate("set") and bench8.deals): 9 distinct uniform cards, the last
one in hand, balanced 50/50 by label by rejection. In translated coordinates: negatives are uniform over the
8-subsets of the 80 vectors with no opposite pair, positives uniform over those with at least one.

Channel s sees, after translation, the histogram (n0, n1, n2) of s.b_j mod 3 over the table:
  arity 2 (phi_q e_1)  is chi_s(-q) * (n0 + n1 w + n2 w^2)     (w = exp(2 pi i / 3))
  arity 3 (phi_q e_2)  is sum over pairs of w^(s.(b_j + b_k)) = (3 M_s - 28)/2 + i (sqrt 3 / 2)(c1 - c2)
with M_s = C(n0, 2) + n1 n2 the number of table pairs with s.(b_j + b_k) = 0, c1 = n0 n1 + C(n2, 2) and
c2 = n0 n2 + C(n1, 2).
"""
import numpy as np

VEC = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
CODE = np.array([1, 3, 9, 27])
NEG = ((-VEC) % 3) @ CODE
REPS = np.array([v for v in range(1, 81) if v < NEG[v]])            # 40 representatives, one per {v, -v}
assert len(REPS) == 40
PAIR = np.full(81, -1)
for i, v in enumerate(REPS):
    PAIR[v] = PAIR[NEG[v]] = i
SIGN = np.zeros(81, int)
SIGN[REPS] = 1
SIGN[NEG[REPS]] = -1
DOT = (VEC[REPS] @ VEC[REPS].T) % 3                                   # DOT[s, i] = s . v_i, both representatives
LINES = None


def sample(rng, count, label, n=8):
    """count deals of one label, as (pair index [count, n], sign [count, n] in {+1, -1})."""
    out_i, out_e, have = [], [], 0
    while have < count:
        B = max(4096, 2 * (count - have))
        if label == 0:
            i = np.argsort(rng.random((B, 40)), axis=1)[:, :n]
            e = rng.choice(np.array([1, -1]), size=(B, n))
            keep = np.ones(B, bool)
        else:
            v = 1 + np.argsort(rng.random((B, 80)), axis=1)[:, :n]
            i, e = PAIR[v], SIGN[v]
            srt = np.sort(i, axis=1)
            keep = (srt[:, 1:] == srt[:, :-1]).any(1)
        out_i.append(i[keep]); out_e.append(e[keep]); have += int(keep.sum())
    return np.concatenate(out_i)[:count], np.concatenate(out_e)[:count]


def histograms(i, e):
    """[deals, 40 channels, 3]: counts of s . b_j = 0, 1, 2 over the table, for every character class s."""
    val = (DOT[:, i].transpose(1, 2, 0) * e[:, :, None]) % 3            # [deals, n, 40]
    return np.stack([(val == c).sum(1) for c in range(3)], -1).astype(np.int16)


def arity3(H):
    """M_s (pairs with s.(b_j+b_k) = 0) and c1 - c2 (the imaginary part, up to sqrt(3)/2), per channel."""
    n0, n1, n2 = (H[..., c].astype(np.int64) for c in range(3))
    M = n0 * (n0 - 1) // 2 + n1 * n2
    c1 = n0 * n1 + n2 * (n2 - 1) // 2
    c2 = n0 * n2 + n1 * (n1 - 1) // 2
    return M, c1 - c2


def count_sets(i):
    """N: number of opposite pairs on the table (= table pairs completing a set with the card in hand)."""
    srt = np.sort(i, axis=1)
    _, n = srt.shape
    N = np.zeros(len(i), int)
    for a in range(n):
        for b in range(a + 1, n):
            N += srt[:, a] == srt[:, b]
    return N


def selftest(rng):
    # The 40 characters reproduce N exactly: sum over all 40 classes of M_s = 27 N + 364 for 8 cards.
    for label in (0, 1):
        i, e = sample(rng, 5000, label)
        M, _ = arity3(histograms(i, e))
        N = count_sets(i)
        assert ((N > 0) == label).all()
        assert (M.sum(1) == 27 * N + 364).all(), "sum_s M_s = 27 N + 364"
    # Cross-check against the bench's own card coordinates on random deals
    q = rng.integers(0, 81, 2000)
    tab = np.array([rng.choice(np.setdiff1d(np.arange(81), [x]), 8, replace=False) for x in q])
    b = (VEC[tab] - VEC[q][:, None]) % 3
    bc = b @ CODE
    i, e = PAIR[bc], SIGN[bc]
    direct = (((VEC[q][:, None, None] + VEC[tab][:, :, None] + VEC[tab][:, None, :]) % 3 == 0).all(-1)
              & ~np.eye(8, dtype=bool)).sum((1, 2)) // 2
    assert (direct == count_sets(i)).all()
    print("geometry selftest passed")


if __name__ == "__main__":
    selftest(np.random.default_rng(0))
