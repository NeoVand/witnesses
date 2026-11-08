"""Checks of the character count (Proposition 2, and its general form over a finite abelian group G and subsets of
size k): exactness, precision, and where distinctness matters.

1. Exactness: the character formula with e_k from power sums by Newton's identities reproduces the exact count of
   k-subsets of partners with q + sum = 0 (integer dynamic programme), for SET (Z_3^4, k = 2), four-card SET (k = 3),
   Match-3 and Match-4 modulo 31, and a general h (random real h on Z_3^4, k = 2).
2. Precision: in float32 and float64, the largest error of the computed count against n, next to the bound of
   Proposition N1-P, |delta e_k| <= H_k(n + eta) - H_k(n), H_k(x) = x (x+1) ... (x+k-1) / k!, with eta the measured
   error of the power sums.
3. Distinctness: the sum over ordered pairs with repetition (p_1^2, what a linearised 2-simplicial layer with tied
   features computes) equals 2N on SET, because q + 2a = 0 forces a = q; it does not on four-card SET (p_1^3 / 6 vs e_3)
   or on Match-3 modulo 31 with repeated residues.

python -I check_count.py check_count.json   (numpy)
"""
import json, math, sys
import numpy as np


def group(kind):
    if kind == "Z3^4":
        els = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])
        return els, 3
    M = int(kind[2:])
    return np.arange(M)[:, None], M


def chars(els, mod, dtype):
    """chi[s, x] = exp(2 pi i <s, x> / mod), all |G| characters, s indexed like the elements."""
    ph = (els @ els.T) % mod
    return np.exp(2j * np.pi * ph / mod).astype(dtype)


def tables(els, mod):
    code = np.array([mod ** i for i in range(els.shape[1])])
    sub = ((els[:, None, :] - els[None, :, :]) % mod) @ code     # sub[g, x] = index of g - x
    neg = ((-els) % mod) @ code
    return sub, neg


def exact_count(a, q, k, T):
    """number of k-subsets of indices with q + sum = 0, by a dynamic programme over partners (Python integers)."""
    sub, neg = T
    G = len(neg)
    cnt = np.zeros((k + 1, G), dtype=object)
    cnt[0, 0] = 1
    for x in a:
        for m in range(k, 0, -1):
            cnt[m] = cnt[m] + cnt[m - 1][sub[:, x]]   # m-subsets with sum g: new ones are (m-1)-subsets with sum g - x
    return int(cnt[k][neg[q]])


def newton_e(p, k):
    e = [np.ones_like(p[0])]
    for m in range(1, k + 1):
        e.append(sum((-1) ** (l - 1) * e[m - l] * p[l - 1] for l in range(1, m + 1)) / m)
    return e


def count_by_characters(chi, a, q, k, h_hat=None):
    """sum over |S| = k of h(q + sum_S a) = sum_s h_hat(s) chi_s(q) e_k(chi_s(a)); default h = indicator of 0."""
    x = chi[:, a]                                   # [|G|, n]
    p = [(x ** m).sum(1) for m in range(1, k + 1)]  # power sums, in the dtype of chi
    e = newton_e(p, k)[k]
    G = chi.shape[0]
    w = np.full(G, 1.0 / G) if h_hat is None else h_hat
    return (w * chi[:, q] * e).sum(), p


def H(k, x):
    return math.prod(x + i for i in range(k)) / math.factorial(k)


def main(dest):
    rng = np.random.default_rng(0)
    out = dict(exact=[], precision=[], distinct=[])
    for kind, k, distinct in (("Z3^4", 2, True), ("Z3^4", 3, True), ("Z_31", 2, False), ("Z_31", 3, False)):
        els, mod = group(kind)
        G = len(els)
        T = tables(els, mod)
        chi64 = chars(els, mod, np.complex128)
        worst = 0.0
        for trial in range(200):
            n = int(rng.integers(3, 12))
            q = int(rng.integers(G))
            a = (rng.choice(np.setdiff1d(np.arange(G), [q]), n, replace=False) if distinct
                 else rng.integers(0, G, n))
            ex = exact_count(a, q, k, T)
            val, _ = count_by_characters(chi64, a, q, k)
            worst = max(worst, abs(val - ex))
        out["exact"].append(dict(group=kind, k=k, trials=200, max_error=float(worst)))
        print(kind, k, "max |formula - exact count|", worst, flush=True)
    # a general real h on Z_3^4, k = 2
    els, mod = group("Z3^4")
    chi = chars(els, mod, np.complex128)
    h = rng.normal(size=81)
    h_hat = (chi.conj() @ h) / 81                   # h_hat(s) = (1/|G|) sum_y h(y) conj chi_s(y)
    code = np.array([1, 3, 9, 27])
    worst = 0.0
    for trial in range(200):
        n = int(rng.integers(3, 12)); q = int(rng.integers(81)); a = rng.integers(0, 81, n)
        direct = sum(h[((els[q] + els[a[i]] + els[a[j]]) % 3) @ code] for i in range(n) for j in range(i + 1, n))
        val, _ = count_by_characters(chi, a, q, 2, h_hat)
        worst = max(worst, abs(val - direct))
    out["exact"].append(dict(group="Z3^4", k=2, h="random real h", trials=200, max_error=float(worst)))
    print("general h, max error", worst)

    # precision: float32 and float64, n up to 2^16, partners with repetition on Z_3^4 and Z_31
    for kind in ("Z3^4", "Z_31"):
        els, mod = group(kind)
        G = len(els)
        T = tables(els, mod)
        for k in (2, 3):
            for n in (8, 64, 512, 4096, 32768):
                if k == 3 and n > 4096:
                    continue
                for dtype, rdt in ((np.complex64, np.float32), (np.complex128, np.float64)):
                    chi = chars(els, mod, dtype)
                    chi_ref = chars(els, mod, np.complex128)
                    errs, etas = [], []
                    for trial in range(20):
                        q = int(rng.integers(G)); a = rng.integers(0, G, n)
                        ex = exact_count(a, q, k, T)
                        val, p = count_by_characters(chi, a, q, k)
                        hist = np.bincount(a, minlength=G).astype(np.float64)   # exact integer counts
                        pref = [(chi_ref ** m) @ hist for m in range(1, k + 1)]  # |G| terms, not n
                        etas.append(max(float(np.abs(np.asarray(pp, np.complex128) - pr).max()) for pp, pr in zip(p, pref)))
                        errs.append(abs(complex(val) - ex))
                    eta = max(etas)
                    out["precision"].append(dict(group=kind, k=k, n=n, dtype=np.dtype(rdt).name, max_count_error=float(max(errs)),
                                                 power_sum_error=eta, bound_on_e_k=H(k, n + eta) - H(k, n),
                                                 rounds_correctly=bool(max(errs) < 0.5)))
                    print(kind, k, n, np.dtype(rdt).name, "count err", max(errs), "eta", eta,
                          "bound", H(k, n + eta) - H(k, n), flush=True)

    # distinctness: ordered pairs with repetition
    els, mod = group("Z3^4"); chi = chars(els, mod, np.complex128)
    T = tables(els, mod)
    bad2 = bad3 = 0
    for trial in range(500):
        q = int(rng.integers(81)); a = rng.choice(np.setdiff1d(np.arange(81), [q]), 8, replace=False)
        x = chi[:, a]
        p1 = x.sum(1)
        N2 = exact_count(a, q, 2, T); N3 = exact_count(a, q, 3, T)
        ordered2 = ((chi[:, q] * p1 ** 2).sum() / 81).real          # sum over ordered pairs (j, k), j = k allowed
        ordered3 = ((chi[:, q] * p1 ** 3).sum() / 81).real
        bad2 += abs(ordered2 - 2 * N2) > 1e-6
        bad3 += abs(ordered3 - 6 * N3) > 1e-6
    out["distinct"].append(dict(task="SET, 8 distinct cards", deals=500, ordered_pairs_neq_2N=int(bad2)))
    out["distinct"].append(dict(task="four-card SET, 8 distinct cards", deals=500, ordered_triples_neq_6N=int(bad3)))
    els, mod = group("Z_31"); chi = chars(els, mod, np.complex128)
    T = tables(els, mod)
    bad = 0
    for trial in range(500):
        q = int(rng.integers(31)); a = rng.integers(0, 31, 8)
        N2 = exact_count(a, q, 2, T)
        ordered2 = ((chi[:, q] * chi[:, a].sum(1) ** 2).sum() / 31).real
        bad += abs(ordered2 - 2 * N2) > 1e-6
    out["distinct"].append(dict(task="Match-3 mod 31, 8 partners with repetition", deals=500, ordered_pairs_neq_2N=int(bad)))
    print(out["distinct"])
    json.dump(out, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
