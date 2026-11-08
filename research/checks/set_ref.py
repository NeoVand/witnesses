"""Exponential Newton: the softmax join over distinct unordered pairs, computed through Jacobi-Anger channels (numpy).

A query q and partners x_1..x_n each carry d clocks theta_i. The join's weight on the pair {j, k} is
    w_jk = exp(beta * sum_i (cos(theta_i(q) + theta_i(x_j) + theta_i(x_k)) - 1)),
scaled by exp(-d beta) so the largest possible weight is 1 (the softmax is unchanged). Its partition function is
Z = sum_{j<k} w_jk and its output, for the symmetric product value (v1_j * v2_k + v1_k * v2_j) / 2, is
Y = sum_{j<k} w_jk (v1_j * v2_k + v1_k * v2_j) / 2 / Z.

Channel form. Per clock, exp(beta (cos phi - 1)) = sum_n a_n e^{i n phi}. With a_n = I_n(beta) e^{-beta} for all n this is
the Jacobi-Anger expansion (exact); truncated at |n| <= F it is an approximation. On phases in (2 pi / M) Z the finite
series with aliased coefficients a_n = sum_{m = n mod M} I_m(beta) e^{-beta} (the DFT of the kernel on Z_M) is exact.
The product over clocks gives channels n in {-F..F}^d with weight W_n = prod_i a_{n_i} and per-token factor
C^n(x) = prod_i e^{i n_i theta_i(x)}, so
    Z = sum_n W_n C^n(q) e_2^n,          e_2^n = ((sum_j C^n_j)^2 - sum_j (C^n_j)^2) / 2,
    Y = sum_n W_n C^n(q) [(sum_j C^n_j v1_j) * (sum_k C^n_k v2_k) - sum_j (C^n_j)^2 v1_j * v2_j] / 2 / Z,
Newton's e_2 removing the repeated partner. Cost O(n (2F+1)^d r) per query, linear in n.

Categorical clocks: clock i of token x is a distribution p_i(x) over Z_M. The weight is the expected exponential over
independent draws, w_jk = prod_i sum_z P(a + b + c = z) exp(beta (cos(2 pi z / M) - 1)), positive for every input and
equal to the hard-clock weight when the distributions are one-hot. In channel form C^n(x) = prod_i u_i(x)_{n_i} with the
characteristic function u_i(x)_n = sum_z p_i(x)_z w^{n z}, w = e^{2 pi i / M}, and the aliased a_n; exact.
"""
import itertools
import numpy as np
from scipy.special import ive


def coef_bessel(beta, F):
    """a_n = I_n(beta) e^{-beta}, n = -F..F (truncated Jacobi-Anger)."""
    n = np.arange(-F, F + 1)
    return ive(np.abs(n), beta)


def freqs(M):
    """Representatives of Z_M centred on 0 (M = 3: -1, 0, 1)."""
    return np.arange(-((M - 1) // 2), M // 2 + 1)


def coef_alias(beta, M):
    """Exact Fourier coefficients on Z_M of exp(beta (cos(2 pi z / M) - 1)), for the frequencies freqs(M)."""
    z = np.arange(M)
    f = np.exp(beta * (np.cos(2 * np.pi * z / M) - 1))
    n = freqs(M)
    return (f[None, :] * np.exp(-2j * np.pi * np.outer(n, z) / M)).mean(1).real


def channel_weights(a_list):
    """a_list: per-clock coefficient vectors (length L each). Returns W of shape (L^d,), C-order over clocks."""
    W = np.ones(1)
    for a in a_list:
        W = np.outer(W, a).ravel()
    return W


def channel_factors_phase(theta, n):
    """theta (..., d); n frequencies (L,). Returns C (..., L^d) = prod_i e^{i n_i theta_i}."""
    d = theta.shape[-1]
    C = np.ones(theta.shape[:-1] + (1,), complex)
    for i in range(d):
        Ui = np.exp(1j * theta[..., i, None] * n)                      # (..., L)
        C = (C[..., :, None] * Ui[..., None, :]).reshape(theta.shape[:-1] + (-1,))
    return C


def channel_factors_cat(prob, M):
    """prob (..., d, M) distributions over Z_M. Returns C (..., M^d) = prod_i u_i(n_i), u = characteristic function."""
    n = freqs(M)
    z = np.arange(M)
    U = np.einsum("...iz,nz->...in", prob, np.exp(2j * np.pi * np.outer(n, z) / M))
    d = prob.shape[-2]
    C = np.ones(prob.shape[:-2] + (1,), complex)
    for i in range(d):
        C = (C[..., :, None] * U[..., i, None, :]).reshape(prob.shape[:-2] + (-1,))
    return C


def newton_join(Cq, Ct, W, v1=None, v2=None):
    """Cq (Q, K) query factors, Ct (n, K) partner factors, W (K,). Returns Z (Q,) and Y (Q, r) or None."""
    s0 = Ct.sum(0)
    e2 = (s0 * s0 - (Ct * Ct).sum(0)) / 2
    Z = ((W * e2)[None] * Cq).sum(1).real
    if v1 is None:
        return Z, None
    s1, s2 = Ct.T @ v1, Ct.T @ v2                                       # (K, r)
    p12 = (Ct * Ct).T @ (v1 * v2)
    num = (Cq * W[None]) @ (s1 * s2 - p12) / 2
    return Z, num.real / Z[:, None]


def brute_phase(theta_q, theta_t, beta, v1=None, v2=None, kernel=None):
    """Brute force over distinct unordered pairs. theta_q (Q, d), theta_t (n, d). kernel(phi) per clock, default exp."""
    if kernel is None:
        kernel = lambda phi: np.exp(beta * (np.cos(phi) - 1))
    n = theta_t.shape[0]
    j, k = np.triu_indices(n, 1)
    phi = theta_q[:, None, :] + theta_t[j][None] + theta_t[k][None]     # (Q, P, d)
    w = kernel(phi).prod(-1)                                            # (Q, P)
    Z = w.sum(1)
    if v1 is None:
        return Z, None
    pv = (v1[j] * v2[k] + v1[k] * v2[j]) / 2                            # (P, r)
    return Z, (w @ pv) / Z[:, None]


def brute_cat(prob_q, prob_t, beta, M, v1=None, v2=None):
    """Expected-exponential weights of categorical clocks, by explicit convolution over Z_M. prob_q (Q, d, M)."""
    n = prob_t.shape[0]
    j, k = np.triu_indices(n, 1)
    g = np.exp(beta * (np.cos(2 * np.pi * np.arange(M) / M) - 1))
    pj, pk = prob_t[j], prob_t[k]                                       # (P, d, M)
    # distribution of b + c
    bc = np.zeros(pj.shape)
    for b in range(M):
        for c in range(M):
            bc[..., (b + c) % M] += pj[..., b] * pk[..., c]
    w = np.ones((prob_q.shape[0], len(j)))
    for i in range(prob_q.shape[1]):
        tot = np.zeros((prob_q.shape[0], len(j), M))
        for a in range(M):
            for s in range(M):
                tot[..., (a + s) % M] += prob_q[:, None, i, a] * bc[None, :, i, s]
        w = w * (tot @ g)
    Z = w.sum(1)
    if v1 is None:
        return Z, None
    pv = (v1[j] * v2[k] + v1[k] * v2[j]) / 2
    return Z, (w @ pv) / Z[:, None]


CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])


def set_deal(rng, n, q=None):
    """A card in hand and n distinct table cards (uniform), as indices into CARDS."""
    q = rng.integers(81) if q is None else q
    others = np.setdiff1d(np.arange(81), [q])
    return q, rng.choice(others, n, replace=False)


def set_count(q, table):
    a = CARDS[table]
    s = (CARDS[q][None, None] + a[:, None] + a[None, :]) % 3
    m = (s == 0).all(-1)
    return int(np.triu(m, 1).sum())


def complement(q, c):
    """The card completing a set with q and c."""
    v = (-CARDS[q] - CARDS[c]) % 3
    return int(v @ np.array([1, 3, 9, 27]))
