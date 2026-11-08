"""
Finite groups: cyclic Z_n, the alternating groups A_4 and A_5 and the symmetric group S_5,
with multiplication tables, conjugacy classes and character tables, the ordered Match-k task and the word
problem (prefix products) over a group, and the analysis that reads which irreducible representations a
trained holonomy layer holds. hoa_polybench.py imports it for its group tasks.

Conventions: elements are indices 0..|G|-1, mul[a, b] is the index of a*b, and a sequence's product is
taken in position order, g_1 g_2 ... g_k. Permutations compose as (a*b)(x) = a(b(x)).
"""

import itertools
import math

import numpy as np


class Group:
    def __init__(self, name, mul, e, classes, chars, dims):
        self.name, self.mul, self.e = name, mul, e
        self.n = mul.shape[0]
        self.inv = np.argmax(mul == e, axis=1)
        self.classes = classes          # list of index arrays
        self.chars = chars              # [irreps, classes] complex, the character table
        self.dims = np.asarray(dims)    # irrep dimensions, trivial first
        cls = np.zeros(self.n, int)
        for c, members in enumerate(classes):
            cls[members] = c
        self.cls = cls                  # class of every element
        self.char_el = chars[:, cls]    # [irreps, elements]

    @property
    def abelian(self):
        return bool((self.mul == self.mul.T).all())


def cyclic(n):
    a = np.arange(n)
    mul = (a[:, None] + a[None, :]) % n
    classes = [np.array([k]) for k in range(n)]
    chars = np.exp(2j * np.pi * np.outer(a, a) / n)  # chi_j(k) = exp(2 pi i j k / n)
    return Group(f"Z{n}", mul, 0, classes, chars, [1] * n)


def _alternating(m):
    perms = [p for p in itertools.permutations(range(m)) if _parity(p) == 0]
    index = {p: i for i, p in enumerate(perms)}
    mul = np.array([[index[tuple(a[b[x]] for x in range(m))] for b in perms] for a in perms])
    e = index[tuple(range(m))]
    return perms, index, mul, e


def _parity(p):
    seen, par = set(), 0
    for s in range(len(p)):
        if s in seen:
            continue
        L, x = 0, s
        while x not in seen:
            seen.add(x); x = p[x]; L += 1
        par ^= (L - 1) & 1
    return par


def _classes(mul, inv):
    n = mul.shape[0]
    left, out = set(range(n)), []
    while left:
        g = min(left)
        members = sorted({int(mul[mul[h, g], inv[h]]) for h in range(n)})
        out.append(np.array(members))
        left -= set(members)
    return out


def _cycle_type(p):
    seen, t = set(), []
    for s in range(len(p)):
        if s in seen:
            continue
        L, x = 0, s
        while x not in seen:
            seen.add(x); x = p[x]; L += 1
        t.append(L)
    return tuple(sorted(t, reverse=True))


def alternating4():
    perms, index, mul, e = _alternating(4)
    inv = np.argmax(mul == e, axis=1)
    cl = _classes(mul, inv)
    by_type = {}
    for c in cl:
        by_type.setdefault(_cycle_type(perms[c[0]]), []).append(c)
    three = by_type[(3, 1)]                        # the 3-cycles split into two classes of 4
    c = three[0][0]
    a, b = (three[0], three[1]) if mul[c, c] in three[1] else (three[1], three[0])
    classes = [by_type[(1, 1, 1, 1)][0], by_type[(2, 2)][0], a, b]   # e, (12)(34), c, c^2
    w = np.exp(2j * np.pi / 3)
    chars = np.array([[1, 1, 1, 1], [1, 1, w, w * w], [1, 1, w * w, w], [3, -1, 0, 0]], complex)
    return Group("A4", mul, e, classes, chars, [1, 1, 1, 3])


def alternating5():
    perms, index, mul, e = _alternating(5)
    inv = np.argmax(mul == e, axis=1)
    cl = _classes(mul, inv)
    by_type = {}
    for c in cl:
        by_type.setdefault(_cycle_type(perms[c[0]]), []).append(c)
    five = by_type[(5,)]                            # the 5-cycles split into two classes of 12
    c = five[0][0]
    a, b = (five[0], five[1]) if mul[c, c] in five[1] else (five[1], five[0])
    classes = [by_type[(1, 1, 1, 1, 1)][0], by_type[(2, 2, 1)][0], by_type[(3, 1, 1)][0], a, b]
    phi = (1 + math.sqrt(5)) / 2
    chars = np.array([[1, 1, 1, 1, 1],
                      [3, -1, 0, phi, 1 - phi],
                      [3, -1, 0, 1 - phi, phi],
                      [4, 0, 1, -1, -1],
                      [5, 1, -1, 0, 0]], complex)
    return Group("A5", mul, e, classes, chars, [1, 3, 3, 4, 5])


def symmetric5():
    perms = list(itertools.permutations(range(5)))
    index = {p: i for i, p in enumerate(perms)}
    mul = np.array([[index[tuple(a[b[x]] for x in range(5))] for b in perms] for a in perms])
    e = index[tuple(range(5))]
    # in S_n the conjugacy classes are the cycle types
    order = [(1, 1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1), (3, 1, 1), (3, 2), (4, 1), (5,)]
    types = [_cycle_type(p) for p in perms]
    classes = [np.array([i for i, t in enumerate(types) if t == c]) for c in order]
    # rows: trivial, sign, standard 4, 4 x sign, 5, 5 x sign, 6 = wedge^2 of the standard
    chars = np.array([[1, 1, 1, 1, 1, 1, 1],
                      [1, -1, 1, 1, -1, -1, 1],
                      [4, 2, 0, 1, -1, 0, -1],
                      [4, -2, 0, 1, 1, 0, -1],
                      [5, 1, 1, -1, 1, -1, 0],
                      [5, -1, 1, -1, -1, 1, 0],
                      [6, 0, -2, 0, 0, 0, 1]], complex)
    G = Group("S5", mul, e, classes, chars, [1, 1, 4, 4, 5, 5, 6])
    G.perms = perms
    return G


def _with_perms(make, m, even):
    def build():
        G = make()
        G.perms = [p for p in itertools.permutations(range(m)) if not even or _parity(p) == 0]
        return G
    return build


GROUPS = {"Z12": lambda: cyclic(12), "A4": _with_perms(alternating4, 4, True), "Z60": lambda: cyclic(60),
          "A5": _with_perms(alternating5, 5, True), "S5": symmetric5}
_cache = {}


def get(name):
    if name not in _cache:
        _cache[name] = GROUPS[name]()
    return _cache[name]


def ordered_product(G, g):
    """g: [..., k] element indices -> [...] index of g_1 g_2 ... g_k."""
    p = g[..., 0]
    for t in range(1, g.shape[-1]):
        p = G.mul[p, g[..., t]]
    return p


def match_labels(G, g, k):
    """g: [B, n] -> [B, n] bool: position i lies in some k positions i_1 < ... < i_k whose ordered product is e."""
    B, n = g.shape
    y = np.zeros((B, n), bool)
    for comb in itertools.combinations(range(n), k):
        hit = ordered_product(G, g[:, list(comb)]) == G.e
        if hit.any():
            y[:, list(comb)] |= hit[:, None]
    return y


def task_gmatch(rng, B, name, n, k=3):
    """Ordered Match-k over a group: a BOS token (index |G|), then n uniform elements; every element is
    scored on whether it completes some position-ordered k-product equal to the identity."""
    G = get(name)
    g = rng.integers(0, G.n, size=(B, n))
    y = match_labels(G, g, k).astype(np.int32)
    tok = np.concatenate([np.full((B, 1), G.n), g], 1)[..., None]
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, n))], 1)
    return tok, tgt, msk


def swap_program_labels(instructions, query, variables=4, namespaces=2):
    """Execute swap instructions in independent namespaces and read one queried variable.

    Instruction id = namespace * binom(variables, 2) + lexicographic unordered pair id.
    Query id = namespace * variables + variable. Each namespace starts as [0, ..., variables-1].
    """
    instructions, query = np.asarray(instructions), np.asarray(query)
    pairs = np.asarray(list(itertools.combinations(range(variables), 2)))
    B = len(instructions)
    state = np.broadcast_to(np.arange(variables), (B, namespaces, variables)).copy()
    rows = np.arange(B)
    for t in range(instructions.shape[1]):
        ids = instructions[:, t]
        key, pair = ids // len(pairs), pairs[ids % len(pairs)]
        a, b = pair[:, 0], pair[:, 1]
        old = state[rows, key, a].copy()
        state[rows, key, a] = state[rows, key, b]
        state[rows, key, b] = old
    return state[rows, query // variables, query % variables].astype(np.int32)


def task_swap_program(rng, B, per_namespace=3, variables=4, namespaces=2):
    """BOS, an interleaving of keyed swaps, then a query; only its four-way answer is scored.

    Exactly per_namespace independently uniform swaps per namespace; their interleaving and the
    queried namespace/variable are uniform. All arms get the identical categorical token stream.
    """
    pairs = math.comb(variables, 2)
    keys = np.broadcast_to(np.repeat(np.arange(namespaces), per_namespace), (B, namespaces * per_namespace))
    keys = np.take_along_axis(keys, np.argsort(rng.random(keys.shape), axis=1), axis=1)
    instructions = keys * pairs + rng.integers(pairs, size=keys.shape)
    query = rng.integers(namespaces * variables, size=B)
    bos = namespaces * (pairs + variables)
    tokens = np.concatenate([np.full((B, 1), bos), instructions, (namespaces * pairs + query)[:, None]], axis=1)
    targets = np.zeros_like(tokens, dtype=np.int32)
    targets[:, -1] = swap_program_labels(instructions, query, variables, namespaces)
    mask = np.zeros_like(tokens, dtype=np.float32)
    mask[:, -1] = 1
    return tokens[..., None], targets, mask


def swap_order_blind_ceiling(per_namespace=3, variables=4):
    """Exact accuracy ceiling from the instruction multiset and query, as an integer ratio.

    Distractor namespaces carry no information. Uniform queried variables and permutation symmetry
    make the target classes equiprobable, so accuracy and balanced accuracy have the same Bayes optimum.
    """
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    pairs = math.comb(variables, 2)
    for seq in itertools.product(range(pairs), repeat=per_namespace):
        instructions = np.broadcast_to(seq, (variables, per_namespace))
        labels = swap_program_labels(instructions, np.arange(variables), variables, 1)
        for q, y in enumerate(labels):
            counts[(tuple(sorted(seq)), q)][int(y)] += 1
    return sum(max(c.values()) for c in counts.values()), variables * pairs ** per_namespace


def prefix_products(G, g):
    """g: [B, T] -> [B, T]: entry t is the index of g_1 g_2 ... g_t (the word problem's answers)."""
    p = np.empty_like(g)
    p[:, 0] = g[:, 0]
    for t in range(1, g.shape[1]):
        p[:, t] = G.mul[p[:, t - 1], g[:, t]]
    return p


def fixes_last(G, idx):
    """[|G|] bool: the permutation fixes its last letter, i.e. lies in the point stabiliser (A4 in A5, S4 in S5)."""
    m = len(G.perms[0])
    return np.array([G.perms[i][m - 1] == m - 1 for i in range(G.n)])[idx]


def task_word(rng, B, name, T, kind="class"):
    """
    The word problem over a group: a BOS token (index |G|), then T uniform elements;
    position t is scored on the running product g_1 ... g_t, as one of |G| classes ('class') or as whether
    it fixes the last letter ('stab': membership of the point stabiliser, index 5 in A5 and S5, so a
    positive rate of 1/5). Both are NC1-complete for A5 and S5: the stabiliser's coset action on 5 points is
    faithful, so its automaton's transition group is the whole group.
    """
    G = get(name)
    g = rng.integers(0, G.n, size=(B, T))
    pp = prefix_products(G, g)
    y = pp if kind == "class" else fixes_last(G, pp).astype(np.int32)
    tok = np.concatenate([np.full((B, 1), G.n), g], 1)[..., None]
    tgt = np.concatenate([np.zeros((B, 1), np.int32), y.astype(np.int32)], 1)
    msk = np.concatenate([np.zeros((B, 1)), np.ones((B, T))], 1)
    return tok, tgt, msk


def word_commuting_ceiling(name, T, kind="class", samples=1000, perms=256, seed=0):
    """
    The best accuracy, per position, of a predictor that sees the position's element g_t, the multiset of
    the elements before it and the position, but not their order: one causal layer whose features commute
    (Newton attention, holonomy attention at d = 2). Given that information every ordering of the prefix is
    equally likely, so the guess (the most frequent answer over `perms` random reorderings) is scored against
    the true answer, an unbiased plug-in estimate (slightly below the ceiling where the guess is noisy).
    'class' returns per-position accuracies; 'stab' returns per-position accuracies at the best single
    threshold on P(label 1) and the pooled balanced accuracy.
    """
    G = get(name)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, G.n, size=(samples, T))
    true = prefix_products(G, g)
    acc, P, Y = np.zeros(T), np.zeros((samples, T)), np.zeros((samples, T), bool)
    for t in range(T):
        if t == 0:
            prods = np.repeat(g[:, :1], perms, 1)
        else:
            order = np.argsort(rng.random((samples, perms, t)), -1)
            pre = np.take_along_axis(np.repeat(g[:, None, :t], perms, 1), order, -1)
            q = pre[..., 0]
            for s in range(1, t):
                q = G.mul[q, pre[..., s]]
            prods = G.mul[q, g[:, t:t + 1]]
        if kind == "class":
            counts = np.zeros((samples, G.n), int)
            np.add.at(counts, (np.repeat(np.arange(samples), perms), prods.ravel()), 1)
            acc[t] = (counts.argmax(1) == true[:, t]).mean()
        else:
            P[:, t] = fixes_last(G, prods).mean(1)
            Y[:, t] = fixes_last(G, true[:, t])
    if kind == "class":
        return {"per_pos": acc, "mean": float(acc.mean())}
    best, per = 0.0, None
    for th in np.unique(np.round(P, 3)):
        pred = P >= th
        bal = 0.5 * (pred[Y].mean() + (~pred[~Y]).mean())
        if bal > best:
            best, per = bal, (pred == Y).mean(0)
    return {"per_pos": per, "bal": float(best)}


def order_blind_ceiling(name, n, k=3, samples=4000, perms=64, seed=0):
    """
    The best balanced accuracy of any predictor that sees a position's element and the multiset of the others
    but not the order (Newton attention, or holonomy attention with commuting rotations). For every position
    it estimates p = P(label 1 | own element, multiset) by relabelling random reorderings of the sequence, then
    scores the best threshold on p against the true labels. Monte Carlo, so slightly optimistic.
    """
    G = get(name)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, G.n, size=(samples, n))
    y = match_labels(G, g, k)
    p = np.zeros((samples, n))
    for _ in range(perms):
        P = np.argsort(rng.random((samples, n)), 1)              # P[s, q] = original position now at q
        yh = match_labels(G, np.take_along_axis(g, P, 1), k)
        np.add.at(p, (np.arange(samples)[:, None], P), yh)
    p /= perms
    best = 0.0
    for t in np.unique(np.round(p, 4)):
        pred = p >= t
        bal = 0.5 * (pred[y].mean() + (~pred[~y]).mean())
        best = max(best, bal)
    return float(best)


def plancherel(G, captured):
    """Share of the nontrivial Plancherel measure held by the irreps in `captured` (indices, trivial = 0)."""
    d = G.dims
    return float(sum(d[i] ** 2 for i in set(captured) if i != 0) / (G.n - 1))


def irrep_det(G, i):
    """det rho_i(g) for every element g, from the characters of the powers of g: the eigenvalue w^j of
    rho_i(g) (w = exp(2 pi i / ord g)) has multiplicity (1 / ord g) sum_k chi_i(g^k) w^(-jk)."""
    out = np.zeros(G.n, complex)
    for g in range(G.n):
        pw = [G.e]
        while (nxt := G.mul[pw[-1], g]) != G.e:
            pw.append(nxt)
        n = len(pw)
        chi = G.char_el[i][pw]
        w = np.exp(2j * np.pi / n)
        m = [np.round((chi * w ** (-j * np.arange(n))).sum().real / n) for j in range(n)]
        out[g] = np.prod([w ** (j * mj) for j, mj in enumerate(m)])
    return out


def householders_needed(G):
    """
    The fewest (generalised) Householders per token with which one exact matrix representation of G can be
    a product of per-token transitions. I - beta k k^T fixes the hyperplane k-perp, so a product of r of them
    fixes a subspace of codimension <= r; rho(g) fixes the multiplicity of eigenvalue 1, (1/ord g) sum_k
    chi(g^k). Codimensions add over constituents and a faithful representation contains a faithful irrep,
    so the answer is the minimum over faithful irreps of the largest codimension of a fixed space.
    """
    best = None
    for i in range(1, len(G.dims)):
        if (np.isclose(G.char_el[i], G.dims[i])).sum() != 1:
            continue                                     # not faithful
        worst = 0
        for g in range(G.n):
            pw = [G.e]
            while (nxt := G.mul[pw[-1], g]) != G.e:
                pw.append(nxt)
            fix = G.char_el[i][pw].real.mean()
            worst = max(worst, int(round(G.dims[i] - fix)))
        best = worst if best is None else min(best, worst)
    return best


def reachable(G, d):
    """
    Nontrivial irreps a head of rotations in SO(d) can hold. A real irrep rho needs d_rho dimensions if
    det rho = 1, and d_rho + 1 if not (rho + det rho, whose determinant is det^2 = 1); a complex
    one-dimensional irrep needs a conjugate pair, a 2 x 2 rotation. (A4, A5 and S5 have no quaternionic
    irreps, so every real character is realisable over the reals.)
    """
    out = []
    for i in range(1, len(G.dims)):
        real = np.allclose(G.chars[i].imag, 0)
        if not real:
            need = 2 * G.dims[i]
        else:
            need = G.dims[i] + (0 if np.allclose(irrep_det(G, i), 1) else 1)
        if need <= d:
            out.append(i)
    return out


def rep_content(G, U, tol=0.5):
    """
    U: [H, |G|, d, d], the matrices one layer assigns to every group element (one per head).
    Returns per head the homomorphism error (mean ||U_a U_b - U_ab||_F / ||U_ab||_F over all pairs) and the
    irrep multiplicities <chi, chi_rho> of its character chi(g) = tr U_g, and the set of nontrivial irreps
    held with multiplicity >= 1/2 by some head whose homomorphism error is below tol.
    """
    U = np.asarray(U, np.float64)
    H, n, d, _ = U.shape
    prod = np.einsum("hacd,hbde->habce", U, U)            # [H, a, b, d, d]
    want = U[:, G.mul]                                    # [H, a, b, d, d]
    err = np.linalg.norm(prod - want, axis=(-2, -1)) / np.maximum(np.linalg.norm(want, axis=(-2, -1)), 1e-9)
    herr = err.mean((1, 2))
    chi = np.trace(U, axis1=-2, axis2=-1)                 # [H, |G|]
    mult = (chi @ G.char_el.conj().T).real / n            # [H, irreps]
    good = herr < tol
    held = sorted({int(i) for h in range(H) if good[h] for i in np.nonzero(mult[h] >= 0.5)[0] if i != 0})
    best = np.where(good[:, None], mult, -np.inf).max(0) if good.any() else np.full(mult.shape[1], np.nan)
    top = np.argsort(herr)[:4]
    return {"hom_err_q": [round(float(q), 3) for q in np.quantile(herr, [0.0, 0.1, 0.5, 0.9])],
            "good_heads": int(good.sum()), "of": H, "held": held, "mass": round(plancherel(G, held), 3),
            "best_mult": [None if not np.isfinite(b) else round(float(b), 2) for b in best],
            "top": [{"hom_err": round(float(herr[h]), 3), "mult": np.round(mult[h], 2).tolist()} for h in top]}
