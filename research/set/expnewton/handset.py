"""Check 2 (CPU): the hand-set exponential Newton layer on SET at 8 to 64 table cards (numpy). Writes handset.json.

Clocks: the four single-attribute characters, theta_i(x) = 2 pi x_i / 3 (hard Z_3 clocks), d = 4, the aliased F = 1
series (81 channels), weights scaled so the largest pair weight is 1. A set pair has weight 1; a non-set pair has at
least one clock off, so weight at most eps = exp(-1.5 beta) (three clocks on, one at cos = -1/2).

Read-out Z (partition function): "set" iff Z > 1/2. Exact when P eps < 1/2, P = C(n, 2), i.e. beta > (2/3) ln(2P).
Read-out Y (normalised output only, as the join reads it): value v_j = (chi_s(x_j))_s over the 81 characters, pair value
v_j * v_k = chi_s(x_j + x_k); r = Re sum_s chi_s(q) Y_s / 81 is the softmax share of set pairs; "set" iff r > 1/2. Exact
when N > Z_rest for every deal; guaranteed for one-set deals when P eps < 1, i.e. beta > (2/3) ln P.

Deals per size: uniform random tables, tables with no set (one card of each of n of the 40 complementary pairs, n <= 40),
and tables with exactly one set (n <= 41). For each read-out: accuracy at the rule's beta, at the fixed beta that the rule
gives at 8 cards, and the smallest beta that is exact on these deals (bisection).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from ref import coef_alias, freqs, channel_weights, channel_factors_phase, CARDS, set_deal, set_count, complement

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(1)
n3 = freqs(3)
CHI = np.exp(2j * np.pi / 3 * ((CARDS @ CARDS.T) % 3))   # CHI[x, s]


def pairs_of(q):
    seen, pairs = set(), []
    for c in range(81):
        if c != q and c not in seen:
            m = complement(q, c)
            seen |= {c, m}
            pairs.append((c, m))
    return pairs


def deal(kind, n):
    if kind == "random":
        return set_deal(rng, n)
    q = rng.integers(81)
    P = pairs_of(q)
    if kind == "none":
        pick = rng.choice(40, n, replace=False)
        return q, np.array([P[i][rng.integers(2)] for i in pick])
    pick = rng.choice(40, n - 1, replace=False)                 # one full pair, n - 2 singles
    table = list(P[pick[0]]) + [P[i][rng.integers(2)] for i in pick[1:]]
    return q, np.array(table)


def features(q, table):
    """Per deal: query factors (1, 81), partner factors (n, 81), and the output read-out's sums."""
    Cq = channel_factors_phase(2 * np.pi * CARDS[[q]] / 3, n3)
    Ct = channel_factors_phase(2 * np.pi * CARDS[table] / 3, n3)
    s0 = Ct.sum(0)
    e2 = (s0 * s0 - (Ct * Ct).sum(0)) / 2                      # (81,)
    V = CHI[table]                                              # (n, 81 value dims)
    s1 = Ct.T @ V
    num = (s1 * s1 - (Ct * Ct).T @ (V * V)) / 2                 # (81 channels, 81 value dims)
    return Cq[0], e2, num, CHI[q]


def readouts(f, beta):
    Cq, e2, num, chiq = f
    W = channel_weights([coef_alias(beta, 3)] * 4)
    Z = (W * Cq * e2).sum().real
    Y = ((W * Cq) @ num) / Z                                    # complex value dims
    r = (chiq * Y).sum().real / 81
    return Z, r


sizes = [8, 16, 20, 32, 40, 64]
rows = []
for n in sizes:
    P = n * (n - 1) // 2
    kinds = ["random"] + (["none"] if n <= 40 else []) + (["one"] if n <= 41 else [])
    deals = [(k, deal(k, n)) for k in kinds for _ in range(500)]
    labels = np.array([set_count(q, t) > 0 for _, (q, t) in deals])
    counts = np.array([set_count(q, t) for _, (q, t) in deals])
    feats = [features(q, t) for _, (q, t) in deals]

    def acc(beta):
        out = np.array([readouts(f, beta) for f in feats])
        return float(((out[:, 0] > .5) == labels).mean()), float(((out[:, 1] > .5) == labels).mean()), out

    rule_Z, rule_Y = (2 / 3) * np.log(2 * P), (2 / 3) * np.log(P)
    fixed_Z, fixed_Y = (2 / 3) * np.log(2 * 28), (2 / 3) * np.log(28)
    row = dict(n=n, pairs=P, deals=len(deals), positives=int(labels.sum()), kinds=kinds,
               rule_beta_Z=rule_Z, rule_beta_Y=rule_Y)
    aZ, _, out = acc(rule_Z)
    _, aY, outY = acc(rule_Y)
    row["acc_rule"] = dict(Z=aZ, Y=aY)
    exact_err = max(abs(o[0] - c) for o, c in zip(acc(40.)[2], counts))  # at large beta Z -> N (float64)
    row["Z_minus_N_at_beta40"] = float(exact_err)
    aZf, _, _ = acc(fixed_Z)
    _, aYf, _ = acc(fixed_Y)
    row["acc_fixed8"] = dict(Z=aZf, Y=aYf, beta_Z=fixed_Z, beta_Y=fixed_Y)
    for which in (0, 1):
        lo, hi = 0.05, 12.                                      # above ~12 float64 cancellation, not the rule, decides
        for _ in range(40):
            mid = (lo + hi) / 2
            a = acc(mid)[which]
            lo, hi = (lo, mid) if a == 1. else (mid, hi)
        row["beta_min_" + "ZY"[which]] = hi
    neg = [o for o, l in zip(out, labels) if not l]
    row["max_negative_Z_at_rule"] = float(max(o[0] for o in neg)) if neg else None
    rows.append(row)
    print(json.dumps(dict(n=n, P=P, acc_rule=row["acc_rule"], acc_fixed8=row["acc_fixed8"],
                          beta_rule=(round(rule_Z, 2), round(rule_Y, 2)), beta_min=(round(row["beta_min_Z"], 2), round(row["beta_min_Y"], 2)),
                          Z_minus_N=f"{exact_err:.1e}")), flush=True)
(HERE / "handset.json").write_text(json.dumps(rows, indent=1))
