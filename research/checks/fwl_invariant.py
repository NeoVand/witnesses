"""Theorem 1(ii), the window invariant: equal invariants with equal censuses give equal 2-FWL colours.

F_l(x, y), with R = 2^l - 1 and lam(i) the label of position i ('A' off the path, 'g', 'c', or '.'):
  near, |y - x| <= 2^l : ('N', y - x, lam on [min(x, y) - R, max(x, y) + R], read from x)
  far,  |y - x| >  2^l : ('F', lam on [x - R, x + R], lam on [y - R, y + R])
The proof (Proofs, Theorem 1(ii)) shows: if the window censuses of the two graphs agree at radius 2^(l-1) - 1, equal
F_l gives equal 2-FWL colour after l rounds (soundness). This script tests soundness on every pair of both graphs and
every round, records whether the colour is also injective on F_l (exactness), compares the prediction for (c, v)
with the observed same/different pattern (fwl_sym.py's six instances per L and a grid of goal distances), tests
random marker placements, and probes the path length the proof needs.

python -I fwl_invariant.py L_min L_max
"""
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fwl import graph, joint_refine


def labeller(n, a, goal):
    def lam(i):
        if i < 0 or i >= n:
            return 'A'
        return (('g' if i == goal else '') + ('c' if i == a else '')) or '.'
    return lam


def F(lam, l, x, y):
    T = 2 ** l; R = T - 1; d = y - x
    if abs(d) <= T:
        return ('N', d, tuple(lam(x + t) for t in range(min(0, d) - R, max(0, d) + R + 1)))
    return ('F', tuple(lam(x + t) for t in range(-R, R + 1)), tuple(lam(y + t) for t in range(-R, R + 1)))


def census(lam, n, r):
    return Counter(tuple(lam(k + t) for t in range(-r, r + 1)) for k in range(n))


def setup(L, D, E):
    """fwl_sym.py's instance: B = max(D, E) + 1, M = 2B + 2^(L+1) + 6 edges, c = a = B + 2^L + 2, v = a + 1."""
    B = max(D, E) + 1; M = 2 * B + 2 ** (L + 1) + 6; a = B + 2 ** L + 2
    return M, a, a + 1 + D, a + 1 - E


def full_check(L, M, a1, g1, a2, g2):
    """Every pair of both graphs, every round l = 0..L. Per round: (census equal at radius 2^(l-1) - 1, colour is a
    function of F_l [soundness], F_l is a function of colour [exactness]); plus (c, v) observed and predicted."""
    n = M + 1; lams = [labeller(n, a1, g1), labeller(n, a2, g2)]; rounds = []
    for l in range(L + 1):
        cols = joint_refine([graph(M, a1, g1), graph(M, a2, g2)], l)
        f2c, c2f = defaultdict(set), defaultdict(set)
        for lam, col in zip(lams, cols):
            for x in range(n):
                for y in range(n):
                    f = F(lam, l, x, y); c = int(col[x, y])
                    f2c[f].add(c); c2f[c].add(f)
        cens = l == 0 or census(lams[0], n, 2 ** (l - 1) - 1) == census(lams[1], n, 2 ** (l - 1) - 1)
        rounds.append((cens, all(len(v) == 1 for v in f2c.values()), all(len(v) == 1 for v in c2f.values())))
    same = bool(cols[0][a1, a1 + 1] == cols[1][a2, a2 + 1]) if a1 + 1 < n and a2 + 1 < n else None
    return rounds, same, predict(lams, n, L, a1, a2)


def predict(lams, n, L, a1, a2):
    """F_L differ -> 'diff' (exactness); F_L equal and censuses equal -> 'same' (soundness)."""
    if a1 + 1 >= n or a2 + 1 >= n:
        return None
    if F(lams[0], L, a1, a1 + 1) != F(lams[1], L, a2, a2 + 1):
        return 'diff'
    r = 2 ** (L - 1) - 1
    return 'same' if census(lams[0], n, r) == census(lams[1], n, r) else '?'


def verdict(rounds):
    """Soundness is required wherever the census agrees; exactness everywhere."""
    ok2 = all(snd for cens, snd, _ in rounds if cens); ok3 = all(ex for _, _, ex in rounds)
    note = [l for l, (cens, snd, _) in enumerate(rounds) if not cens and not snd]
    return ok2, ok3, note


def cv_only(L, M, a, g1, g2):
    n = M + 1; lams = [labeller(n, a, g1), labeller(n, a, g2)]
    c1, c2 = joint_refine([graph(M, a, g1), graph(M, a, g2)], L)
    return ('same' if c1[a, a + 1] == c2[a, a + 1] else 'diff'), predict(lams, n, L, a, a)


random.seed(0)
for L in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    T = 2 ** L
    # 1. fwl_sym.py's six instances: every pair, every round.
    for D in (T - 1, T, T + 1):
        for E in (D, D + 1):
            M, a, g1, g2 = setup(L, D, E)
            rounds, same, pred = full_check(L, M, a, g1, a, g2)
            ok2, ok3, note = verdict(rounds)
            print(f"L={L} D={D} E={E} n={M + 1}: (c,v) observed {'same' if same else 'diff'}, predicted {pred}; "
                  f"soundness holds={ok2} exactness holds={ok3}; rounds with census differing and colour not a function of F:"
                  f" {note}", flush=True)
    # 2. (c, v) on a grid of goal distances (E = 1 would put the goal on c).
    res = Counter()
    for D in range(max(1, T - 2), T + 3):
        for E in range(max(2, T - 1), T + 4):
            obs, pred = cv_only(L, *setup(L, D, E))
            res[(D >= T and E >= T + 1, obs, pred)] += 1
    print(f"L={L} grid D in {max(1, T - 2)}..{T + 2}, E in {max(2, T - 1)}..{T + 3}: "
          f"(D>=2^L and E>=2^L+1, observed, predicted) -> count {dict(res)}", flush=True)
    # 3. Random marker placements (goal and current anywhere, may coincide), every pair, every round.
    if L <= 4:
        bad2 = bad3 = 0; tot = 0; mism = 0; cens_rounds = 0; pred_same = 0
        for _ in range(40 if L <= 3 else 12):
            n = random.randint(6, 3 * T + 12); M = n - 1
            a1, g1, a2, g2 = (random.randrange(n) for _ in range(4))
            rounds, same, pred = full_check(L, M, a1, g1, a2, g2)
            ok2, ok3, _ = verdict(rounds); tot += 1; bad2 += not ok2; bad3 += not ok3
            cens_rounds += sum(cens for cens, _, _ in rounds[1:]); pred_same += pred == 'same'
            if same is not None and pred in ('same', 'diff') and (pred == 'same') != same:
                mism += 1
        print(f"L={L} random placements: {tot} pairs of graphs ({cens_rounds} of {tot * L} rounds l >= 1 with equal "
              f"censuses, {pred_same} predicted 'same' at (c,v)); soundness failures {bad2}, exactness failures {bad3}, "
              f"(c,v) prediction failures {mism}", flush=True)
    # 4. Path length. Sufficient condition of the proof: 2^(L+1) - 2 <= c and n - c >= 2^(L+1), so n >= 2^(L+2) - 2.
    for n in (4 * T - 3, 4 * T - 2, 4 * T + 11):
        same_at = []
        for a in range(T, n - T - 1):
            obs, pred = cv_only(L, n - 1, a, a + 1 + T, a - T)
            same_at.append(a) if obs == 'same' else None
        print(f"L={L} n={n}: positions c (goal 2^L beyond v / 2^L + 1 behind v) with equal (c,v) colour: {same_at}; "
              f"proof's range [{2 * T - 2}, {n - 2 * T}]", flush=True)
