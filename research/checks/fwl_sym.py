"""Theorem 1(ii), the length condition: 2-FWL colour of (c, v) after L rounds on two paths of the same length.

G1 has the goal D steps beyond the child v = c + 1 (reachable); G2 has it E steps behind v (unreachable). For
D = 2^L - 1, 2^L, 2^L + 1 and E = D, D + 1, print whether (c, v) has the same colour in both. The path has
M = 2B + 2^(L+1) + 6 edges with B = max(D, E) + 1 and c at vertex B + 2^L + 2; at (D, E) = (2^L, 2^L + 1) that is
2^(L+2) + 11 vertices with c at vertex 2^(L+1) + 4.

python -I fwl_sym.py L_min L_max
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fwl import graph, joint_refine

for L in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
    out = []
    for D in (2**L - 1, 2**L, 2**L + 1):
        for E in (D, D + 1):
            B = max(D, E) + 1
            M = 2 * B + 2 ** (L + 1) + 6; a = B + 2 ** L + 2
            c1, c2 = joint_refine([graph(M, a, a + 1 + D), graph(M, a, a + 1 - E)], L)
            out.append(f"D={D},E={E}:{'same' if c1[a, a+1] == c2[a, a+1] else 'diff'}")
    print(f"L={L}: " + "  ".join(out), flush=True)
