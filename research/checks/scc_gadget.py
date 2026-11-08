"""Proposition 4, hardness: the reduction from lexicographic DFS order to the library's SCC representative.

Given a digraph G in which every vertex is reachable from 0 and two vertices u, v != 0, build G': each original
vertex x >= 1 becomes x + 2 (vertex 0 stays 0), with new vertices u' = 1 and v' = 2 and edges u -> u', v -> v',
u' -> v', v' -> u'. Claim: the CLRS-30 library's Kosaraju representative of u' is u' exactly when u is discovered
before v in the library-order (lexicographic) DFS of G. Checked against the library's own
`strongly_connected_components` on 400 random digraphs of 5 to 29 vertices.

Needs the CLRS-30 library (dm-clrs). python -I scc_gadget.py
"""
import numpy as np
from clrs._src.algorithms import graphs as G
rng = np.random.default_rng(0); ok = tot = 0
def disc_order(A):
    n = len(A); color = np.zeros(n, int); order = []
    for s in range(n):
        if color[s]: continue
        st = [(s, 0)]; color[s] = 1; order.append(s)
        while st:
            u, i = st[-1]
            while i < n and not (A[u, i] and not color[i]): i += 1
            if i < n: st[-1] = (u, i + 1); color[i] = 1; order.append(i); st.append((i, 0))
            else: st.pop()
    r = np.empty(n, int); r[order] = np.arange(n); return r
for t in range(400):
    n = int(rng.integers(5, 30)); p = rng.uniform(.08, .5)
    A = (rng.random((n, n)) < p).astype(float); np.fill_diagonal(A, 0)
    for i in range(1, n): A[rng.integers(0, i), i] = 1          # every vertex reachable from 0
    u, v = rng.choice(np.arange(1, n), 2, replace=False)
    rank = disc_order(A)
    m = n + 2; B = np.zeros((m, m)); old = lambda x: 0 if x == 0 else x + 2
    for a in range(n):
        for b in range(n):
            if A[a, b]: B[old(a), old(b)] = 1
    B[old(u), 1] = 1; B[old(v), 2] = 1; B[1, 2] = 1; B[2, 1] = 1
    scc, _ = G.strongly_connected_components(B)
    ok += int((scc[1] == 1) == (rank[u] < rank[v])); tot += 1
print('gadget agrees on', ok, 'of', tot)
