"""Folklore 2-WL (2-FWL) colour refinement on labelled directed paths, the object of Theorem 1(ii).

Two graphs are refined jointly with one shared colour table per round, so colours are comparable across them, and
each graph's multisets range over its own vertices only, as a pair network applied to each graph alone.

The atomic type of a pair (u, v) records equality, the edge in each direction, and the goal and current markers on
either end. `graph(M, a, goal)` is the directed path 0 -> 1 -> ... -> M (n = M + 1 vertices) with the current vertex
at a and the goal at `goal`.
"""
import numpy as np


def atomic(n, adj, mg, mc):
    I = np.eye(n, dtype=np.int64)
    return I + adj * 2 + adj.T * 4 + mg[:, None] * 8 + mg[None, :] * 16 + mc[:, None] * 32 + mc[None, :] * 64


def joint_refine(cols, rounds):
    """`rounds` rounds of 2-FWL on the colour matrices `cols`, with one colour table shared by all of them."""
    flat = np.concatenate([c.ravel() for c in cols]); _, lab = np.unique(flat, return_inverse=True)
    sizes = [c.size for c in cols]; cols = [x.reshape(c.shape) for x, c in zip(np.split(lab, np.cumsum(sizes)[:-1]), cols)]
    for _ in range(rounds):
        C = int(max(c.max() for c in cols)) + 1; sigs = []
        for col in cols:
            n = col.shape[0]
            pairs = np.sort(col[:, None, :] * C + col.T[None, :, :], axis=2)   # [u, v, k] = (col[u,k], col[k,v])
            sigs.append(np.concatenate([col[:, :, None], pairs], axis=2).reshape(n * n, -1))
        _, lab = np.unique(np.concatenate(sigs), axis=0, return_inverse=True)
        cols = [x.reshape(c.shape) for x, c in zip(np.split(lab.ravel(), np.cumsum([s.shape[0] for s in sigs])[:-1]), cols)]
    return cols


def graph(M, a, goal):
    n = M + 1; adj = np.zeros((n, n), np.int64); mg = np.zeros(n, np.int64); mc = np.zeros(n, np.int64)
    for i in range(M): adj[i, i + 1] = 1
    mc[a] = 1; mg[goal] = 1
    return atomic(n, adj, mg, mc)
