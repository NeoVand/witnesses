"""Do the residual errors on weighted tasks sit at small decision margins (the margin of the size law, Theorem 2)?
From saved predictions on the official 64-node test split (no model evaluation; CPU).
Margins, all relative to the weight (or distance) of the true choice:
- mst_prim (pointer pi[v], tree T): for v's tree edge e = (v, pi[v]), m(v) = min over non-tree edges f whose tree path
  passes through e of (w(f) - w(e)) / w(e), the smallest relative increase that lets another spanning tree replace e (and so
  change v's pointer); inf if no such f. For a wrong pointer u: whether (v, u) is an edge, and if so its cycle gap
  (w(v, u) - max tree weight on the path v..u) / w(v, u).
- bellman_ford, dijkstra (predecessor pointers from s): m(v) = min over in-neighbours u' != pi[v] of
  (d(u') + w(u', v) - d(v)) / d(v); for a wrong pointer u: (d(u) + w(u, v) - d(v)) / d(v).
- mst_kruskal (edge mask): tree edges get the replacement margin above, non-tree edges the cycle gap; errors are
  misclassified edges, counted once per unordered pair.
Reports, per task over seeds, the margins of all decisions against those of the wrong ones: medians and the share below
1% and 5%. Writes results/near_ties_<prefix>.json.
python3 near_ties.py <results dir> <data dir> <run prefix> [K, default 8]   (e.g. results data_local c13)"""
import heapq, json, sys
from pathlib import Path
import numpy as np


def tree_parent_paths(n, edges):
    """Adjacency of an undirected tree on n nodes; returns a function path(a, b) -> list of tree edges (min, max)."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    par, dep, order = [-1] * n, [0] * n, []
    seen = [False] * n
    for r in range(n):
        if seen[r]:
            continue
        seen[r] = True
        st = [r]
        while st:
            u = st.pop(); order.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True; par[w] = u; dep[w] = dep[u] + 1; st.append(w)

    def path(a, b):
        out = []
        while a != b:
            if dep[a] >= dep[b]:
                out.append((min(a, par[a]), max(a, par[a]))); a = par[a]
            else:
                out.append((min(b, par[b]), max(b, par[b]))); b = par[b]
            if a == -1 or b == -1:
                return None
        return out
    return path


def mst_margins(W, has, tree):
    """Replacement margin of each tree edge and cycle gap of each non-tree edge (relative)."""
    n = len(W)
    path = tree_parent_paths(n, tree)
    tset = set(tree)
    rep = {e: np.inf for e in tree}
    cyc = {}
    for a in range(n):
        for b in range(a + 1, n):
            if not has[a, b] or (a, b) in tset:
                continue
            p = path(a, b)
            if not p:
                continue
            w = W[a, b]
            cyc[(a, b)] = (w - max(W[x, y] for x, y in p)) / w
            for e in p:
                rep[e] = min(rep[e], (w - W[e]) / W[e])
    return rep, cyc


def dists(W, has, s):
    n = len(W)
    d = np.full(n, np.inf); d[s] = 0
    h = [(0., s)]
    while h:
        du, u = heapq.heappop(h)
        if du > d[u]:
            continue
        for v in np.nonzero(has[u])[0]:
            if du + W[u, v] < d[v]:
                d[v] = du + W[u, v]; heapq.heappush(h, (d[v], v))
    return d


def summary(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), median=round(float(np.median(x)), 4), below_1pct=round(float((x < .01).mean()), 3),
                below_5pct=round(float((x < .05).mean()), 3))


def main(results, data_dir, prefix, K=8):
    K = int(K)
    out = {}
    for task in ('mst_prim', 'bellman_ford', 'dijkstra', 'mst_kruskal'):
        d = {k: v for k, v in np.load(Path(data_dir) / task / 'test.npz').items()}
        A, nn = d['A'], d['n']
        name = 'in_mst' if task == 'mst_kruskal' else 'pi'
        allm, errm, errgap, nonedge, nerr, ndec = [], [], [], 0, 0, 0
        for sd in (0, 1, 2):
            pz = np.load(Path(results) / f'{prefix}-{task}-s{sd}' / f'preds_s{sd}.npz')
            pr = pz[f'test/{name}'][K - 1].astype(int)
            for g in range(len(nn)):
                n = int(nn[g]); W = A[g, :n, :n].astype(np.float64); has = W != 0
                if task == 'mst_kruskal':
                    t = d['in_mst'][g, :n, :n]
                    tree = [(a, b) for a in range(n) for b in range(a + 1, n) if t[a, b] == 1]
                    rep, cyc = mst_margins(W, has, tree)
                    for a in range(n):
                        for b in range(a + 1, n):
                            if not has[a, b]:
                                continue
                            m = rep.get((a, b), cyc.get((a, b), np.inf))
                            wrong = pr[g, a, b] != t[a, b]
                            allm.append(m); ndec += 1
                            if wrong:
                                errm.append(m); nerr += 1
                    continue
                t = d['pi'][g, :n]
                if task == 'mst_prim':
                    tree = [(min(v, int(t[v])), max(v, int(t[v]))) for v in range(n) if t[v] != v]
                    rep, _ = mst_margins(W, has, tree)
                    path = tree_parent_paths(n, tree)
                    for v in range(n):
                        if t[v] == v:
                            continue
                        m = rep[(min(v, int(t[v])), max(v, int(t[v])))]
                        allm.append(m); ndec += 1
                        u = int(pr[g, v])
                        if u != t[v]:
                            nerr += 1; errm.append(m)
                            if u < n and has[v, u]:
                                p = path(v, u)
                                errgap.append((W[v, u] - max(W[x, y] for x, y in p)) / W[v, u] if p else np.inf)
                            else:
                                nonedge += 1
                else:
                    s = int(d['s'][g])
                    dd = dists(W, has, s)
                    for v in range(n):
                        if v == s or not np.isfinite(dd[v]) or t[v] == v:
                            continue
                        alts = [(dd[u] + W[u, v] - dd[v]) / dd[v] for u in np.nonzero(has[:, v])[0] if u != t[v] and np.isfinite(dd[u])]
                        m = min(alts) if alts else np.inf
                        allm.append(m); ndec += 1
                        u = int(pr[g, v])
                        if u != t[v]:
                            nerr += 1; errm.append(m)
                            if u < n and has[u, v] and np.isfinite(dd[u]):
                                errgap.append((dd[u] + W[u, v] - dd[v]) / dd[v])
                            else:
                                nonedge += 1
        r = dict(decisions=ndec, errors=nerr, error_rate=round(nerr / ndec, 4), margin_all=summary(allm), margin_at_errors=summary(errm))
        if task != 'mst_kruskal':
            r |= dict(chosen_gap_at_errors=summary(errgap), errors_not_an_edge=nonedge)
        out[task] = r
        print(task, json.dumps(r), flush=True)
    (Path(__file__).resolve().parent / 'results' / f'near_ties_{prefix}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
