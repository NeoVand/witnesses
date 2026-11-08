"""Where the models err on the 64-node test split (CPU, eval only, from the saved predictions).
dfs: agreement with the fixed rule "largest-index in-neighbour below v", and accuracy by depth in the true DFS tree.
scc: of the wrong pointers, the share that still point inside the node's true SCC; agreement with "smallest index in
the node's SCC". articulation_points, bridges, mst_kruskal: precision and recall; bridges: error rate on non-bridge
edges by the length of the shortest cycle through the edge (16 nodes is the largest training graph); mst_kruskal: for
each wrong edge, the relative gap |w(u,v) - B(u,v)| / w(u,v), B the min-max path weight between u and v without the edge.
Prints one line per seed. results/c13_scc_dfs_anatomy.txt is this script on c13 (scc, then dfs; seeds 0,1,2, K = 8);
results/c17_scc_anatomy.txt on c17 at K = 8 and K = 34.
python diagnose_c3.py <results dir> <data dir> <task> [seeds] [K, default 8] [run prefix, default c3]"""
import json, sys
from collections import deque
from pathlib import Path
import numpy as np


def bfs_dist(adj, s, skip):
    n = len(adj)
    d = np.full(n, np.inf)
    d[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in np.nonzero(adj[u])[0]:
            if (u, v) in skip or v == u:
                continue
            if d[v] == np.inf:
                d[v] = d[u] + 1
                q.append(v)
    return d


def minimax(W, has, u, v):
    """Min over u-v paths avoiding edge (u, v) of the largest edge weight (Dijkstra on the bottleneck)."""
    n = len(W)
    best = np.full(n, np.inf)
    best[u] = 0.
    done = np.zeros(n, bool)
    for _ in range(n):
        x = np.where(done, np.inf, best).argmin()
        if best[x] == np.inf:
            break
        done[x] = True
        for y in np.nonzero(has[x])[0]:
            if {x, y} == {u, v} or y == x:
                continue
            best[y] = min(best[y], max(best[x], W[x, y]))
    return best[v]


def prf(pred, t):
    ok = t != -1
    tp, fp, fn = ((pred > .5) & (t > .5) & ok).sum(), ((pred > .5) & (t < .5) & ok).sum(), ((pred < .5) & (t > .5) & ok).sum()
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), int(fp), int(fn)


def main(results, data_dir, task, seeds='0,1,2', K=8, prefix='c3'):
    d = {k: v for k, v in np.load(Path(data_dir) / task / 'test.npz').items()}
    name = {'dfs': 'pi', 'strongly_connected_components': 'scc_id', 'articulation_points': 'is_cut', 'bridges': 'is_bridge', 'mst_kruskal': 'in_mst'}[task]
    t, A = d[name], d['A']
    G, n = A.shape[:2]
    out = {}
    for sd in map(int, seeds.split(',')):
        pz = np.load(Path(results) / f'{prefix}-{task}-s{sd}' / f'preds_s{sd}.npz')
        pr = pz[f'test/{name}'][int(K) - 1].astype(int)
        r = {}
        if task == 'dfs':
            has = A != 0
            rule = np.array([[max([u for u in range(v) if has[g, u, v]], default=v) for v in range(n)] for g in range(G)])
            depth = np.zeros_like(t)
            for g in range(G):
                for v in range(n):
                    u, k = v, 0
                    while t[g, u] != u:
                        u, k = t[g, u], k + 1
                    depth[g, v] = k
            r['acc'] = float((pr == t).mean())
            r['agree_with_rule'] = float((pr == rule).mean())
            r['rule_acc'] = float((rule == t).mean())
            r['acc_by_depth'] = {f'{a}-{b}': round(float((pr == t)[(depth >= a) & (depth <= b)].mean()), 3) for a, b in ((0, 7), (8, 15), (16, 31), (32, 63))}
        elif task == 'strongly_connected_components':
            wrong = pr != t
            same = np.array([[t[g, pr[g, v]] == t[g, v] for v in range(n)] for g in range(G)])
            mins = np.array([[np.nonzero(t[g] == t[g, v])[0].min() for v in range(n)] for g in range(G)])
            r['acc'] = float((pr == t).mean())
            r['wrong_but_inside_true_scc'] = float(same[wrong].mean()) if wrong.any() else None
            r['agree_with_min_index_rule'] = float((pr == mins).mean())
            r['acc_where_rep_is_min_index'] = float((pr == t)[t == mins].mean())
            r['acc_where_rep_is_not_min_index'] = float((pr == t)[t != mins].mean())
        else:
            p, rc, fp, fn = prf(pr, t)
            r |= dict(precision=round(float(p), 4), recall=round(float(rc), 4), false_pos=fp, false_neg=fn, positives=int((t == 1).sum()))
            if task == 'bridges':
                rows = []
                for g in range(G):
                    adj = (A[g] != 0)
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u, v] and t[g, u, v] == 0:
                                L = bfs_dist(adj, u, {(u, v), (v, u)})[v] + 1
                                rows.append((L, pr[g, u, v] != 0))
                rows = np.array(rows)
                r['nonbridge_error_by_cycle_len'] = {f'{a}-{b}': [round(float(rows[(rows[:, 0] >= a) & (rows[:, 0] <= b), 1].mean()), 3), int(((rows[:, 0] >= a) & (rows[:, 0] <= b)).sum())]
                                                     for a, b in ((3, 8), (9, 16), (17, 64)) if ((rows[:, 0] >= a) & (rows[:, 0] <= b)).any()}
            if task == 'mst_kruskal':
                gaps = []
                for g in range(G):
                    has = A[g] != 0
                    for u in range(n):
                        for v in range(u + 1, n):
                            if has[u, v] and (pr[g, u, v] != t[g, u, v] or pr[g, v, u] != t[g, v, u]):
                                B = minimax(A[g], has, u, v)
                                gaps.append(abs(A[g, u, v] - B) / A[g, u, v] if np.isfinite(B) else np.inf)
                g_ = np.array(gaps)
                r['wrong_edges'] = len(g_)
                r['wrong_on_non_edges'] = int(((pr != t) & (A == 0)).sum())
                r['gap_median'] = float(np.median(g_[np.isfinite(g_)])) if np.isfinite(g_).any() else None
                r['gap_under_5pct'] = float((g_ < .05).mean()) if len(g_) else None
                r['wrong_edge_with_no_other_path'] = int((~np.isfinite(g_)).sum())
        out[sd] = r
        print(task, 'seed', sd, json.dumps(r))
    return out


if __name__ == '__main__':
    main(*sys.argv[1:])
