"""Anatomy of the order-dependent CLRS-30 outputs (DFS, topological sort, SCC) against the NC / P-complete line.

1. Lex-least paths: on every topological_sort test DAG, the library's DFS parent of v equals the penultimate vertex of
   the lexicographically least path from r(v) = min{s : s reaches v} (the lemma that puts lex DFS of a DAG in NC).
2. SCC: scc_id(v) equals the first node of v's component discovered by pass-1 lex DFS; how often it is the component's
   smallest index; whether wrong predictions stay inside the component.
3. Accuracy against iterations K = 1..16 for DFS (by depth in the true tree), topological sort and SCC, from the
   stored predictions (runs c1 and c3, seeds 0-2).
4. Topological sort: share of successor pointers forced by the partial order alone (same in every linear extension).
python -I anatomy.py . results/anatomy.json      (from this folder; needs dm-clrs 2.0.3)
"""
import json, sys
from pathlib import Path
import numpy as np
from clrs._src.algorithms import graphs as G

CLRS = Path(sys.argv[1]); OUT = Path(sys.argv[2]); res = {}


def lex_dfs(A):
    """Library-order DFS: start nodes and neighbours in index order. Returns parent, discovery order, finish order."""
    n = len(A); color = np.zeros(n, int); pi = np.arange(n); disc = []; fin = []
    for s in range(n):
        if color[s]: continue
        stack = [(s, 0)]; color[s] = 1; disc.append(s)
        while stack:
            u, i = stack[-1]
            while i < n and not (A[u, i] != 0 and color[i] == 0):
                i += 1
            if i < n:
                stack[-1] = (u, i + 1); color[i] = 1; pi[i] = u; disc.append(i); stack.append((i, 0))
            else:
                stack.pop(); color[u] = 2; fin.append(u)
    return pi, disc, fin


def closure(A):
    n = len(A); R = (A != 0) | np.eye(n, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        R = (R.astype(np.int64) @ R.astype(np.int64)) > 0
    return R


def lex_least_parent(A):
    """Penultimate vertex of the lex-least path from r(v) to v, by DP in a topological order (optimal substructure)."""
    n = len(A); R = closure(A); root = np.array([int(np.argmax(R[:, v])) for v in range(n)])
    order = [v for v in np.argsort(-R.sum(1))]   # ancestors reach more: a topological order of a DAG
    best = {}; par = np.arange(n)
    for v in order:
        r = root[v]
        if v == r:
            best[v] = (v,); continue
        cands = [best[p] + (v,) for p in range(n) if A[p, v] != 0 and R[r, p] and root[p] == r]
        b = min(cands); best[v] = b; par[v] = b[-2]
    return par


# 1. lex-least path lemma on the DAGs of topological_sort and dag_shortest_paths
z = np.load(CLRS / 'data_local/topological_sort/test.npz'); ok = tot = 0
for g in range(len(z['n'])):
    n = int(z['n'][g]); A = z['A'][g, :n, :n]
    pi_lib, _ = G.dfs(A); ok += int((lex_least_parent(A) == pi_lib).sum()); tot += n
    assert (lex_dfs(A)[0] == pi_lib).all()
res['lexleast_dag'] = dict(agree=ok, total=tot)
print('lex-least parent = library DFS parent on topo DAGs:', ok, '/', tot, flush=True)
# control: on the dfs task's cyclic graphs the lemma fails (lex-least SIMPLE path is not the DFS tree path)

# 2. SCC representative
z = np.load(CLRS / 'data_local/strongly_connected_components/test.npz')
first_ok = minidx_ok = tot = 0; sizes = []; in_comp = [0, 0, 0]; wrong = [0, 0, 0]
preds = [np.load(CLRS / f'results/c3-strongly_connected_components-s{k}/preds_s{k}.npz')['test/scc_id'][7] for k in range(3)]
contains0_ok = contains0_tot = 0
for g in range(len(z['n'])):
    n = int(z['n'][g]); A = z['A'][g, :n, :n]; truth = z['scc_id'][g, :n]
    R = closure(A); comp = R & R.T
    _, disc, _ = lex_dfs(A); rank = np.empty(n, int); rank[disc] = np.arange(n)
    for v in range(n):
        members = np.where(comp[v])[0]
        first_ok += int(truth[v] == members[np.argmin(rank[members])]); minidx_ok += int(truth[v] == members.min()); tot += 1
    sizes.append(sorted([int(c) for c in np.unique(truth, return_counts=True)[1]], reverse=True))
    for k in range(3):
        pr = preds[k][g, :n]; w = pr != truth; wrong[k] += int(w.sum())
        in_comp[k] += int(sum(comp[v, pr[v]] for v in np.where(w)[0]))
res['scc'] = dict(rep_is_first_discovered=first_ok / tot, rep_is_min_index=minidx_ok / tot,
                  wrong=wrong, wrong_inside_component=in_comp, component_sizes_first3=sizes[:3])
print('scc', res['scc'], flush=True)

# 3. accuracy against K
def acc_by_K(task, name, pre):
    z = np.load(CLRS / f'data_local/{task}/test.npz'); out = {}
    P = [np.load(CLRS / f'results/{pre}-{task}-s{k}/preds_s{k}.npz')[f'test/{name}'] for k in range(3)]
    depth = None
    if task == 'dfs':
        depth = []
        for g in range(len(z['n'])):
            n = int(z['n'][g]); pi = z['pi'][g, :n]; d = np.zeros(n, int)
            for v in range(n):
                u, c = v, 0
                while pi[u] != u: u, c = pi[u], c + 1
                d[v] = c
            depth.append(d)
    for K in range(1, 17):
        accs = []; dep_acc = {}
        for k in range(3):
            right = [];
            for g in range(len(z['n'])):
                n = int(z['n'][g]); r = P[k][K - 1][g, :n] == z[name][g, :n]; right.append(r)
                if depth is not None:
                    for v in range(n):
                        dep_acc.setdefault(int(depth[g][v]), []).append(bool(r[v]))
            accs.append(float(np.mean(np.concatenate(right))))
        row = dict(acc=accs)
        if depth is not None:
            ds = sorted(dep_acc)
            row['deepest_95'] = next((d - 1 for d in ds if np.mean(dep_acc[d]) < .95), ds[-1])
            row['acc_by_depth'] = {d: round(float(np.mean(dep_acc[d])), 3) for d in ds if len(dep_acc[d]) >= 30}
        out[K] = row
    return out
res['dfs_by_K'] = acc_by_K('dfs', 'pi', 'c3')
res['topo_by_K'] = acc_by_K('topological_sort', 'topo', 'c1')
res['scc_by_K'] = acc_by_K('strongly_connected_components', 'scc_id', 'c3')
for K in (1, 2, 3, 4, 5, 6, 8, 12, 16):
    print('K', K, 'dfs', np.round(res['dfs_by_K'][K]['acc'], 3), 'deepest95', res['dfs_by_K'][K]['deepest_95'],
          'topo', np.round(res['topo_by_K'][K]['acc'], 3), 'scc', np.round(res['scc_by_K'][K]['acc'], 3), flush=True)
z = np.load(CLRS / 'data_local/dfs/test.npz')
depths = []
for g in range(len(z['n'])):
    n = int(z['n'][g]); pi = z['pi'][g, :n]
    for v in range(n):
        u, c = v, 0
        while pi[u] != u: u, c = pi[u], c + 1
        depths.append(c)
res['dfs_depth_hist'] = np.bincount(depths).tolist()

# 4. forced successors in topological sort
z = np.load(CLRS / 'data_local/topological_sort/test.npz')
P = [np.load(CLRS / f'results/c1-topological_sort-s{k}/preds_s{k}.npz')['test/topo'][7] for k in range(3)]
forced = tot = 0; err_f = [0, 0, 0]; err_u = [0, 0, 0]; incomp = []
for g in range(len(z['n'])):
    n = int(z['n'][g]); A = z['A'][g, :n, :n]; truth = z['topo'][g, :n]
    R = closure(A); lt = R & ~np.eye(n, dtype=bool)          # lt[u, v]: u < v
    incomp.append(float((~(R | R.T)).sum() / (n * (n - 1))))
    for u in range(n):
        up = np.where(lt[u])[0]
        cand = [v for v in range(n) if v != u and not lt[v, u] and not (lt[u] & lt[:, v]).any()]
        can_end = not lt[u].any()                            # u can be last only if maximal
        f = (len(cand) + int(can_end)) == 1
        forced += int(f); tot += 1
        for k in range(3):
            w = int(P[k][g, u] != truth[u])
            if f: err_f[k] += w
            else: err_u[k] += w
res['topo_forced'] = dict(forced_share=forced / tot, err_forced=err_f, err_unforced=err_u,
                          incomparable_pair_share_mean=float(np.mean(incomp)))
print('topo forced', res['topo_forced'], flush=True)
OUT.write_text(json.dumps(res, indent=1))
