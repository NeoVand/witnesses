"""Every CLRS-30 number of the paper, recomputed from the result files in this folder (no model, no library; CPU).
Accuracies are clrs.evaluate scores (score.py, score_fresh.py) in %, mean over seeds (min to max), at the paper's
loop counts: K = 8 (index 7), K = 34 (index 33) for the runs trained with 4 to 24 loops (c14, c17).
'matches' = mean at least the published score minus 0.5 points; 'within a point' = mean at least the reference's minus 1.
python3 -I report_paper.py"""
import json, re
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
TASKS = ['bfs', 'dfs', 'topological_sort', 'articulation_points', 'bridges', 'strongly_connected_components', 'mst_kruskal',
         'mst_prim', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall']  # Supplementary Table order
PUB = ['floydnet_no_hint', 'bevilacqua2023_no_hints', 'rodionov2023_no_hint_ours', 'ibarz2022_triplet_gmpnn_hints',
       'muller2024_edge_transformer', 'floydnet_hint']
J = lambda f: json.loads((H / f).read_text())
pub = J('published.json')


def scores(f, key='test', K=8, prefix=None):
    """{task: array over seeds (sorted) of the score in % after K runs}."""
    out = {}
    for v in J(f).values():
        if prefix and not v['run'].startswith(prefix):
            continue
        out.setdefault(v['task'], []).append((v['seed'], 100 * v[key][K - 1]))
    return {t: np.array([x for _, x in sorted(r)]) for t, r in out.items()}


def curve(f, key='test'):
    """array (seeds, K) of scores in %."""
    return 100 * np.array([v[key] for v in sorted(J(f).values(), key=lambda v: v['seed'])])


fmt = lambda a: f'{a.mean():.2f} ({a.min():.2f}-{a.max():.2f})'
c13, c13f = scores('scores_c13.json'), scores('scores_fresh_20261109_c13.json', 'fresh')
rows = {t: (c13[t], c13f[t]) for t in TASKS}
extra = {'dfs, 34 loops (c14)': (scores('scores_c14.json', K=34)['dfs'], None),
         'scc, 34 loops (c17)': (scores('scores_c17.json', K=34)['strongly_connected_components'],
                                 scores('scores_fresh_20261109_c17.json', 'fresh', 34)['strongly_connected_components']),
         'prim, Kirchhoff (c15)': (scores('scores_c15.json')['mst_prim'], scores('scores_fresh_20261109_c15.json', 'fresh')['mst_prim']),
         'prim, Kirchhoff + log n (c16)': (scores('scores_c16.json')['mst_prim'], scores('scores_fresh_20261109_c16.json', 'fresh')['mst_prim'])}
print('== Supplementary Table CLRS-30 (official file | fresh draw 20,261,109 | published: ' + ', '.join(PUB) + ')')
for t, (o, f) in list(rows.items()) + list(extra.items()):
    p = ' '.join('-' if t not in pub[k] or pub[k][t][0] is None else f'{pub[k][t][0]:.2f}' for k in PUB) if t in TASKS else ''
    print(f'{t:32s} {fmt(o):24s} | {fmt(f) if f is not None else "-":24s} | {p}')

print('\n== Against FloydNet without hints (c13, K = 8)')
fl = {t: pub['floydnet_no_hint'][t][0] for t in TASKS}
for name, s in (('official', c13), ('fresh', c13f)):
    m = {t: s[t].mean() for t in TASKS}
    ok = [t for t in TASKS if m[t] >= fl[t] - .5]
    print(f'{name}: matches or beats on {len(ok)} of 12; beats on {sum(m[t] > fl[t] for t in TASKS)}; below:',
          ', '.join(f'{t} {m[t] - fl[t]:+.2f}' for t in TASKS if t not in ok), f'| dijkstra {m["dijkstra"]:.3f} against {fl["dijkstra"] - .5:.2f}')
print('parameters:', sorted({v['params'] for v in J('scores_c13.json').values()}),
      '| seconds per run (L4): %.0f to %.0f' % (min(v['seconds'] for v in J('scores_c13.json').values()), max(v['seconds'] for v in J('scores_c13.json').values())))
for t in ('floyd_warshall', 'topological_sort', 'dfs', 'mst_prim', 'strongly_connected_components'):
    print(f'  {t} {c13[t].mean():.2f}', end='')
print(f"\n  FloydNet floyd_warshall {fl['floyd_warshall']}, dfs {fl['dfs']}, scc {fl['strongly_connected_components']}, prim {fl['mst_prim']};"
      f" no-hint floyd_warshall: Bevilacqua {pub['bevilacqua2023_no_hints']['floyd_warshall'][0]}, Rodionov {pub['rodionov2023_no_hint_ours']['floyd_warshall'][0]}")

print('\n== Width-32 designs against the looped triangle layer (c1/c3, seeds 0-2), within a point at K = 8')
tri = scores('scores_c1.json') | scores('scores_c3.json')
for run, name in (('c7', 'design 1, OR join'), ('c8', 'design 2, OR join + power mean'), ('c9', 'design 3, + log n shift')):
    s = scores(f'scores_{run}.json')
    print(f'{run} {name}: {sum(s[t].mean() >= tri[t].mean() - 1 for t in tri)} of 12; parameters',
          sorted({v['params'] for v in J(f'scores_{run}.json').values()}))
f10, ftri = scores('scores_fresh_c10.json', 'fresh'), scores('scores_fresh_c10tri_c10c9.json', 'fresh', prefix='c10tri-')
print(f'c10 (c9 on seeds 3-5) against c10tri on the fresh draw 20,261,107: {sum(f10[t].mean() >= ftri[t].mean() - 1 for t in ftri)} of 12;',
      'below by more:', ', '.join(f'{t} {f10[t].mean() - ftri[t].mean():+.2f}' for t in ftri if f10[t].mean() < ftri[t].mean() - 1))
print('triangle layer parameters', sorted({v['params'] for f in ('scores_c1.json', 'scores_c3.json') for v in J(f).values()}))

print('\n== DFS with 4 to 24 loops (c14)')
c14 = curve('scores_c14.json')
print('K = 34 per seed', np.round(c14[:, 33], 2), f'mean {c14[:, 33].mean():.2f}; at least 90%: {(c14[:, 33] >= 90).sum()} of 3;',
      'mean by K:', {k: round(float(c14[:, k - 1].mean()), 1) for k in (2, 4, 8, 14, 20, 24, 32, 34)})
print('c13 dfs at K = 8: %.2f; seconds per run (L4): %.0f to %.0f' % (c13['dfs'].mean(), *(f(v['seconds'] for v in J('scores_c14.json').values()) for f in (min, max))))

print('\n== SCC with 4 to 24 loops (c17) and the split by representative')
c17 = curve('scores_c17.json')
print(f'K = 34 official {np.round(c17[:, 33], 2)} mean {c17[:, 33].mean():.2f}; fresh {fmt(extra["scc, 34 loops (c17)"][1])};',
      'mean by K:', {k: round(float(c17[:, k - 1].mean()), 1) for k in (2, 4, 8, 14, 19, 25, 34)})
for f, K in (('results/c17_split_c13.json', 8), ('results/c17_split_c17.json', 34)):
    s = J(f)
    r = [s[f'test/s{sd}/K{K}'] for sd in (0, 1, 2)]
    print(f'{f} K = {K}: DFS-fixed {np.mean([x["dfs_fixed"] for x in r]):.2f}, smallest-index {np.mean([x["min_index"] for x in r]):.2f}, '
          f'wrong inside {sum(x["wrong_inside"] for x in r)} of {sum(x["wrong"] for x in r)}')
an = {}
for ln in (H / 'results' / 'c13_scc_dfs_anatomy.txt').read_text().splitlines():
    m = re.match(r'(\w+) seed (\d) (\{.*\})', ln)
    an[m.group(1), int(m.group(2))] = json.loads(m.group(3))
scc = [an['strongly_connected_components', sd] for sd in (0, 1, 2)]
dfs = [an['dfs', sd] for sd in (0, 1, 2)]
print('c13 anatomy: smallest index', [round(100 * r['acc_where_rep_is_min_index'], 2) for r in scc], 'otherwise',
      [round(100 * r['acc_where_rep_is_not_min_index'], 2) for r in scc], 'wrong inside', [r['wrong_but_inside_true_scc'] for r in scc],
      '| dfs depth 0-7', [r['acc_by_depth']['0-7'] for r in dfs], 'depth 32-63', [r['acc_by_depth']['32-63'] for r in dfs])

print('\n== Near ties (c13, K = 8): median relative margin at errors | over all decisions')
for t, r in J('results/near_ties_c13.json').items():
    print(f"{t:14s} {r['margin_at_errors']['median']:.4f} | {r['margin_all']['median']:.4f}  ({r['errors']} errors in {r['decisions']})")

print('\n== Prim: errors to the own child, error rate by depth (c15_anatomy.py) and the Kirchhoff read-outs')
for run in ('c13', 'c15', 'c16'):
    a = J(f'results/c15_anatomy_{run}.json')['pooled']
    print(f"{run}: {a['orient']} of {a['err']} errors to the own child; depth 0-4 {100 * a['rate_le4']:.2f}%, 8+ {100 * a['rate_ge8']:.2f}%;",
          'bins', {k: (round(100 * v, 2) if v is not None else None) for k, v in a['c14_rates'].items()})
hand = J('results/c15_hand_eps0.0001.json')
k0 = [k for k in hand if 'off=0.0' in k][0]
print('hand-built read-out (conductance 1 on the tree):', hand[k0]['exact_tree_rule'], 'and', hand[k0]['exact_blind_rule'], 'of', hand[k0]['nodes'])
for f in ('results/c15_cond_rules.json', 'results/c16_cond_rules.json'):
    print(f, 'best head orients', [max(h['parent_lower'] for h in r['heads'].values()) for r in J(f).values()], 'of 2048')
for run in ('c15', 'c16'):
    print(run, 'parameters', sorted({v['params'] for v in J(f'scores_{run}.json').values()}), 'official', fmt(extra[[k for k in extra if run in k][0]][0]),
          'fresh', fmt(extra[[k for k in extra if run in k][0]][1]))
kap = J('results/c16_kappa.json')
print(f"c16 learned kappa: {kap['min']} to {kap['max']}, mean {kap['mean']} (required {kap['required_mean']})")
