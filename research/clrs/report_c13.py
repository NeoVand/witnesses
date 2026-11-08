"""The width-128 layer (c13) per task and seed on the official file and the fresh draw (seed 20,261,109), K = 8, against
FloydNet without hints and the width-32 triangle layer (c1/c3); 'matches' = mean at least FloydNet - 0.5, 'beats' =
above. python3 report_c13.py > results/c13_report.txt"""
import json
from statistics import mean
from pathlib import Path
H = Path(__file__).resolve().parent
fn = json.loads((H / 'published.json').read_text())['floydnet_no_hint']
off, new, tri = {}, {}, {}
for v in json.loads((H / 'scores_c13.json').read_text()).values():
    off.setdefault(v['task'], []).append((v['seed'], 100 * v['test'][7], v['seconds'], v['params']))
for v in json.loads((H / 'scores_fresh_20261109_c13.json').read_text()).values():
    new.setdefault(v['task'], []).append((v['seed'], 100 * v['fresh'][7]))
for f in ('scores_c1.json', 'scores_c3.json'):
    for v in json.loads((H / f).read_text()).values():
        tri.setdefault(v['task'], []).append(100 * v['test'][7])
cnt = {'official': [0, 0], 'new': [0, 0]}
for t in fn:
    if t.startswith('_') or t == 'cite':
        continue
    f = fn[t][0]
    o = sorted(off[t]); n = sorted(new[t])
    mo, mn = mean(x[1] for x in o), mean(x[1] for x in n)
    lab = lambda m: 'beats' if m > f else 'matches' if m >= f - .5 else 'below'
    for k, m in (('official', mo), ('new', mn)):
        cnt[k][0] += lab(m) != 'below'; cnt[k][1] += lab(m) == 'beats'
    print(f"{t:30s} FloydNet {f:6.2f} | official {mo:6.2f} {[round(x[1], 2) for x in o]} {lab(mo):7s} | new {mn:6.2f} {[round(x[1], 2) for x in n]} {lab(mn):7s} | "
          f"triangle {mean(tri[t]):6.2f} | s {[round(x[2]) for x in o]} | params {o[0][3]}")
print('matches or beats (beats):', {k: f'{v[0]} of 12 ({v[1]})' for k, v in cnt.items()})
