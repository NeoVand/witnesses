"""Lay out the demo's 125-vertex questions (static/model/graphs_v125.json) as road maps with the figures' map engine
(map_engine.py, this folder), into static/maps125/<question>.json. Every guarantee the engine asserts at
49 vertices is asserted here too (map_engine.check on the full and the rounded layout). Paths relative to the repository root.

    nice -n 19 python -I maps_v125.py [indices, comma-separated]    (numpy, scipy, matplotlib 3.10.7, networkx)
"""
import json, pathlib, sys, time
from multiprocessing import Pool

TT = pathlib.Path(__file__).resolve().parents[2]      # repository root
ENGINE = TT / 'research' / 'search'
GRAPHS = TT / 'static' / 'model' / 'graphs_v125.json'
MAPS = TT / 'static' / 'maps125'
DEMO_PX = 900.0


def setup():
    sys.path.insert(0, str(ENGINE))
    import map_engine as me
    me.GRAPHS, me.MAPS = GRAPHS, MAPS
    # 125 towns on the engine's largest sheet (112 mm) leave too little room at 700 px; these maps are laid out (and
    # every guarantee checked) for the sheet drawn DEMO_PX px wide, the width the page gives them
    me.DEMO_PX = DEMO_PX
    me.place_labels.__defaults__ = (DEMO_PX,)
    return me


def one(i):
    me = setup()
    t0 = time.time()
    r = me.one(i)
    return r + (round(time.time() - t0, 1),) if r[1] == 'FAILED' else r


if __name__ == '__main__':
    MAPS.mkdir(parents=True, exist_ok=True)
    n = len(json.loads(GRAPHS.read_text())['questions'])
    idx = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(n))
    rows = []
    with Pool(6, initializer=setup) as pool:
        for r in pool.imap_unordered(one, idx):
            rows.append(r)
            print(*r, flush=True)
    failed = [r for r in rows if r[1] == 'FAILED']
    total = sum((MAPS / f'{i}.json').stat().st_size for i in idx if (MAPS / f'{i}.json').exists())
    print(f'{len(rows)} maps, {len(failed)} failed, {total / 1e6:.2f} MB in {MAPS}')
