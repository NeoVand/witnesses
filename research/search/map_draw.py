"""A benchmark question drawn as a road map, once, for every figure, the film and the demo.

The question is test question 64 of the demo's set (static/model/graphs_v49.json): 49 towns and 48
one-way roads, a start with eight branches of six roads, one of which reaches the goal. data/map_question.json holds the demo
model's scores on it; the strongest wrong turn is the start's child the model scores highest
after the last layer, apart from the right one. Every road drawn is an edge of the question; nothing else is a road.

Geography (seeded, millimetres, y down, on an MW x MH sheet): a long lake, a smaller lake and a pond; a river
winding from the top edge to a bay, with a bridge on each of the three branches that cross it; a coast with the bay;
two woods and a pale ridge. Each branch is routed, not drawn: a shortest path from the start through a few waypoints
to its dead end, on a 0.25 mm cost grid where water is impassable (with a margin), river cells open only in a narrow
corridor square across each bridge, roads already laid are blocked, the ridge and woods cost a little more, the
ground just off a shore a little less (valley roads), and a smooth noise field gives the gentle wander of country
roads. The path is simplified and splined, and its six towns are spaced along it, one on the far side of every
bridge. The checks are exact: no road touches a lake or the sea, roads cross the river only at bridges, and no two
roads cross.

    build:  uv run ... python -I map_draw.py            (writes data/map_layout.json and out/map_preview.png)
    use:    import map_draw as md; L = md.load(); ax = md.sheet(fig, x, y, w)
            md.base(ax, L); md.names(ax, L); md.roads(ax, L, md.standard_highlight(L)); md.arc(ax, L)
            md.towns(ax, L); md.route_names(ax, L)

data/map_layout.json holds everything the drawing needs (towns, road polylines with their rank along the branch,
bridges, water, woods, ridge, names, label anchors for the full-size map), so code in any language can redraw it.
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parents[1] / "paper" / "figures" / "data"   # the figures' data
LAYOUT = DATA / "map_layout.json"
QUESTION = DATA / "map_question.json"
INDEX = 64                     # the test question (lookahead 6, 49 towns); 61 has the same shape

MW, MH = 84.0, 76.0            # the sheet, mm; drawn at full size (s = 1) the map is MW mm wide
RES = 0.25                     # routing grid, mm
SEED = 20261107
MEANDER = 1.5                  # mm, the amplitude of the roads' wander across their line

# ---------------------------------------------------------------------------------------------------- the land
HUB = (37.0, 29.0)             # the start: the hub town, where eight valley roads meet
# Each branch: waypoints after the hub, the last its dead end. "right" reaches the goal, "wrong" is the model's
# strongest wrong turn; the other six take the remaining directions in the order of their first town's name.
PLAN = {"right": [(43.5, 32.8), (48.5, 37.8), (54.0, 38.0), (58.8, 41.5), (59.3, 47.5), (61.0, 53.0), (64.5, 56.5), (66.5, 57.5)],
        "wrong": [(39.0, 37.0), (43.5, 44.0), (46.0, 52.0), (49.5, 58.0), (50.0, 65.0)],
        "E": [(42.0, 23.5), (48.0, 18.0), (52.5, 16.5), (61.0, 14.5), (68.0, 18.5), (72.5, 25.5)],
        "NE": [(39.5, 21.0), (44.0, 12.0), (45.5, 3.0)],
        "N": [(34.5, 22.0), (33.5, 12.0), (29.0, 3.5)],
        "NW": [(31.0, 26.0), (21.5, 23.0), (16.0, 14.0), (10.0, 6.5)],
        "W": [(30.0, 32.0), (19.5, 34.0), (9.0, 35.0), (3.5, 41.0)],
        "SW": [(35.0, 36.5), (36.5, 45.0), (37.5, 56.5), (29.0, 60.0), (19.0, 62.5), (10.0, 65.5)]}
SIDES = ["E", "NE", "N", "NW", "W", "SW"]
LAKES = [dict(c=(22.0, 45.5), r=(13.5, 4.8), rot=32, seed=4, amp=0.26, name="Long Lake"),
         dict(c=(25.5, 14.5), r=(5.8, 4.0), rot=-25, seed=11, amp=0.22),
         dict(c=(41.5, 49.0), r=(2.4, 1.8), rot=40, seed=17, amp=0.12)]
RIVER = dict(ctrl=[(50.0, -4.0), (49.0, 5.0), (53.0, 12.0), (50.5, 20.0), (55.0, 27.0), (52.0, 35.0),
                   (56.5, 42.0), (53.5, 49.0), (57.5, 54.0), (59.5, 60.0)],
             w=(0.8, 1.4), wig=0.8, seed=5)
BRIDGES = [(51.5, 16.5), (54.0, 38.0)]                      # where roads may cross the river
COAST = dict(ctrl=[(45.0, 82.0), (47.5, 74.0), (51.5, 67.0), (55.5, 61.5), (59.5, 58.5), (63.5, 59.0),
                   (67.5, 61.5), (70.0, 66.0), (73.0, 68.5), (76.5, 66.0), (78.0, 59.0), (76.5, 50.0),
                   (78.5, 41.0), (76.0, 31.0), (78.0, 20.0), (75.5, 9.0), (77.0, -6.0)],
             close=[(MW + 12, -12.0), (MW + 12, MH + 12), (45.0, MH + 12)], wig=0.7, seed=9, name="Mill Bay",
             name_at=(59.0, 68.0))
ISLANDS = [dict(c=(28.5, 50.0), r=(1.5, 0.85), rot=30, seed=41, amp=0.15),        # in Long Lake
           dict(c=(64.5, 71.5), r=(1.3, 0.8), rot=-20, seed=43, amp=0.15)]        # in the bay
WOODS = [dict(c=(7.0, 9.0), r=(8.5, 6.0), rot=-10, seed=21, amp=0.22),
         dict(c=(10.0, 69.0), r=(11.0, 6.0), rot=-12, seed=23, amp=0.22)]
RIDGE = dict(c=(69.0, 40.0), r=(10.0, 2.4), rot=-72, seed=31, amp=0.16)


# ---------------------------------------------------------------------------------------------------- geometry
def _spline(ctrl, n, closed=False):
    from scipy.interpolate import splprep, splev
    c = np.asarray(ctrl, float)
    tck, _ = splprep([c[:, 0], c[:, 1]], s=0, per=closed, k=min(3, len(c) - 1))
    u = np.linspace(0, 1, n, endpoint=not closed)
    x, y = splev(u, tck)
    return np.c_[x, y]


def _normals(p):
    d = np.gradient(p, axis=0)
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    return np.c_[-d[:, 1], d[:, 0]]


def _wiggle(p, amp, seed, waves=(1.5, 3.5, 7.0)):
    """Displace an open curve along its normals by a few seeded sine waves (in cycles over its length)."""
    if not amp:
        return p
    rng = np.random.default_rng(seed)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    u = s / s[-1]
    off = sum(amp * rng.uniform(0.6, 1.0) * (1.5 / f) ** 0.7 * np.sin(2 * np.pi * f * u + rng.uniform(0, 2 * np.pi))
              for f in waves)
    return p + _normals(p) * off[:, None]


def blob(c, r, rot, seed, amp, n=480, K=10, **_):
    """A closed, irregular, smooth outline: an ellipse with seeded low harmonics on its radius."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.ones(n)
    for k in range(2, K + 1):
        rad += amp * rng.normal() / k ** 1.5 * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    x, y = r[0] * rad * np.cos(th), r[1] * rad * np.sin(th)
    a = math.radians(rot)
    return np.c_[c[0] + math.cos(a) * x - math.sin(a) * y, c[1] + math.sin(a) * x + math.cos(a) * y]


def river(spec):
    centre = _wiggle(_spline(spec["ctrl"], 900), spec["wig"], spec["seed"], waves=(2.0, 4.5, 9.0))
    w = np.linspace(*spec["w"], len(centre))
    nrm = _normals(centre)
    poly = np.r_[centre + nrm * (w / 2)[:, None], (centre - nrm * (w / 2)[:, None])[::-1]]
    return centre, w, poly


def coast(spec):
    line = _wiggle(_spline(spec["ctrl"], 900), spec["wig"], spec["seed"], waves=(3.0, 7.0, 13.0, 23.0, 37.0))
    return line, np.r_[line, spec["close"]]


def rdp(p, eps):
    """Ramer-Douglas-Peucker simplification of a polyline."""
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.hypot(*ab)
    d = np.abs(ab[0] * (p - a)[:, 1] - ab[1] * (p - a)[:, 0]) / n if n > 0 else np.hypot(*(p - a).T)
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.r_[rdp(p[:i + 1], eps)[:-1], rdp(p[i:], eps)]
    return np.array([a, b])


def catmull_rom(P, step=0.15, alpha=0.5):
    """Centripetal Catmull-Rom spline through every point of P, sampled about every `step` mm."""
    P = np.asarray(P, float)
    ext = np.r_[[2 * P[0] - P[1]], P, [2 * P[-1] - P[-2]]]
    out = [P[:1]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t1 = max(np.hypot(*(p1 - p0)), 1e-6) ** alpha
        t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-6) ** alpha
        t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-6) ** alpha
        n = max(2, int(np.hypot(*(p2 - p1)) / step) + 1)
        t = np.linspace(t1, t2, n)[1:, None]
        A1 = (t1 - t) / t1 * p0 + t / t1 * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / t2 * A1 + t / t2 * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    return np.concatenate(out)


def _seg_cross(P, Q):
    """Which segments of polyline P properly cross any segment of polyline Q (vectorised)."""
    a, b = P[:-1, None, :], P[1:, None, :]
    c, d = Q[None, :-1, :], Q[None, 1:, :]

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (np.sign(o1) * np.sign(o2) < 0) & (np.sign(o3) * np.sign(o4) < 0)


def _closed(p):
    return np.r_[p, p[:1]]


def _arclen(p):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def _cut(p, cum, a, b):
    """The piece of polyline p between arc lengths a and b, endpoints interpolated."""
    pa = [np.interp(a, cum, p[:, 0]), np.interp(a, cum, p[:, 1])]
    pb = [np.interp(b, cum, p[:, 0]), np.interp(b, cum, p[:, 1])]
    return np.r_[[pa], p[(cum > a + 1e-6) & (cum < b - 1e-6)], [pb]]


def _dense(p, step):
    p = np.asarray(p, float)
    return np.concatenate([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step) + 1)) for a, b in zip(p[:-1], p[1:])])


# ---------------------------------------------------------------------------------------------------- the question
def question():
    """The question, its branches from the start (lists of towns), the right branch's and the model's strongest
    wrong branch's first towns, and the model's final score and probability for each of the start's children."""
    Q = json.loads(QUESTION.read_text())["questions"][str(INDEX)]
    ch = {}
    for u, v in Q["edges"]:
        ch.setdefault(u, []).append(v)
    s, g = Q["start"], Q["goal"]

    def chain(v):
        out = [v]
        while len(ch.get(out[-1], [])) == 1:
            out.append(ch[out[-1]][0])
        assert not ch.get(out[-1]), "branches do not split again in this question"
        return out
    branches = {c: chain(c) for c in ch[s]}
    right = next(c for c, b in branches.items() if g in b)
    fin = dict(zip(Q["names"], Q["readout"][-1]))
    prob = dict(zip(Q["names"], Q["probs"][-1]))
    assert Q["answer"] == right == Q["label"], "the model answers this question right"
    wrong = max((c for c in branches if c != right), key=lambda c: fin[c])
    verdict = {c: dict(score=fin[c], p=prob[c]) for c in branches}
    return Q, branches, right, wrong, verdict


# ---------------------------------------------------------------------------------------------------- the build
def build():
    from matplotlib.path import Path as MPath
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    Q, branches, right, wrong, verdict = question()
    s0 = Q["start"]
    others = sorted(c for c in branches if c not in (right, wrong))
    role = {right: "right", wrong: "wrong", **dict(zip(others, SIDES))}
    lakes = [blob(**k) for k in LAKES]
    centre, rw, rpoly = river(RIVER)
    coastline, sea = coast(COAST)
    woods = [blob(**k) for k in WOODS]
    ridge = blob(**RIDGE)

    nx, ny = int(round(MW / RES)), int(round(MH / RES))
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES)
    pts = np.c_[X.ravel(), Y.ravel()]
    inside = lambda poly: MPath(poly).contains_points(pts).reshape(ny, nx)  # noqa: E731
    lake_m = np.zeros((ny, nx), bool)
    for p in lakes:
        lake_m |= inside(p)
    water = lake_m | inside(sea) | inside(rpoly)
    dist = ndimage.distance_transform_edt(~water) * RES
    rough = np.zeros((ny, nx))
    for p in woods:
        rough += 0.3 * inside(p)
    rough += 0.9 * inside(ridge)

    bridges, corridor = [], np.zeros((ny, nx), bool)
    for near in BRIDGES:
        i = int(np.argmin(np.hypot(*(centre - near).T)))
        t = centre[min(i + 4, len(centre) - 1)] - centre[max(i - 4, 0)]
        t /= np.hypot(*t)
        n = np.array([-t[1], t[0]])
        rel = pts - centre[i]
        corridor |= ((np.abs(rel @ t) <= 0.40) & (np.abs(rel @ n) <= rw[i] / 2 + 2.4)).reshape(ny, nx)
        bridges.append(dict(at=centre[i], river_dir=t, normal=n, width=float(rw[i])))

    rng = np.random.default_rng(SEED)
    noise = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), 4.5 / RES)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    margin = (X > 1.8) & (X < MW - 1.8) & (Y > 1.8) & (Y < MH - 1.8)
    base_ok = ((~water & (dist >= 1.15)) | corridor) & margin
    valley = np.where((dist > 1.6) & (dist < 3.4), -0.4, 0.0)           # roads like the ground just off a shore
    shore = np.where(corridor, 0.0, np.clip((2.0 - dist) / 0.85, 0, 1) * 1.2)
    cost0 = 1.2 + 2.2 * noise + rough + shore + valley

    offs = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2)]
    mids = {(2, 1): [(1, 0), (1, 1)], (1, 2): [(0, 1), (1, 1)], (2, -1): [(1, 0), (1, -1)], (1, -2): [(0, -1), (1, -1)],
            (1, 1): [(1, 0), (0, 1)], (1, -1): [(1, 0), (0, -1)]}
    idx = np.arange(nx * ny).reshape(ny, nx)
    cell = lambda p: idx[min(ny - 1, int(p[1] / RES)), min(nx - 1, int(p[0] / RES))]  # noqa: E731

    def graph(ok, cost):
        rows, cols, wts = [], [], []
        for dx, dy in offs:
            ys0, ys1 = max(0, -dy), ny - max(0, dy)
            A = (slice(ys0, ys1), slice(0, nx - dx))
            B = (slice(ys0 + dy, ys1 + dy), slice(dx, nx))
            good = ok[A] & ok[B]
            for mx, my in mids.get((dx, dy), []):
                good &= ok[ys0 + my:ys1 + my, mx:nx - dx + mx]
            rows.append(idx[A][good])
            cols.append(idx[B][good])
            wts.append((math.hypot(dx, dy) * RES * (cost[A] + cost[B]) / 2)[good])
        return coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nx * ny,) * 2).tocsr()

    def route(a, b, G):
        s, e = cell(a), cell(b)
        D, pred = dijkstra(G, directed=False, indices=s, return_predecessors=True)
        assert np.isfinite(D[e]), f"no route from {np.round(a, 1)} to {np.round(b, 1)}"
        path, k = [], e
        while k != s:
            path.append(k)
            k = pred[k]
        path = [s] + path[::-1]
        return np.c_[(np.array(path) % nx + 0.5) * RES, (np.array(path) // nx + 0.5) * RES]

    hub = np.array(HUB)
    near_hub = np.hypot(X - hub[0], Y - hub[1])
    blocked = np.zeros((ny, nx), bool)                    # roads already laid, with a margin
    towns, roads, rank = {s0: hub}, {}, {}
    for c in [right, wrong] + others:
        w1 = np.array(PLAN[role[c]][0])
        spoke = hub + 3.2 * (w1 - hub) / np.hypot(*(w1 - hub))             # each road leaves the hub straight
        G = graph((base_ok & ~blocked) | (np.hypot(X - spoke[0], Y - spoke[1]) < 1.0), cost0)
        way = [spoke] + [np.array(w) for w in PLAN[role[c]]]
        try:
            legs = [route(a, b, G) for a, b in zip(way[:-1], way[1:])]
        except AssertionError:
            _debug(water, blocked, roads, way)
            raise
        simp = np.concatenate([rdp(leg, 0.55)[(1 if k else 0):] for k, leg in enumerate(legs)])
        simp[0], simp[-1] = way[0], way[-1]
        simp = np.r_[[hub], simp]
        path = catmull_rom(simp)
        # a gentle meander across the road, none at the hub, its end or a bridge; smaller if it nears water
        cum = _arclen(path)
        near_b = np.min([np.hypot(*(path - b["at"]).T) for b in bridges], axis=0)
        taper = np.clip(cum / 5.0, 0, 1) * np.clip((cum[-1] - cum) / 3.0, 0, 1) * np.clip((near_b - 2.0) / 3.0, 0, 1)
        rngm = np.random.default_rng(SEED + 7 * c)
        wave = sum(rngm.uniform(0.6, 1.0) * np.sin(2 * np.pi * cum / lam + rngm.uniform(0, 2 * np.pi))
                   for lam in (13.0, 23.0)) / 1.6
        nrm = _normals(path)
        for amp in (MEANDER, MEANDER * 0.6, MEANDER * 0.35, 0.0):
            trial = path + nrm * (amp * wave * taper)[:, None]
            ii = np.clip((trial[:, 1] / RES).astype(int), 0, ny - 1), np.clip((trial[:, 0] / RES).astype(int), 0, nx - 1)
            if ((dist[ii] >= 1.15) | corridor[ii]).all() and not (blocked[ii] & (near_hub[ii] > 3.6)).any():
                break
        path = trial
        cum = _arclen(path)
        total = cum[-1]
        # six towns along the branch, evenly spaced, except that a town stands on the far side of every bridge
        d = np.r_[0.0, np.full(5, np.nan), total]
        for b in bridges:
            local = centre[np.hypot(*(centre - b["at"]).T) < 3.0]
            hit = np.nonzero(_seg_cross(path, local).any(axis=1))[0]
            if len(hit):
                db = cum[hit[0]] + b["width"] / 2 + 2.0
                k = 1 + int(np.argmin([abs(total * j / 6 - db) for j in range(1, 6)]))
                d[k] = db
                b["branch"] = c
        fixed = [j for j in range(7) if not np.isnan(d[j])]
        for a, b_ in zip(fixed[:-1], fixed[1:]):
            for j in range(a + 1, b_):
                d[j] = d[a] + (d[b_] - d[a]) * (j - a) / (b_ - a)
        jit = np.random.default_rng(SEED + c).uniform(-0.16, 0.16, 7) * total / 6     # towns are not evenly spaced
        for j in range(1, 6):
            if j not in fixed:
                d[j] += jit[j]
        d[1] = max(d[1], 4.4)
        assert np.all(np.diff(d) > 3.0), f"towns crowd on branch {c}: {np.round(np.diff(d), 1)}"
        names_ = [s0] + branches[c]
        for j in range(1, 7):
            towns[names_[j]] = np.array([np.interp(d[j], cum, path[:, 0]), np.interp(d[j], cum, path[:, 1])])
            roads[(names_[j - 1], names_[j])] = _cut(path, cum, d[j - 1], d[j])
            rank[(names_[j - 1], names_[j])] = j
        dense = _dense(path, 0.1)
        mask = np.zeros((ny, nx), bool)
        mask[np.clip((dense[:, 1] / RES).astype(int), 0, ny - 1), np.clip((dense[:, 0] / RES).astype(int), 0, nx - 1)] = True
        blocked |= (ndimage.distance_transform_edt(~mask) * RES < np.minimum(2.2, 0.3 + 0.32 * near_hub)) & (near_hub > 3.6)

    # the checks: no road touches a lake or the sea; roads cross the river only at bridges; no two roads cross
    for (u, v), p in roads.items():
        dense = _dense(p, 0.05)
        for k, lk in enumerate(lakes):
            assert not MPath(lk).contains_points(dense).any(), f"road {u}>{v} enters lake {k}"
            assert not _seg_cross(p, _closed(lk)).any(), f"road {u}>{v} crosses the shore of lake {k}"
        assert not MPath(sea).contains_points(dense).any(), f"road {u}>{v} enters the sea"
        assert not _seg_cross(p, coastline).any(), f"road {u}>{v} crosses the coast"
        wet = MPath(rpoly).contains_points(dense)
        if wet.any():
            assert any(np.hypot(*(dense[wet] - b["at"]).T).max() < b["width"] / 2 + 1.0 for b in bridges), \
                f"road {u}>{v} crosses the river away from a bridge"
        nc = int(_seg_cross(p, centre).sum())
        if nc != int(wet.any()):
            _debug(water, blocked, roads, [p[0], p[-1]])
        assert nc == int(wet.any()), f"road {u}>{v} crosses the river's centre {nc} times, wet: {wet.any()}"
    for t, p in towns.items():
        dd = dist[min(ny - 1, int(p[1] / RES)), min(nx - 1, int(p[0] / RES))]
        assert dd >= 1.0, f"town {t} sits {dd:.2f} mm from water"
    keys = list(roads)
    for i, e in enumerate(keys):
        for f in keys[i + 1:]:
            if not set(e) & set(f):
                assert not _seg_cross(roads[e], roads[f]).any(), f"roads {e} and {f} cross"
    for b in bridges:
        assert "branch" in b, f"the bridge at {np.round(b['at'], 1)} carries no road"
        for e, p in roads.items():
            dd = np.hypot(*(p - b["at"]).T)
            if dd.min() < b["width"]:
                i = int(np.argmin(dd))
                t = p[min(i + 2, len(p) - 1)] - p[max(i - 2, 0)]
                b["road"], b["dir"] = list(e), t / np.hypot(*t)
        assert "road" in b

    for k in ISLANDS:
        isl = blob(**k)
        assert any(MPath(w).contains_points(isl).all() for w in lakes + [sea]), "an island off the water"
    r2 = lambda a: np.round(np.asarray(a, float), 2).tolist()  # noqa: E731
    out = dict(
        note="Map of test question 64 of the demo's set at 49 vertices; written by map_draw.py, do not edit by hand. "
             "Millimetres on an MW x MH sheet, y down; drawn full size the map is MW mm wide.",
        question=dict(index=INDEX, lookahead=Q["lookahead"], tokens=Q["tokens"], source=QUESTION.name),
        size=[MW, MH], seed=SEED, start=s0, goal=Q["goal"], current=Q["current"], answer=Q["answer"],
        branches={str(c): b for c, b in branches.items()}, right=branches[right], wrong=branches[wrong],
        model={str(c): dict(score=round(v["score"], 4), p=v["p"]) for c, v in verdict.items()},
        edges=[list(e) for e in Q["edges"]],
        towns={str(t): r2(p) for t, p in towns.items()},
        roads=[dict(edge=[u, v], rank=rank[(u, v)], path=r2(p)) for (u, v), p in roads.items()],
        bridges=[dict(at=r2(b["at"]), dir=r2(b["dir"]), river_dir=r2(b["river_dir"]), width=round(b["width"], 2),
                      road=b["road"]) for b in bridges],
        water=dict(sea=r2(sea), coast=r2(coastline), lakes=[r2(p) for p in lakes],
                   rivers=[dict(centre=r2(centre), width=r2(rw), outline=r2(rpoly))]),
        woods=[r2(p) for p in woods], ridges=[r2(ridge)], islands=[r2(blob(**k)) for k in ISLANDS],
        names=[dict(text=LAKES[0]["name"], at=r2(LAKES[0]["c"]), rot=0),
               dict(text=COAST["name"], at=r2(COAST["name_at"]), rot=0)])
    rb = branches[right]
    out["join"] = dict(u=s0, k=rb[len(rb) // 2 - 1], v=Q["goal"])
    out["labels"] = place_labels(out, 1.0)
    return out


def _debug(water, blocked, roads, way):
    """On a routing failure: out/map_debug.png shows water, the blocked margins and the roads laid so far."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8 * MH / MW))
    ax.imshow(water * 1.0 + blocked * 0.5, extent=(0, MW, MH, 0), cmap="Greys", vmin=0, vmax=2)
    for p in roads.values():
        ax.plot(p[:, 0], p[:, 1], "r-", lw=1)
    w = np.array(way)
    ax.plot(w[:, 0], w[:, 1], "bo-", ms=3)
    fig.savefig(HERE / "out" / "map_debug.png", dpi=120)
    print("routing failed: see out/map_debug.png")


def load(rebuild=False):
    """The map layout, built (and written to data/map_layout.json) if missing or asked for."""
    if rebuild or not LAYOUT.exists():
        L = build()
        LAYOUT.write_text(json.dumps(L, separators=(",", ":")))
        print("wrote", LAYOUT.relative_to(HERE))
    return json.loads(LAYOUT.read_text())


# ---------------------------------------------------------------------------------------------------- drawing
_PAL = json.loads((DATA / "style" / "palette.json").read_text())
ROSE, COB, INK, GREY = _PAL["rose"], _PAL["cobalt"], _PAL["ink"], _PAL["blue_grey"]
WATER, SHORE, WATER_INK = "#E6ECF8", "#C3CEE5", "#7C90BC"   # water, its shoreline, its names
WOOD, RIDGE_C = "#F2F1EE", "#ECEAE5"
ROAD = "#BAC1CD"
SMALL, LABEL = 6.0, 7.0                              # the paper's text sizes (figs.py)
PT = 72 / 25.4                                       # points per mm
FAMILY = ["Helvetica Neue", "Helvetica", "Arial"]


def _A(a):
    return np.asarray(a, float)


def _mix(a, b, f):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return tuple(a + (b - a) * f)


def scale(ax):
    """Printed millimetres per map millimetre, from the axes' width on the page (the axes spans the sheet)."""
    w_mm = ax.get_position().width * ax.figure.get_size_inches()[0] * 25.4
    x0, x1 = ax.get_xlim()
    return w_mm / abs(x1 - x0)


def _f(s):
    """Line weights shrink more slowly than the map: full at s >= 1, about two thirds at s = 0.4."""
    return min(1.0, 0.45 + 0.55 * s)


def sheet(fig, x, y, w, crop=None):
    """An axes for the map with its top-left corner at (x, y) mm on the page, `w` mm wide; `crop` = (x0, y0, x1,
    y1) in map mm shows only that part."""
    x0, y0, x1, y1 = crop or (0.0, 0.0, MW, MH)
    W_, H_ = fig.get_size_inches() * 25.4
    h = w * (y1 - y0) / (x1 - x0)
    ax = fig.add_axes([x / W_, 1 - (y + h) / H_, w / W_, h / H_])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("none")
    return ax


def base(ax, L, s=None, z=0.0):
    """Land (the white page), the woods and the ridge, and water with a slightly darker shoreline. Shorelines are
    stroked under every water fill, so the river runs into the bay without a seam."""
    from matplotlib.patches import Polygon
    s = scale(ax) if s is None else s
    for p in L["woods"]:
        ax.add_patch(Polygon(_A(p), closed=True, fc=WOOD, ec="none", zorder=z + 1))
    for p in L.get("ridges", []):
        ax.add_patch(Polygon(_A(p), closed=True, fc=RIDGE_C, ec="none", zorder=z + 1))
    W = L["water"]
    shapes = [_A(W["sea"])] + [_A(p) for p in W["lakes"]] + [_A(r["outline"]) for r in W["rivers"]]
    for p in shapes:
        ax.add_patch(Polygon(p, closed=True, fc="none", ec=SHORE, lw=0.75 * _f(s), zorder=z + 2, joinstyle="round"))
    for p in shapes:
        ax.add_patch(Polygon(p, closed=True, fc=WATER, ec="none", zorder=z + 3))
    for p in L.get("islands", []):
        ax.add_patch(Polygon(_A(p), closed=True, fc="white", ec=SHORE, lw=0.6 * _f(s), zorder=z + 3.2))


def names(ax, L, s=None, z=3.5):
    """The lake's and the bay's names, italic, in the water's darker tone (full-size map only)."""
    s = scale(ax) if s is None else s
    if s < 0.8:
        return []
    return [ax.text(*n["at"], n["text"], ha="center", va="center", fontsize=SMALL, fontstyle="italic",
                    color=WATER_INK, rotation=n.get("rot", 0), zorder=z) for n in L.get("names", [])]


def _style(e, highlight, default):
    h = (highlight or {}).get(tuple(e))
    col, a = default, 1.0
    if isinstance(h, (int, float)):
        a = float(h)
    elif isinstance(h, tuple):
        col, a = h
    elif isinstance(h, str):
        col = h
    return col, a


def chevron(ax, p, t, size, col, z, alpha=1.0):
    """A small one-way chevron at p pointing along t (map mm), `size` long."""
    from matplotlib.patches import Polygon
    t = _A(t) / np.hypot(*t)
    n = np.array([-t[1], t[0]])
    tip, back = p + t * size * 0.5, p - t * size * 0.5
    pts = [tip, back + n * size * 0.55, back + t * size * 0.45, back - n * size * 0.55]
    ax.add_patch(Polygon(np.array(pts), closed=True, fc=col, ec="none", zorder=z, alpha=alpha, lw=0))


def _along(p, f):
    """Point and direction at fraction f of polyline p's length, and the length."""
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    d = f * cum[-1]
    i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1))
    u = (d - cum[i]) / seg[i] if seg[i] > 0 else 0.0
    j0, j1 = max(i - 3, 0), min(i + 4, len(p) - 1)
    return p[i] + u * (p[i + 1] - p[i]), p[j1] - p[j0], cum[-1]


def road_width(s, lit, rank=1):
    """Road stroke in points: a quiet hierarchy, wider near the start; coloured roads wider still."""
    return (1.3 if lit else 0.95 - 0.06 * (rank - 1)) * _f(s)


def chevron_spot(L, r):
    """Where a road's one-way chevron goes: its middle, moved off a bridge."""
    p = _A(r["path"])
    for f in (0.5, 0.62, 0.38, 0.72, 0.28):
        c, t, _ = _along(p, f)
        if min((np.hypot(*(c - _A(b["at"]))) for b in L["bridges"]), default=9e9) > 2.0:
            return c, t
    return _along(p, 0.5)[:2]


def roads(ax, L, highlight=None, default=ROAD, s=None, z=4.0, chevrons=None):
    """Every road of the question. A road is `default` grey unless `highlight` maps its edge (u, v) to a colour, an
    alpha, or (colour, alpha); coloured roads are wider and on top. Each road has a thin white casing, a light
    one-way chevron (only when the map is drawn at 60% of full size or more) and, at a bridge, parapets."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    s = scale(ax) if s is None else s
    chevrons = s >= 0.6 if chevrons is None else chevrons
    for r in sorted(L["roads"], key=lambda r: _style(r["edge"], highlight, default)[0] != default):
        e, p = tuple(r["edge"]), _A(r["path"])
        col, a = _style(e, highlight, default)
        lit = col != default
        lw = road_width(s, lit, r.get("rank", 1))
        zz = z + (0.5 if lit else 0.0)
        ax.add_patch(PathPatch(MPath(p), fill=False, ec="white", lw=lw + 0.9 * _f(s), zorder=zz,  # casing
                               capstyle="round", joinstyle="round"))
        ax.plot(p[:, 0], p[:, 1], color=col, lw=lw, alpha=a, zorder=zz + 0.1, solid_capstyle="round",
                solid_joinstyle="round")
        if chevrons:
            c, t = chevron_spot(L, r)
            tone = _mix(col, INK, 0.28) if lit else _mix(col, INK, 0.22)
            chevron(ax, c, t, max(lw, 0.9) / PT / s * 1.55, tone, zz + 0.2, alpha=a)
    for b in L["bridges"]:                            # parapets: two short lines along the road, over the river
        at, d, w = _A(b["at"]), _A(b["dir"]), b["width"]
        nrm = np.array([-d[1], d[0]])
        lit = _style(b["road"], highlight, default)[0] != default
        rk = next(r.get("rank", 1) for r in L["roads"] if r["edge"] == b["road"])
        off = (road_width(s, lit, rk) / PT / s) / 2 + 0.42 * _f(s) / s
        half = w / 2 + 0.55
        for sg in (1, -1):
            q = at + nrm * sg * off
            line = np.array([q - d * half + nrm * sg * 0.3, q - d * (half - 0.3), q + d * (half - 0.3),
                             q + d * half + nrm * sg * 0.3])
            ax.plot(line[:, 0], line[:, 1], color=_mix(ROAD, INK, 0.3), lw=0.4 * _f(s), zorder=z + 1.0,
                    solid_capstyle="butt", solid_joinstyle="miter")


def arc_points(L, u, k, v, n=240):
    """The circle through towns u, k and v, from u over k to v."""
    T = {int(t): _A(p) for t, p in L["towns"].items()}
    a, b, c = T[u], T[k], T[v]
    o = np.linalg.solve(np.array([b - a, c - b]) * 2, np.array([b @ b - a @ a, c @ c - b @ b]))
    ang = lambda p: math.atan2(p[1] - o[1], p[0] - o[0])  # noqa: E731
    t0, t1, t2 = ang(a), ang(b), ang(c)
    tau = 2 * math.pi
    if (t1 - t0) % tau <= (t2 - t0) % tau:
        ts = t0 + np.linspace(0, (t2 - t0) % tau, n)
    else:
        ts = t0 - np.linspace(0, (t0 - t2) % tau, n)
    r = np.hypot(*(a - o))
    return np.c_[o[0] + r * np.cos(ts), o[1] + r * np.sin(ts)]


R_TOWN = 0.78                                        # town radius, mm, at full size


def town_radius(s):
    """Town radius in map mm for the map at scale s."""
    return R_TOWN * (0.6 + 0.4 * min(s, 1.0)) / s


R_HUB = 1.6                                         # the start's disc, mm at full size, its number inside


def hub_radius(s):
    return R_HUB * (0.6 + 0.4 * min(s, 1.0)) / s if s >= 0.8 else town_radius(s) * 1.3


def goal_ring(s):
    return town_radius(s) + 0.55 * _f(s) / s


def arc(ax, L, u=None, k=None, v=None, s=None, col=ROSE, z=7.0, names_=True):
    """The join as a thin dashed arc from u over k to v (default: the layout's join), with a small head at v's
    ring; names_=True writes k and v at their placed anchors (full-size map only)."""
    s = scale(ax) if s is None else s
    J = L["join"]
    u, k, v = u or J["u"], k or J["k"], v or J["v"]
    p = arc_points(L, u, k, v)
    T = {int(t): _A(q) for t, q in L["towns"].items()}
    p = p[(np.hypot(*(p - T[u]).T) > town_radius(s) * 1.8) & (np.hypot(*(p - T[v]).T) > goal_ring(s) + 0.35 / s)]
    ax.plot(p[:, 0], p[:, 1], color=col, lw=0.6 * _f(s), ls=(0, (3.0, 1.8)), zorder=z, dash_capstyle="butt")
    chevron(ax, p[-1], p[-1] - p[-6], 1.3 * _f(s) / s, col, z)
    out = []
    if names_ and abs(s - 1.0) < 1e-6:
        for key, sym in (("k", "$k$"), ("v", "$v$")):
            if key in L["labels"]:
                x, y = L["labels"][key][:2]
                out.append(ax.text(x, y, sym, ha="center", va="center", fontsize=LABEL, color=INK, zorder=z + 2))
    return out


def towns(ax, L, labels="all", marks=True, s=None, z=8.0, ring=None, words=("start", "goal")):
    """Towns as small open circles; the start (the current town) filled in ink and the goal ringed in rose, as in
    Fig. 1a. labels: "all" (every town number, and start and goal), "marks" (only the words start and goal), None,
    or a collection of towns to number. ring: {town: colour} outlines some towns in another colour."""
    from matplotlib.patches import Circle
    s = scale(ax) if s is None else s
    r = town_radius(s)
    lw = 0.55 * _f(s)
    for t, p in L["towns"].items():
        t = int(t)
        cur = t == L["current"]
        ax.add_patch(Circle(p, hub_radius(s) if cur else r, fc=INK if cur else "white",
                            ec=INK if cur else (ring or {}).get(t, INK), lw=lw, zorder=z))
        if t == L["goal"] and marks:
            ax.add_patch(Circle(p, goal_ring(s), fc="none", ec=ROSE, lw=lw, zorder=z))
    if not labels:
        return []
    full = abs(s - 1.0) < 1e-6
    lab = L["labels"] if full else place_labels(L, s, numbers=labels != "marks", join=False, routes=False)
    keep = None if labels == "all" else set() if labels == "marks" else {str(t) for t in labels}
    out = []
    for key, a in lab.items():
        if key in ("k", "v") or key.startswith("route:"):
            continue
        if key in ("start", "goal"):
            if not marks or key not in words:
                continue
        elif keep is not None and key not in keep:
            continue
        inside = key == str(L["current"]) and s >= 0.8
        out.append(ax.text(a[0], a[1], key, ha="center", va="center_baseline" if inside else "center",
                           fontsize=SMALL, color="white" if inside else INK, zorder=z + 1))
    return out


def route_names(ax, L, text=None, s=None, z=9.0):
    """The route names at their placed anchors (full-size map only)."""
    s = scale(ax) if s is None else s
    out = []
    for key in ("right", "wrong"):
        a = L["labels"].get("route:" + key)
        if a and abs(s - 1.0) < 1e-6:
            out.append(ax.text(a[0], a[1], (text or {}).get(key, a[2]), ha="center", va="center", fontsize=SMALL,
                               color=INK, zorder=z, linespacing=1.05))
    return out


# ---------------------------------------------------------------------------------------------------- labels
def measure(txt, size, italic=False):
    """Ink width and height of a text, mm, in the house font (math in $...$ as matplotlib sets it)."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    lines = txt.split("\n")
    ws, hs = [], []
    for ln in lines:
        e = TextPath((0, 0), ln, size=size, prop=FontProperties(family=FAMILY,
                                                                style="italic" if italic else "normal")).get_extents()
        ws.append(e.width / PT)
        hs.append(max(e.height, 0.72 * size) / PT)
    return max(ws), sum(hs) + (len(lines) - 1) * 0.3 * size / PT


def place_labels(L, s, numbers=True, join=True, routes=True):
    """Anchors (centres, map mm) for the town numbers, start and goal, the join's k and v and the route names, for
    the map drawn at scale s: each label goes to the candidate spot that touches the fewest roads, towns and labels
    and keeps off water, most crowded towns first; then two more passes move any label a better spot now suits."""
    from matplotlib.path import Path as MPath
    T = {int(t): _A(p) for t, p in L["towns"].items()}
    W = L["water"]
    road_pts = np.concatenate([_dense(r["path"], 0.15) for r in L["roads"]])
    shore_pts = np.concatenate([_dense(o, 0.3) for o in [W["coast"], *W["lakes"], *[r["outline"] for r in W["rivers"]]]])
    J = L.get("join")
    arc_pts = _dense(arc_points(L, J["u"], J["k"], J["v"]), 0.2) if (join and J) else np.zeros((0, 2))
    wet = [MPath(_A(p)) for p in [W["sea"], *W["lakes"], *[r["outline"] for r in W["rivers"]]]]
    k_ = 1.0 / s
    r = town_radius(s)
    gr = goal_ring(s)
    fixed = [np.r_[p - r - 0.15 * k_, p + r + 0.15 * k_] for p in T.values()]
    hr = hub_radius(s)
    fixed.append(np.r_[T[L["current"]] - hr - 0.1 * k_, T[L["current"]] + hr + 0.1 * k_])
    TP = np.array(list(T.values()))
    fixed.append(np.r_[T[L["goal"]] - gr - 0.1 * k_, T[L["goal"]] + gr + 0.1 * k_])
    if s >= 0.8:
        for n in L.get("names", []):
            w, h = measure(n["text"], SMALL, italic=True)
            fixed.append(np.r_[_A(n["at"]) - (w / 2 + 0.3, h / 2 + 0.2), _A(n["at"]) + (w / 2 + 0.3, h / 2 + 0.2)])

    def count(pts, box, pad):
        x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
        return int(((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)).sum())

    def overl(box, boxes):
        return sum(1 for b in boxes if b[0] < box[2] and b[2] > box[0] and b[1] < box[3] and b[3] > box[1])

    def score(box, a, gap, prefer, others, anchor):
        sc = 3.0 * count(road_pts, box, 0.25 * k_) + 1.0 * count(shore_pts, box, 0.1 * k_)
        sc += 4.0 * count(arc_pts, box, 0.25 * k_)
        grid = np.array([(x_, y_) for x_ in np.linspace(box[0], box[2], 4) for y_ in np.linspace(box[1], box[3], 3)])
        sc += 5.0 * sum(int(m.contains_points(grid).sum()) for m in wet)
        mine = [b for b in fixed if not (anchor is not None and b[0] < anchor[0] < b[2] and b[1] < anchor[1] < b[3])]
        pad = 0.35 * k_
        padded = [np.r_[b[:2] - pad, b[2:] + pad] for b in others]
        sc += 80 * overl(box, padded) + 80 * overl(box, mine)
        if anchor is not None:                                   # a number reads as its nearest town's
            ctr = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            dd = np.hypot(*(TP - ctr).T)
            own = np.hypot(*(anchor - ctr))
            sc += 25 * int((dd < own - 0.3 * k_).sum())
        sc += 0.02 * abs(((a - prefer) + 180) % 360 - 180) + 1.5 * gap
        if box[0] < 0.4 or box[2] > MW - 0.4 or box[1] < 0.4 or box[3] > MH - 0.4:
            sc += 1000
        return sc

    items = []                                                   # key, anchor, text, size, radius, preferred angle
    if numbers:
        crowd = {t: int((np.hypot(*(np.array(list(T.values())) - p).T) < 6.0).sum()) for t, p in T.items()}
        first = [L["goal"]] + ([J["k"]] if J else [])
        rest = sorted((t for t in T if t not in first + [L["current"]]), key=lambda t: (-crowd[t], t))
        if s >= 0.8:                                             # the start's number sits inside its disc
            out_hub = [round(T[L["current"]][0], 2), round(T[L["current"]][1], 2), str(L["current"])]
        else:
            rest.insert(0, L["current"])
        items += [(str(t), T[t], str(t), SMALL, gr if t == L["goal"] else hr if t == L["current"] else r, 45)
                  for t in first + rest]
    items += [("start", T[L["current"]], "start", SMALL, hr, 200), ("goal", T[L["goal"]], "goal", SMALL, gr, 270)]
    if join and J:
        items += [("k", T[J["k"]], "$k$", LABEL, r, 0), ("v", T[J["v"]], "$v$", LABEL, gr, 180)]

    boxes, out = {}, {}
    if numbers and s >= 0.8:
        out[str(L["current"])] = out_hub
    for _ in range(3):
        for key, anchor, txt, size, rad, prefer in items:
            w, h = measure(txt, size)
            w, h = w * k_, h * k_
            others = [b for kk, b in boxes.items() if kk != key]
            best = None
            for a in range(0, 360, 15):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                for gap in ((0.5, 0.85, 1.25) if key not in ("start", "goal") else (0.5, 0.9, 1.4, 2.0)):
                    cx = anchor[0] + ca * (rad + gap * k_ + abs(ca) * w / 2)
                    cy = anchor[1] - sa * (rad + gap * k_ + abs(sa) * h / 2)
                    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                    sc = score(box, a, gap, prefer, others, anchor)
                    if best is None or sc < best[0]:
                        best = (sc, cx, cy, box)
            boxes[key] = best[3]
            out[key] = [round(best[1], 2), round(best[2], 2), txt]
    if routes:
        for key, txt, chain in (("right", "correct\nroute", L["right"]), ("wrong", "strongest\nwrong turn", L["wrong"])):
            w, h = measure(txt, SMALL)
            w, h = w * k_, h * k_
            ch = [L["start"]] + chain
            best = None
            for u, v in list(zip(ch[:-1], ch[1:]))[2:]:
                p = _A(next(q["path"] for q in L["roads"] if q["edge"] == [u, v]))
                for f in (0.3, 0.5, 0.7):
                    c, t, _ = _along(p, f)
                    t = t / np.hypot(*t)
                    n = np.array([-t[1], t[0]])
                    for sg in (1, -1):
                        for d in (0.9, 1.5, 2.3):
                            off = d * k_ + abs(n[0]) * w / 2 + abs(n[1]) * h / 2
                            cx, cy = c[0] + sg * n[0] * off, c[1] + sg * n[1] * off
                            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                            sc = score(box, 0, d, 0, list(boxes.values()), None) + 0.5 * d
                            ctr = np.array([cx, cy])
                            sc += 3.0 * sum(1 for b in boxes.values()
                                            if np.hypot(*(ctr - (b[:2] + b[2:]) / 2)) < (w + h) * 0.75)
                            if best is None or sc < best[0]:
                                best = (sc, cx, cy, box)
            boxes["route:" + key] = best[3]
            out["route:" + key] = [round(best[1], 2), round(best[2], 2), txt]
    return out


# ---------------------------------------------------------------------------------------------------- Theorem 1
def known_from(L, row, layer):
    """The roads on routes from `row` known after `layer` (0 = the input), by Theorem 1's construction: channel R
    starts as I + A (reach in at most 1 step) and each layer squares it, so after layer l it holds reach in at most
    2**l steps. Returns (towns reached, roads on the routes to them)."""
    names_ = sorted({int(t) for t in L["towns"]})
    ix = {t: i for i, t in enumerate(names_)}
    A = np.zeros((len(names_),) * 2, int)
    for u, v in L["edges"]:
        A[ix[u], ix[v]] = 1
    R = ((np.eye(len(names_), dtype=int) + A) > 0).astype(int)
    for _ in range(layer):
        R = ((R @ R) > 0).astype(int)
    reached = {t for t in names_ if R[ix[row], ix[t]] and t != row}
    return reached, {(u, v) for u, v in L["edges"] if (u == row or u in reached) and v in reached}


def standard_highlight(L):
    """The correct route in rose and the model's strongest wrong turn in cobalt."""
    s = L["start"]
    hl = {e: ROSE for e in zip([s] + L["right"][:-1], L["right"])}
    hl.update({e: COB for e in zip([s] + L["wrong"][:-1], L["wrong"])})
    return hl


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figs  # noqa: F401  (house fonts)
    L = load(rebuild=True)
    fig = plt.figure(figsize=(MW / 25.4, MH / 25.4))
    ax = sheet(fig, 0, 0, MW)
    base(ax, L)
    names(ax, L)
    roads(ax, L, standard_highlight(L))
    arc(ax, L)
    towns(ax, L)
    route_names(ax, L)
    (HERE / "out").mkdir(exist_ok=True)
    fig.savefig(HERE / "out" / "map_preview.png", dpi=300)
    print("wrote out/map_preview.png")
