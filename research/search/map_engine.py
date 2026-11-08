"""An automatic road map for any question: the look of map_draw.py, without hand planning.

Input: a question's edge list, start, goal and path so far. Output: a layout in map_layout.json's schema (towns, road
polylines with their direction, bridges, water, woods, names, label anchors), in millimetres on a W x H sheet, y down,
drawable by map_draw's functions and by the demo's SVG.

How a map is made (everything seeded from the question's graph, so a graph always gets the same land):
1. Towns and roads. The graph is spread from the start along a breadth-first spanning tree. Each branch of the start
   gets a sector of the sheet in proportion to its size (and to the room in that direction); its towns are laid one
   road at a time, round robin over all branches, each road a step of about `a` mm whose turn is chosen by a score
   (smooth curvature, stay in the sector, keep clear of other roads, leave room for the rest of the branch), so long
   branches wind through their sector instead of running off the sheet. Roads keep a clearance from every road and
   town they do not meet. Edges off the tree (merges, the random graphs' extra edges) are routed afterwards on a
   raster, around everything; where no way round exists the road crosses another as an overpass. The random graphs
   are placed by stress (Kamada-Kawai) instead, and all their roads routed; the attempt with fewest crossings wins.
   Roads are centripetal Catmull-Rom curves through their towns, continuous through every town of degree two.
2. Land, after the roads, in the room they leave: the sea fills one side up to a coast that keeps off the roads (its
   bays are the gaps between branches); lakes take the cores of the larger interior gaps; a river runs from an edge
   of the sheet to the sea along the valleys, crossing roads only through gates square across a road's middle, where
   a bridge is drawn; woods take some of the dry land left.
3. Label anchors for the demo's drawing size (the sheet drawn DEMO_PX px wide, numbers at 11 px): each number goes to
   the free spot beside its town that touches fewest roads, never another label or town.

Guarantees, asserted for every map: no road enters a lake or the sea; roads cross the river only at bridges, once
each; roads meet only at shared towns, or cross as recorded overpasses, and a planar graph has none; no road passes a
town it does not end at; no two towns or labels overlap at the demo's drawing size.

    python -I map_engine.py all              lay out the 300 questions into static/maps/
    python -I map_engine.py sheet            out/engine_sheet.png, a contact sheet of 24 questions
    python -I map_engine.py one 64           one question, to out/engine_one.png
"""
import hashlib
import json
import math
import pathlib
import sys
from collections import deque

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import map_draw as md  # noqa: E402

TT = HERE.parents[1]                       # the repository root
GRAPHS = TT / "static" / "model" / "graphs_v49.json"
MAPS = TT / "static" / "maps"

# The demo draws the sheet DEMO_PX px wide; its marks, in px (the label checks run at this size).
DEMO_PX = 700.0
TOWN_PX, HUB_PX, RING_PX = 4.5, 9.0, 3.5
FONT_PX, DIGIT_PX, CAP_PX, GAP_PX, LPAD_PX = 11.0, 6.7, 8.0, 2.5, 1.5
RES = 0.3                                  # raster, mm
DEBUG = False


# ---------------------------------------------------------------------------------------------------- the graph
class Graph:
    def __init__(self, edges, start, goal, path=None):
        self.edges = [(int(u), int(v)) for u, v in edges]
        assert len(set(self.edges)) == len(self.edges), "repeated edge"
        self.start, self.goal = int(start), int(goal)
        self.path = [int(x) for x in (path or [start])]
        nodes = {x for e in self.edges for x in e} | {self.start, self.goal} | set(self.path)
        self.nodes = sorted(nodes)
        self.adj = {v: [] for v in self.nodes}
        for u, v in self.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
        key = json.dumps([sorted(self.edges), self.start])
        self.seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        # breadth-first spanning tree from the start
        self.parent, self.depth, order = {self.start: None}, {self.start: 0}, [self.start]
        q = deque([self.start])
        while q:
            u = q.popleft()
            for w in sorted(self.adj[u]):
                if w not in self.parent:
                    self.parent[w], self.depth[w] = u, self.depth[u] + 1
                    order.append(w)
                    q.append(w)
        assert len(order) == len(self.nodes), "the graph is not connected"
        self.order = order
        self.kids = {v: [] for v in self.nodes}
        for v in order[1:]:
            self.kids[self.parent[v]].append(v)
        tree = {frozenset((v, self.parent[v])) for v in order[1:]}
        self.extra = [e for e in self.edges if frozenset(e) not in tree]
        self.size = {}
        for v in reversed(order):
            self.size[v] = 1 + sum(self.size[k] for k in self.kids[v])
        self.rank = cycles = len(self.edges) - len(self.nodes) + 1
        self.cycles = cycles

    def planar(self):
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(self.edges)
        return nx.check_planarity(G)[0]


# ---------------------------------------------------------------------------------------------------- geometry
def seg_dist(X, Y, P, Q):
    """Distance from points (X, Y) to segment PQ."""
    d = Q - P
    L2 = float(d @ d)
    if L2 == 0:
        return np.hypot(X - P[0], Y - P[1])
    t = np.clip(((X - P[0]) * d[0] + (Y - P[1]) * d[1]) / L2, 0, 1)
    return np.hypot(X - P[0] - t * d[0], Y - P[1] - t * d[1])


def cr_segment(p0, p1, p2, p3, step=0.25, alpha=0.5):
    """Centripetal Catmull-Rom between p1 and p2 (p0 and p3 set the tangents)."""
    t1 = max(np.hypot(*(p1 - p0)), 1e-6) ** alpha
    t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-6) ** alpha
    t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-6) ** alpha
    n = max(3, int(np.hypot(*(p2 - p1)) / step) + 1)
    t = np.linspace(t1, t2, n)[:, None]
    A1 = (t1 - t) / t1 * p0 + t / t1 * p1
    A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    B1 = (t2 - t) / t2 * A1 + t / t2 * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


def arclen(p):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def resample(p, step):
    cum = arclen(p)
    n = max(2, int(cum[-1] / step) + 1)
    s = np.linspace(0, cum[-1], n)
    return np.c_[np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])]


def normals(p):
    d = np.gradient(p, axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    return np.c_[-d[:, 1], d[:, 0]]


def crossings(P, Q):
    """Points where polyline P properly crosses polyline Q."""
    hit = md._seg_cross(P, Q)
    out = []
    for i, j in zip(*np.nonzero(hit)):
        a, b, c, d = P[i], P[i + 1], Q[j], Q[j + 1]
        r, s = b - a, d - c
        den = r[0] * s[1] - r[1] * s[0]
        t = ((c - a)[0] * s[1] - (c - a)[1] * s[0]) / den
        out.append(a + t * r)
    return out


def smooth_noise(rng, shape, sigma_cells):
    from scipy import ndimage
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma_cells, mode="wrap")
    return f / (np.abs(f).max() + 1e-9)


# ---------------------------------------------------------------------------------------------------- the sheet
class Sheet:
    """The raster the layout is built on: distance to every road and every town laid so far."""

    def __init__(self, W, H, res=RES):
        self.W, self.H, self.res = W, H, res
        self.nx, self.ny = int(math.ceil(W / res)), int(math.ceil(H / res))
        xs, ys = (np.arange(self.nx) + 0.5) * res, (np.arange(self.ny) + 0.5) * res
        self.X, self.Y = np.meshgrid(xs, ys)
        self.road = np.full((self.ny, self.nx), 1e3)
        self.town = np.full((self.ny, self.nx), 1e3)

    def win(self, lo, hi):
        i0, i1 = max(0, int(lo[1] / self.res)), min(self.ny, int(hi[1] / self.res) + 2)
        j0, j1 = max(0, int(lo[0] / self.res)), min(self.nx, int(hi[0] / self.res) + 2)
        return slice(i0, i1), slice(j0, j1)

    def stamp_seg(self, P, Q, reach):
        w = self.win(np.minimum(P, Q) - reach, np.maximum(P, Q) + reach)
        self.road[w] = np.minimum(self.road[w], seg_dist(self.X[w], self.Y[w], P, Q))

    def stamp_line(self, p, reach):
        for P, Q in zip(p[:-1], p[1:]):
            self.stamp_seg(P, Q, reach)

    def stamp_town(self, P, reach):
        w = self.win(P - reach, P + reach)
        self.town[w] = np.minimum(self.town[w], np.hypot(self.X[w] - P[0], self.Y[w] - P[1]))

    def at(self, field, pts):
        pts = np.atleast_2d(pts)
        i = np.clip((pts[:, 1] / self.res).astype(int), 0, self.ny - 1)
        j = np.clip((pts[:, 0] / self.res).astype(int), 0, self.nx - 1)
        return field[i, j]


# ---------------------------------------------------------------------------------------------------- towns and roads
SIDES = ("S", "E", "W", "N")


def plan(G, rng):
    """Step, sheet size, the sea's side and the usable box for the towns."""
    n = len(G.nodes)
    d0 = len(G.adj[G.start])
    a = float(np.clip(7.6 - 0.032 * n, 5.9, 7.4))
    W = math.sqrt(n * a * a * 3.4 / 0.8) + 14.0
    if d0 > 6:                                            # a big hub: its first towns ring it
        Rh = max(a, d0 * max(3.8, 0.66 * a) / (2 * math.pi))
        deep = max(G.size[k] for k in G.kids[G.start])
        W = max(W, (2 * (Rh * 1.12 + min(deep - 1, 3) * a * 0.9 + 6.0) + 14.0) / 0.8)
    W = float(np.clip(W, 62.0, 112.0))
    H = round(0.8 * W, 1)
    W = round(W, 1)
    side = SIDES[rng.choice(4, p=[0.34, 0.28, 0.26, 0.12])]
    band = (0.13 + 0.05 * rng.random()) * (H if side in "SN" else W)
    m = 3.2
    box = [m, m, W - m, H - m]
    if side == "S":
        box[3] -= band
    elif side == "N":
        box[1] += band
    elif side == "E":
        box[2] -= band
    else:
        box[0] += band
    return a, W, H, side, band, box


def sectors(G, hub, box, order, weights, rng, nang=720):
    """Angular sectors around the hub, one per branch, each holding room in proportion to its weight."""
    ph = np.linspace(0, 2 * np.pi, nang, endpoint=False)
    c, s = np.cos(ph), np.sin(ph)
    with np.errstate(divide="ignore"):
        tx = np.where(c > 0, (box[2] - hub[0]) / c, np.where(c < 0, (box[0] - hub[0]) / c, np.inf))
        ty = np.where(s > 0, (box[3] - hub[1]) / s, np.where(s < 0, (box[1] - hub[1]) / s, np.inf))
    R = np.minimum(tx, ty)
    dens = 0.5 * np.minimum(R, 60.0) ** 1.6
    cum = np.r_[0, np.cumsum(dens)]
    cum /= cum[-1]
    w = np.array([weights[b] for b in order], float)
    edges = np.r_[0, np.cumsum(w)] / w.sum()
    off = rng.random()
    out = {}
    for b, e0, e1 in zip(order, edges[:-1], edges[1:]):
        def ang(f):
            f = (f + off) % 1.0
            return float(np.interp(f, cum, np.r_[ph, 2 * np.pi]))
        a0, a1 = ang(e0), ang(e1)
        if a1 < a0:
            a1 += 2 * np.pi
        mid = ang((e0 + e1) / 2)
        if mid < a0:
            mid += 2 * np.pi
        out[b] = (a0, a1, mid)
    return out


def branch_order(G, rng):
    """The start's branches around it, with branches joined by off-tree edges next to each other."""
    roots = list(G.kids[G.start])
    rng.shuffle(roots)
    top = {}
    for r in roots:
        st = [r]
        while st:
            x = st.pop()
            top[x] = r
            st.extend(G.kids[x])
    link = {r: {} for r in roots}
    for u, v in G.extra:
        if u in top and v in top and top[u] != top[v]:
            a, b = top[u], top[v]
            link[a][b] = link[a].get(b, 0) + 1
            link[b][a] = link[b].get(a, 0) + 1
    order, left = [], set(roots)
    while left:
        start = max(left, key=lambda r: (sum(link[r].values()), -roots.index(r)))
        chain = [start]
        left.discard(start)
        while True:
            nxt = [b for b in link[chain[-1]] if b in left]
            if not nxt:
                break
            b = max(nxt, key=lambda b: link[chain[-1]][b])
            chain.append(b)
            left.discard(b)
        order.extend(chain)
    # spread big branches apart: interleave by size, keeping linked runs together
    return order


class Fail(Exception):
    pass


def grow(G, a, W, H, box, rng):
    """Place every town of the spanning tree and lay its roads (straight steps); returns positions and steps."""
    S = Sheet(W, H, res=0.4)
    c = 0.40 * a                     # clearance from roads and towns one does not meet
    ct = 0.46 * a
    d0 = len(G.adj[G.start])
    big = d0 > 6
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hub = np.array([cx + (box[2] - box[0]) * 0.08 * (rng.random() - 0.5),
                    cy + (box[3] - box[1]) * 0.08 * (rng.random() - 0.5)])
    roots = branch_order(G, rng)
    weights = {r: G.size[r] ** 0.9 + 0.8 for r in roots}
    sec = sectors(G, hub, box, roots, weights, rng)
    Rh = max(a, d0 * max(3.8, 0.66 * a) / (2 * math.pi)) if big else a
    # each spoke from the hub: its angle (its sector's middle) and the radius beyond which it keeps clear of its
    # neighbours (spokes converge on the hub, so they are exempt inside that radius)
    angs = {r: sec[r][2] % (2 * math.pi) for r in roots}
    srt = sorted(roots, key=lambda r: angs[r])
    skip = {}
    for i, r in enumerate(srt):
        if len(srt) == 1:
            gap = math.pi
        else:
            gl = (angs[r] - angs[srt[i - 1]]) % (2 * math.pi)
            gr = (angs[srt[(i + 1) % len(srt)]] - angs[r]) % (2 * math.pi)
            gap = min(gl, gr)
        skip[r] = max(ct * 1.02, 1.08 * c / math.sin(min(gap, math.pi / 2)))
    hub_zone = min(skip.values()) if big else 0.0
    usable = (S.X > box[0]) & (S.X < box[2]) & (S.Y > box[1]) & (S.Y < box[3])
    if big:
        usable &= np.hypot(S.X - hub[0], S.Y - hub[1]) > hub_zone

    psi = smooth_noise(rng, (S.ny, S.nx), 13.0 / S.res)
    gy, gx = np.gradient(psi)
    flow = np.arctan2(-gx, gy)                    # along the level lines of psi
    pos = {G.start: hub}
    steps = {}                       # (parent, child) -> (P, Q)
    S.stamp_town(hub, 4 * a)
    head = {}                        # child -> (heading, last turn, sector)
    for i, r in enumerate(srt):
        head[r] = (angs[r], 0.0, sec[r], i)
    tasks = [(G.start, r) for r in roots]
    A_town = 0.85 * a * 2 * c
    from scipy import ndimage

    def inside(Q):
        return box[0] + 0.3 < Q[0] < box[2] - 0.3 and box[1] + 0.3 < Q[1] < box[3] - 0.3

    def seg_ok(P, Q, skipP):
        L = float(np.hypot(*(Q - P)))
        n = max(3, int(L / 0.25))
        t = np.linspace(0, 1, n)[:, None]
        pts = P + t * (Q - P)
        dP = np.hypot(*(pts - P).T)
        pts = pts[dP > skipP]
        if not len(pts):
            return True
        return bool((S.at(S.road, pts) > c).all() and (S.at(S.town, pts[np.hypot(*(pts - Q).T) > 0.01]) > ct * 0.9).all())

    rounds = 0
    while tasks:
        rounds += 1
        free = (np.minimum(S.road, S.town) > c) & usable
        lab, nl = ndimage.label(free)
        area = np.bincount(lab.ravel(), minlength=nl + 1) * S.res ** 2
        area[0] = 0
        tasks.sort(key=lambda t: -G.size[t[1]])
        nxt = []
        for (u, v) in tasks:
            P = pos[u]
            hd, lt, sc, idx = head[v]
            root = u == G.start
            placed_nb = [pos[w] for w in G.adj[v] if w in pos and w != u]
            best = None
            if root:
                cands = []
                base = max(skip[v] + ct * 1.1, a * 0.95)
                if big:
                    wob = 1.0 + 0.12 * (1 if idx % 2 else -1) * (0.6 + 0.4 * rng.random())
                    Ls = [max(Rh * wob, base) * f for f in (1.0, 1.07, 0.95, 1.15, 1.25)]
                    js = [0.0]
                else:
                    Ls = [base * f for f in (1.0, 1.15, 1.3, 1.5)]
                    js = [0.0, -0.06, 0.06]
                for j in js:
                    for L in Ls:
                        cands.append((hd + j, L, 0.0))
            else:
                cands = [(hd + math.radians(t), a * f, math.radians(t))
                         for t in (0, -15, 15, -30, 30, -45, 45, -60, 60, -80, 80)
                         for f in (0.9, 1.0, 1.15, 1.35)]
            need = G.size[v] - 1
            why = {}
            for ang, L, turn in cands:
                Q = P + L * np.array([math.cos(ang), math.sin(ang)])
                if not inside(Q):
                    why["edge"] = why.get("edge", 0) + 1
                    continue
                if S.at(S.town, Q)[0] < ct * 1.25 or S.at(S.road, Q)[0] < ct:
                    why["town"] = why.get("town", 0) + 1
                    continue
                if not seg_ok(P, Q, skip[v] if root else min(c * 1.25, L * 0.5)):
                    why["road"] = why.get("road", 0) + 1
                    continue
                k = lab[min(S.ny - 1, int(Q[1] / S.res)), min(S.nx - 1, int(Q[0] / S.res))]
                room = area[k]
                if need and room < need * A_town:
                    why["room"] = why.get("room", 0) + 1
                    continue
                pol = math.atan2(Q[1] - hub[1], Q[0] - hub[0])
                mid = sc[2]
                dev = (pol - mid + math.pi) % (2 * math.pi) - math.pi
                half = (sc[1] - sc[0]) / 2
                out = max(0.0, abs(dev) - half)
                sco = 2.2 * (math.degrees(out) / 25) ** 2
                if not root:
                    sco += 0.9 * (math.degrees(turn) / 60) ** 2 + 0.7 * (math.degrees(turn - 0.55 * lt) / 60) ** 2
                    sco += 0.25 * abs(L / a - 1.05)
                    rP, rQ = np.hypot(*(P - hub)), np.hypot(*(Q - hub))
                    if G.depth[v] <= 3 and rQ < rP + 0.3 * L:
                        sco += 1.2
                    if G.depth[v] >= 3:                  # deep towns do not curl back to the start
                        rmin = min(0.5 * G.depth[v] * a, 0.34 * min(W, H))
                        sco += 3.0 * max(0.0, rmin - rQ) / a
                    if G.depth[v] >= 3:                  # follow the land's valleys (a divergence-free flow)
                        fa = S.at(flow, (P + Q) / 2)[0]
                        sco += 1.1 * (1 - abs(math.cos(ang - fa)))
                edge = min(Q[0] - box[0], box[2] - Q[0], Q[1] - box[1], box[3] - Q[1])
                sco += 3.0 * max(0.0, 1.2 * a - edge) / a
                fr = min(S.at(S.road, Q)[0], S.at(S.town, Q)[0])
                sco += 1.6 * max(0.0, 1.8 * c - fr) / c
                if need:
                    sco += 0.6 * max(0.0, 1.0 - room / (need * A_town * 2.5))
                for nb in placed_nb:
                    sco += 0.9 * np.hypot(*(Q - nb)) / a
                sco += 0.55 * rng.gumbel()
                if best is None or sco < best[0]:
                    best = (sco, Q, ang, turn)
            if best is None:
                raise Fail(f"no room for {u}>{v} ({why})")
            _, Q, ang, turn = best
            pos[v] = Q
            steps[(u, v)] = (P, Q)
            S.stamp_seg(P, Q, 3 * a)
            S.stamp_town(Q, 3 * a)
            kids = G.kids[v]
            spread = {1: [0], 2: [-38, 38], 3: [-55, 0, 55], 4: [-70, -24, 24, 70]}.get(len(kids))
            if spread is None:
                spread = list(np.linspace(-85, 85, len(kids)))
            order = list(kids)
            rng.shuffle(order)
            for k, s_ in zip(order, spread):
                head[k] = (ang + math.radians(s_), turn if len(kids) == 1 else 0.0, sc, idx)
                nxt.append((v, k))
        tasks = nxt
    return hub, pos, steps, S, dict(c=c, ct=ct, hub_zone=hub_zone, Rh=Rh, big=big)


def route(S, P, Q, c, ct, rng, avoid_town=None, cross_cost=None, ends=()):
    """A road from P to Q on the raster, keeping c from roads and ct from towns (except near its own ends). With
    cross_cost, it may pass over other roads at that price; the caller records the crossings."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    ny, nx = S.ny, S.nx
    near = np.zeros((ny, nx), bool)
    for E in [P, Q, *ends]:
        near |= np.hypot(S.X - E[0], S.Y - E[1]) < ct * 1.15
    town_ok = (S.town > ct) | near
    road_ok = (S.road > c) | near
    ok = town_ok & (road_ok if cross_cost is None else True)
    ok &= (S.X > 1.5) & (S.X < S.W - 1.5) & (S.Y > 1.5) & (S.Y < S.H - 1.5)
    crowd = np.clip((1.6 * c - np.minimum(S.road, S.town)) / c, 0, 2)
    cost = 1.0 + 0.3 * crowd + (0.0 if cross_cost is None else np.where(road_ok, 0.0, cross_cost))
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, wts = [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        ys0, ys1 = max(0, -dy), ny - max(0, dy)
        A = (slice(ys0, ys1), slice(0, nx - dx))
        B = (slice(ys0 + dy, ys1 + dy), slice(dx, nx))
        good = ok[A] & ok[B]
        rows.append(idx[A][good])
        cols.append(idx[B][good])
        wts.append((math.hypot(dx, dy) * S.res * (cost[A] + cost[B]) / 2)[good])
    Gm = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2).tocsr()
    cell = lambda p: idx[min(ny - 1, int(p[1] / S.res)), min(nx - 1, int(p[0] / S.res))]  # noqa: E731
    s, e = cell(P), cell(Q)
    D, pred = dijkstra(Gm, directed=False, indices=s, return_predecessors=True)
    if not np.isfinite(D[e]):
        return None
    path, k = [], e
    while k != s:
        path.append(k)
        k = pred[k]
    path = [s] + path[::-1]
    pts = np.c_[(np.array(path) % nx + 0.5) * S.res, (np.array(path) // nx + 0.5) * S.res]
    pts[0], pts[-1] = P, Q
    simp = md.rdp(pts, 0.5)
    return simp


def stress_layout(G, box, rng):
    """Town positions for a graph with many cycles: Kamada-Kawai from a random start, fitted to the box."""
    import networkx as nx
    Gx = nx.Graph()
    Gx.add_nodes_from(G.nodes)
    Gx.add_edges_from(G.edges)
    init = {v: rng.random(2) for v in G.nodes}
    p = nx.kamada_kawai_layout(Gx, pos=init)
    P = np.array([p[v] for v in G.nodes])
    ang = rng.random() * 2 * np.pi
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    P = (P - P.mean(0)) @ R.T
    lo, hi = P.min(0), P.max(0)
    bw, bh = box[2] - box[0], box[3] - box[1]
    k = min(bw * 0.86 / (hi - lo)[0], bh * 0.86 / (hi - lo)[1])
    P = (P - (lo + hi) / 2) * k + np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return {v: P[i] for i, v in enumerate(G.nodes)}


def lay_roads(G, a, W, H, box, rng, mesh=False):
    """Towns and road polylines (unsmoothed control points) for a graph, and the overpasses."""
    if mesh or G.cycles >= 0.2 * len(G.nodes):
        return lay_mesh(G, a, W, H, box, rng)
    hub, pos, steps, S, k = grow(G, a, W, H, box, rng)
    ctrl = {e: [P, Q] for e, (P, Q) in steps.items()}
    if not G.extra:
        return pos, ctrl, k
    # merges: let the towns settle with the extra roads as springs, never adding a crossing, then lay the extra
    # roads straight where they are clear and route the rest around everything
    pos = anneal(G, pos, box, a, k["ct"], rng, iters=2500, hard=True, keep=G.start)
    ctrl = {e: [pos[e[0]], pos[e[1]]] for e in steps}
    S = Sheet(W, H, res=0.4)
    for v in G.nodes:
        S.stamp_town(pos[v], 3 * a)
    for e in steps:
        S.stamp_seg(pos[e[0]], pos[e[1]], 3 * a)
    for u, v in sorted(G.extra, key=lambda e: np.hypot(*(pos[e[0]] - pos[e[1]]))):
        P, Q = pos[u], pos[v]
        L = np.hypot(*(Q - P))
        t = np.linspace(0, 1, max(4, int(L / 0.25)))[:, None]
        pts = P + t * (Q - P)
        far = (np.hypot(*(pts - P).T) > k["ct"] * 1.1) & (np.hypot(*(pts - Q).T) > k["ct"] * 1.1)
        if far.any() and (S.at(S.road, pts[far]) > k["c"] * 0.9).all() and (S.at(S.town, pts[far]) > k["ct"]).all():
            p = np.array([P, Q])
        else:
            p = route(S, P, Q, k["c"] * 0.9, k["ct"], rng)
            if p is None:
                p = route(S, P, Q, k["c"] * 0.9, k["ct"], rng, cross_cost=60.0)
                if p is None:
                    raise Fail(f"no route for {u}>{v}")
            if arclen(p)[-1] > max(2.2 * L, L + 3 * a):
                raise Fail(f"the road {u}>{v} goes the long way round")
        ctrl[(u, v)] = list(p)
        S.stamp_line(p, 3 * a)
    return pos, ctrl, k


def seg_pairs_cross(A, B, C, D):
    """For segment arrays AB (m) and CD (n): which pairs properly cross."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    a, b, c, d = A[:, None], B[:, None], C[None], D[None]
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (np.sign(o1) * np.sign(o2) < 0) & (np.sign(o3) * np.sign(o4) < 0)


def straight_stats(G, pos, ct):
    E = G.edges
    A = np.array([pos[u] for u, v in E])
    B = np.array([pos[v] for u, v in E])
    X = seg_pairs_cross(A, B, A, B)
    share = np.array([[bool(set(e) & set(f)) for f in E] for e in E])
    cr = int(np.triu(X & ~share, 1).sum())
    near = 0
    T = np.array([pos[v] for v in G.nodes])
    for (u, v), P, Q in zip(E, A, B):
        d = seg_dist(T[:, 0], T[:, 1], P, Q)
        mask = np.array([w not in (u, v) for w in G.nodes])
        near += int((d[mask] < ct).sum())
    D = np.hypot(*(T[:, None] - T[None]).transpose(2, 0, 1)) + np.eye(len(T)) * 1e3
    return cr, near, float(D.min())


def repel(pos, box, dmin, iters=80):
    V = list(pos)
    P = np.array([pos[v] for v in V], float)
    for _ in range(iters):
        D = P[:, None, :] - P[None, :, :]
        d = np.hypot(D[..., 0], D[..., 1]) + np.eye(len(V)) * 1e3
        push = np.clip(dmin - d, 0, None) / np.maximum(d, 1e-6)
        P += 0.5 * (D * push[..., None]).sum(1)
        P[:, 0] = np.clip(P[:, 0], box[0] + 1, box[2] - 1)
        P[:, 1] = np.clip(P[:, 1], box[1] + 1, box[3] - 1)
    return {v: P[i] for i, v in enumerate(V)}


def planar_relax(G, box, rng, ct, edges=None):
    """A crossing-free drawing of a planar graph (or of the planar part `edges` of one): its combinatorial embedding
    on a grid, then relaxed toward neighbours' centres and apart from each other, a move kept only if it adds no
    crossing."""
    import networkx as nx
    edges = list(edges or G.edges)
    Gx = nx.Graph()
    Gx.add_nodes_from(G.nodes)
    Gx.add_edges_from(edges)
    ok, emb = nx.check_planarity(Gx)
    p = nx.combinatorial_embedding_to_pos(emb)
    P = np.array([p[v] for v in G.nodes], float)
    lo, hi = P.min(0), P.max(0)
    bw, bh = box[2] - box[0], box[3] - box[1]
    P = (P - lo) / np.maximum(hi - lo, 1e-6) * [bw * 0.9, bh * 0.9] + [box[0] + bw * 0.05, box[1] + bh * 0.05]
    pos = {v: P[i] for i, v in enumerate(G.nodes)}
    idx = {v: i for i, v in enumerate(G.nodes)}
    nb = {v: [] for v in G.nodes}
    for x, y in edges:
        nb[x].append(y)
        nb[y].append(x)

    def crosses(v, q):
        if not nb[v]:
            return False
        A = np.array([q for _ in nb[v]])
        B = np.array([pos[w] for w in nb[v]])
        others = [(x, y) for x, y in edges if v not in (x, y)]
        C = np.array([pos[x] for x, y in others])
        D = np.array([pos[y] for x, y in others])
        X = seg_pairs_cross(A, B, C, D)
        sh = np.array([[w in (x, y) for x, y in others] for w in nb[v]])
        return bool((X & ~sh).any())
    for it in range(1200):
        v = G.nodes[rng.integers(len(G.nodes))]
        cen = np.mean([pos[w] for w in nb[v]], axis=0) if nb[v] else pos[v]
        rep = np.zeros(2)
        for w in G.nodes:
            if w != v:
                d = pos[v] - pos[w]
                r = np.hypot(*d)
                rep += d / max(r, 1e-3) * (ct * 3.0 / max(r, 1e-3)) ** 2
        step = 0.35 * (cen - pos[v]) + 0.6 * rep + rng.normal(0, 0.3, 2)
        for f in (1.0, 0.5, 0.25):
            q = pos[v] + f * step
            q = np.clip(q, [box[0] + 1, box[1] + 1], [box[2] - 1, box[3] - 1])
            if not crosses(v, q):
                pos[v] = q
                break
    return pos


def tutte_starts(G, box, edges=None, n=4):
    """Tutte (barycentric) drawings of a planar graph, each with a different small face of its embedding as the
    outer polygon (faces whose towns have few roads first): crossing-free for 3-connected graphs."""
    import networkx as nx
    edges = list(edges or G.edges)
    Gx = nx.Graph()
    Gx.add_nodes_from(G.nodes)
    Gx.add_edges_from(edges)
    ok, emb = nx.check_planarity(Gx)
    faces, seen = [], set()
    for v, w in emb.edges():
        if (v, w) in seen:
            continue
        f = emb.traverse_face(v, w, mark_half_edges=seen)
        if 3 <= len(f) <= 9 and len(set(f)) == len(f):
            faces.append(f)
    deg = dict(Gx.degree())
    faces.sort(key=lambda f: (max(deg[x] for x in f), -len(f)))
    V = list(G.nodes)
    ix = {v: i for i, v in enumerate(V)}
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx, ry = (box[2] - box[0]) * 0.46, (box[3] - box[1]) * 0.46
    out = []
    for f in faces[:n]:
        Lm = np.zeros((len(V), len(V)))
        for x, y in edges:
            Lm[ix[x], ix[y]] = Lm[ix[y], ix[x]] = -1
        np.fill_diagonal(Lm, -Lm.sum(1))
        P = np.zeros((len(V), 2))
        fixed = np.zeros(len(V), bool)
        for j, x in enumerate(f):
            th = 2 * np.pi * j / len(f)
            P[ix[x]] = [cx + rx * math.cos(th), cy + ry * math.sin(th)]
            fixed[ix[x]] = True
        free = ~fixed
        A = Lm[np.ix_(free, free)] + np.eye(free.sum()) * 1e-9
        b = -Lm[np.ix_(free, fixed)] @ P[fixed]
        try:
            P[free] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        out.append({v: P[ix[v]] for v in V})
    return out


def bow(P, Q, k):
    """A gentle arc from P to Q, bulging k x its length to the left (k < 0: right)."""
    d = Q - P
    n = np.array([-d[1], d[0]])
    t = np.linspace(0, 1, max(8, int(np.hypot(*d) / 0.4)))[:, None]
    C = (P + Q) / 2 + n * k * 2
    return (1 - t) ** 2 * P + 2 * (1 - t) * t * C + t ** 2 * Q


def anneal(G, pos, box, a, ct, rng, iters=3000, hard=False, keep=None):
    """Simulated annealing on town positions: few crossings first, then roads near their length, towns apart and
    clear of roads they do not meet."""
    V = list(G.nodes)
    ix = {v: i for i, v in enumerate(V)}
    P = np.array([pos[v] for v in V], float)
    E = np.array([[ix[u], ix[v]] for u, v in G.edges])
    nb = [np.array([ix[w] for w in set(G.adj[v])]) for v in V]
    inc = [np.nonzero((E[:, 0] == i) | (E[:, 1] == i))[0] for i in range(len(V))]
    L0, dmin = 1.25 * a, 1.0 * a
    lo = np.array([box[0] + 1.0, box[1] + 1.0])
    hi = np.array([box[2] - 1.0, box[3] - 1.0])

    def pt_seg(Pt, A, B):
        d = B - A
        L2 = np.maximum((d * d).sum(-1), 1e-9)
        t = np.clip(((Pt - A) * d).sum(-1) / L2, 0, 1)
        r = Pt - A - t[..., None] * d
        return np.hypot(r[..., 0], r[..., 1])

    others = [np.array([k for k in range(len(E)) if i not in E[k]], int) for i in range(len(V))]
    endmask = []
    for i in range(len(V)):
        m = np.ones((len(V), len(inc[i])), bool)
        for j, k in enumerate(inc[i]):
            m[E[k], j] = False
        endmask.append(m)
    shared = [((E[inc[i], 0][:, None] == E[None, :, 0]) | (E[inc[i], 0][:, None] == E[None, :, 1]) |
               (E[inc[i], 1][:, None] == E[None, :, 0]) | (E[inc[i], 1][:, None] == E[None, :, 1]))
              for i in range(len(V))]
    allv = np.arange(len(V))
    amin = math.radians(32)

    def angpen(x, ys):
        if len(ys) < 2 or len(ys) > 8:
            return 0.0
        th = np.sort(np.arctan2(ys[:, 1] - x[1], ys[:, 0] - x[0]))
        gaps = np.diff(np.r_[th, th[0] + 2 * np.pi])
        return 12 * float((np.clip((amin - gaps) / amin, 0, None) ** 2).sum())

    def local(i, q):
        Q = P.copy()
        Q[i] = q
        dn = np.hypot(*(Q[nb[i]] - q).T)
        e = 3 * ((dn / L0 - 1) ** 2).sum()
        e += angpen(q, Q[nb[i]]) + sum(angpen(Q[w], Q[nb[w]]) for w in nb[i])
        dv = np.hypot(*(Q[allv != i] - q).T)
        e += 8 * (np.clip((dmin * 1.5 - dv) / a, 0, None) ** 2).sum() + 200 * (dv < 0.9 * dmin).sum()
        o = others[i]
        if len(o):
            d1 = pt_seg(q[None], Q[E[o, 0]], Q[E[o, 1]])
            e += 15 * (np.clip((1.5 * ct - d1) / ct, 0, None) ** 2).sum()
        k = inc[i]
        if len(k):
            A, B = Q[E[k, 0]], Q[E[k, 1]]
            d2 = pt_seg(Q[:, None, :], A[None], B[None])
            e += 15 * (np.clip((1.5 * ct - d2[endmask[i]]) / ct, 0, None) ** 2).sum()
            X = seg_pairs_cross(A, B, Q[E[:, 0]], Q[E[:, 1]])
            return e, int((X & ~shared[i]).sum())
        return e, 0

    for it in range(iters):
        T = 2.0 * (0.003 / 2.0) ** (it / iters)
        i = int(rng.integers(len(V)))
        if V[i] == keep:
            continue
        if rng.random() < 0.25 and len(nb[i]):
            q = P[nb[i]].mean(0) + rng.normal(0, 0.7 * a, 2)
        else:
            q = P[i] + rng.normal(0, a * (0.08 + 0.6 * T / 2.0), 2)
        q = np.clip(q, lo, hi)
        (eo, co), (en, cn) = local(i, P[i]), local(i, q)
        if hard and cn > co:
            continue
        old, new = eo + 40 * co, en + 40 * cn
        if new < old or rng.random() < math.exp(-(new - old) / max(T, 1e-6)):
            P[i] = q
    return {v: P[ix[v]] for v in V}


def lay_mesh(G, a, W, H, box, rng):
    """Graphs with many cycles: stress drawings, then annealed toward no crossings (none at all for a planar graph:
    its own planar drawing is the fallback), then each road bowed gently, never into a new crossing."""
    import networkx as nx
    ct = 0.40 * a
    best = None
    planar = G.planar()

    def consider(pos):
        nonlocal best
        cr, near, dmin = straight_stats(G, pos, ct)
        Ls = np.array([np.hypot(*(pos[u] - pos[v])) for u, v in G.edges]) / (1.25 * a)
        sc = (cr, near > 0, float(((Ls - 1) ** 2).sum()) - 3 * min(dmin / a, 1.0))
        if best is None or sc < best[0]:
            best = (sc, pos)
    # (1) the planar part drawn without crossings, the rest added, then annealed never adding a crossing
    for t in range(3 if planar else 6):
        if planar:
            part = list(G.edges)
        else:
            order = list(G.edges)
            rng.shuffle(order)
            H, part = nx.Graph(), []
            H.add_nodes_from(G.nodes)
            for e in order:
                H.add_edge(*e)
                if nx.check_planarity(H)[0]:
                    part.append(e)
                else:
                    H.remove_edge(*e)
        p0 = planar_relax(G, box, rng, ct, edges=part)
        consider(anneal(G, p0, box, a, ct, rng, iters=5000, hard=True))
        if t == 0:
            for p0 in tutte_starts(G, box, edges=part):
                if straight_stats(G, p0, ct)[0] <= len(G.edges) - len(part):
                    consider(anneal(G, p0, box, a, ct, rng, iters=5000, hard=True))
    # (2) stress drawings, annealed freely, then without adding a crossing
    if True:
        starts = [repel(stress_layout(G, box, rng), box, 1.05 * a) for _ in range(6)]
        starts.sort(key=lambda p: straight_stats(G, p, ct)[:2])
        for p0 in starts[:3]:
            p1 = anneal(G, p0, box, a, ct, rng)
            consider(anneal(G, p1, box, a, ct, rng, iters=2000, hard=True))
    (cr, near, _), pos = best
    T = np.array([pos[v] for v in G.nodes])
    lines = {e: bow(pos[e[0]], pos[e[1]], 0.0) for e in G.edges}

    def count(e, line):
        return sum(len(crossings(line, o)) for f, o in lines.items() if f != e and not set(f) & set(e))

    def clear(e, line):
        mask = np.array([w not in e for w in G.nodes])
        if not mask.any():
            return 9e9
        return min(seg_dist(T[mask, 0], T[mask, 1], A, B).min() for A, B in zip(line[:-1], line[1:]))
    def depart(line, t):
        p = line if np.hypot(*(line[0] - pos[t])) < 1e-6 else line[::-1]
        d = p[min(3, len(p) - 1)] - p[0]
        return math.atan2(d[1], d[0])

    def spread(e, line):
        worst = math.pi
        for t in e:
            mine = depart(line, t)
            for f, o in lines.items():
                if f != e and t in f:
                    worst = min(worst, abs((depart(o, t) - mine + math.pi) % (2 * math.pi) - math.pi))
        return worst
    for e in sorted(G.edges, key=lambda e: -np.hypot(*(pos[e[0]] - pos[e[1]]))):
        P, Q = pos[e[0]], pos[e[1]]
        now = count(e, lines[e])
        base = 0.07 * (1 if rng.random() < 0.5 else -1)
        for k in (base, -base, 0.13, -0.13, 0.2, -0.2, 0.0):
            line = bow(P, Q, k)
            if k and spread(e, line) < math.radians(26):
                continue
            if clear(e, line) >= ct and count(e, line) <= now:
                lines[e] = line
                break
    ctrl = {e: [l_[0], l_[len(l_) // 2], l_[-1]] for e, l_ in lines.items()}
    return pos, ctrl, dict(c=0.3 * a, ct=ct, hub_zone=0.0, Rh=a, big=False, lines=lines)


def smooth(G, pos, ctrl, rng, k):
    """Road polylines: centripetal Catmull-Rom through each road's points, continuous through towns of degree two
    (tangent from the neighbour along the road); a road from a junction leaves it straight. A gentle bow on some."""
    if "lines" in k:
        return {e: np.asarray(p, float) for e, p in k["lines"].items()}
    deg = {v: len(G.adj[v]) for v in G.nodes}
    out = {}
    for (u, v), pts in ctrl.items():
        pts = [np.asarray(p, float) for p in pts]

        def far(x, near_pt, other_end):
            if deg[x] == 2 and x != G.start:
                w = next(w for w in G.adj[x] if w != other_end)
                e = (x, w) if (x, w) in ctrl else (w, x)
                q = ctrl[e]
                nb = np.asarray(q[1] if e[0] == x else q[-2], float)
                return nb
            return 2 * pos[x] - near_pt
        P = [far(u, pts[1], v)] + pts + [far(v, pts[-2], u)]
        P = np.array(P)
        seg = [cr_segment(P[i - 1], P[i], P[i + 1], P[i + 2]) for i in range(1, len(P) - 2)]
        line = np.concatenate([s if i == 0 else s[1:] for i, s in enumerate(seg)])
        out[(u, v)] = line
    return out


# ---------------------------------------------------------------------------------------------------- checks
def road_checks(G, pos, roads, over, k, strict=True):
    """Roads meet only at shared towns or at recorded overpasses; they keep off towns they do not end at."""
    from scipy.spatial import cKDTree
    keys = list(roads)
    found = []
    for i, e in enumerate(keys):
        for f in keys[i + 1:]:
            for x in crossings(roads[e], roads[f]):
                shared = set(e) & set(f)
                if shared and min(np.hypot(*(x - pos[s])) for s in shared) < 0.3:
                    continue
                found.append((e, f, x))
    T = np.array([pos[v] for v in G.nodes])
    names = list(G.nodes)
    tree = cKDTree(T)
    for e, p in roads.items():
        d = resample(p, 0.2)
        dist, j = tree.query(d)
        bad = [(names[jj], dd) for dd, jj in zip(dist, j) if names[jj] not in e and dd < 1.25]
        if bad:
            raise Fail(f"road {e} passes town {bad[0][0]} at {bad[0][1]:.2f} mm")
    # near misses: two roads closer than 0.7 mm away from shared towns and crossings read as junctions
    pts, own = [], []
    for i, e in enumerate(keys):
        d = resample(roads[e], 0.25)
        pts.append(d)
        own += [i] * len(d)
    pts = np.concatenate(pts)
    own = np.array(own)
    tree = cKDTree(pts)
    xs = np.array([x for *_, x in found]) if found else np.zeros((0, 2))

    def leave(e, t):
        p = roads[e] if e[0] == t else roads[e][::-1]
        cum = arclen(p)
        q = np.array([np.interp(1.5, cum, p[:, 0]), np.interp(1.5, cum, p[:, 1])])
        d = q - pos[t]
        return d / np.hypot(*d)
    exempt = {}
    for i, e in enumerate(keys):                 # roads that share a town converge on it: exempt them near it
        for f in keys[i + 1:]:
            for t in set(e) & set(f):
                cosd = float(np.clip(leave(e, t) @ leave(f, t), -1, 1))
                half = math.acos(cosd) / 2
                exempt[(e, f)] = exempt[(f, e)] = (t, max(2.2, 0.75 / (2 * max(math.sin(half), 0.02)) + 0.4))
    for i, j in tree.query_pairs(0.7):
        a_, b_ = own[i], own[j]
        if a_ == b_:
            continue
        e, f = keys[a_], keys[b_]
        p = pts[i]
        if (e, f) in exempt:
            t, r = exempt[(e, f)]
            if np.hypot(*(p - pos[t])) < r:
                continue
        if len(xs) and np.hypot(*(xs - p).T).min() < 1.6:
            continue
        raise Fail(f"roads {e} and {f} touch at {np.round(p, 1)}")
    return found


# ---------------------------------------------------------------------------------------------------- towns, roads
def network(G, salt=0, tries=40):
    """The road network: the best of several seeded attempts (fewest crossings, then shortest roads)."""
    best = None
    a0, W, H, side, band, box = plan(G, np.random.default_rng(G.seed + 104729 * salt))
    planar = G.planar()
    mesh = G.cycles >= 0.2 * len(G.nodes)
    if mesh:
        tries = 3
    a = a0
    wins, switched = 0, 0
    for t in range(tries):
        rng = np.random.default_rng(G.seed + 7919 * (t + 1) + 104729 * salt)
        if t and t % 12 == 0 and not mesh:
            a *= 0.94
        # a planar graph the growth cannot draw without a crossing is drawn as a mesh (its planar drawing)
        if not mesh and planar and G.extra and t >= 18 and (best is None or best[0][0] > 0):
            mesh, a, switched = True, a0, t
        if mesh and tries > 3 and t >= switched + 2:
            break
        try:
            pos, ctrl, k = lay_roads(G, a, W, H, box, rng, mesh=mesh)
            E = set(G.edges)                       # every road runs from its edge's first town to its second
            ctrl = {(e if e in E else e[::-1]): (list(p) if e in E else list(p)[::-1]) for e, p in ctrl.items()}
            assert set(ctrl) == E
            if "lines" in k:
                k["lines"] = {(e if e in E else e[::-1]): (p if e in E else p[::-1]) for e, p in k["lines"].items()}
            roads = smooth(G, pos, ctrl, rng, k)
            found = road_checks(G, pos, roads, [], k)
        except Fail as err:
            last = str(err)
            if DEBUG:
                print("  fail", t, round(a, 2), last)
            continue
        score = (len(found), sum(arclen(p)[-1] for p in roads.values()))
        if best is None or score < best[0]:
            best = (score, pos, roads, found, k, a)
        if not found:
            wins += 1
        if G.cycles == 0 and not found:
            break
        if not found and wins >= (1 if mesh else 4):
            break
    if best is None:
        raise Fail(f"no layout: {last}")
    (n_cross, _), pos, roads, found, k, a = best
    if planar and n_cross:
        raise Fail(f"a planar graph drawn with {n_cross} crossings")
    return dict(pos=pos, roads=roads, cross=found, a=a, W=W, H=H, side=side, band=band, box=box, k=k, planar=planar)


# ---------------------------------------------------------------------------------------------------- the land
LAKE_NAMES = ["Long Lake", "Mirror Lake", "Heron Lake", "Clear Lake", "Reed Lake", "Pine Lake", "Otter Lake",
              "Silver Lake", "Swan Lake", "Fern Lake", "Still Lake", "Crane Lake", "Birch Lake", "Echo Lake"]
SEA_NAMES = ["Mill Bay", "Gull Bay", "Salt Bay", "Cove Bay", "Seal Bay", "Shell Bay", "Wren Bay", "Tern Bay",
             "Kelp Bay", "Stone Bay"]


class Land:
    """A raster over the sheet and a margin round it, with the distance to the nearest road or town."""

    def __init__(self, W, H, roads, towns, res=0.3, pad=6.0):
        from scipy import ndimage
        self.W, self.H, self.res, self.pad = W, H, res, pad
        self.nx, self.ny = int((W + 2 * pad) / res), int((H + 2 * pad) / res)
        xs = (np.arange(self.nx) + 0.5) * res - pad
        ys = (np.arange(self.ny) + 0.5) * res - pad
        self.X, self.Y = np.meshgrid(xs, ys)
        mark = np.zeros((self.ny, self.nx), bool)
        for p in roads:
            q = resample(p, res * 0.6)
            self._mark(mark, q)
        tmark = np.zeros_like(mark)
        self._mark(tmark, np.array(towns))
        mark |= tmark
        self.dist = ndimage.distance_transform_edt(~mark) * res
        self.tdist = ndimage.distance_transform_edt(~tmark) * res
        self.sheet = (self.X > 0) & (self.X < W) & (self.Y > 0) & (self.Y < H)

    def _mark(self, m, q):
        i = np.clip(((q[:, 1] + self.pad) / self.res).astype(int), 0, self.ny - 1)
        j = np.clip(((q[:, 0] + self.pad) / self.res).astype(int), 0, self.nx - 1)
        m[i, j] = True

    def at(self, F, pts):
        pts = np.atleast_2d(pts)
        i = np.clip(((pts[:, 1] + self.pad) / self.res).astype(int), 0, self.ny - 1)
        j = np.clip(((pts[:, 0] + self.pad) / self.res).astype(int), 0, self.nx - 1)
        return F[i, j]

    def outline(self, mask, smooth=0.9):
        """The outline of the largest piece of a mask, as a closed polygon in mm (blurred a little, so smooth)."""
        from scipy import ndimage
        import contourpy
        f = ndimage.gaussian_filter(mask.astype(float), smooth / self.res)
        f = np.pad(f, 1)
        lines = contourpy.contour_generator(z=f).lines(0.5)
        if not lines:
            return None
        best = max(lines, key=lambda L: len(L))
        p = (best - 1 + 0.5) * self.res - self.pad
        return md.rdp(p, 0.06)

    def fill(self, poly):
        from matplotlib.path import Path as MPath
        return MPath(poly).contains_points(np.c_[self.X.ravel(), self.Y.ravel()]).reshape(self.ny, self.nx)


def geography(G, N, rng):
    """Sea, lakes, river, woods, islands and names in the room the roads leave."""
    from scipy import ndimage
    W, H, side, a = N["W"], N["H"], N["side"], N["a"]
    roads, pos = N["roads"], N["pos"]
    L = Land(W, H, list(roads.values()), [pos[v] for v in G.nodes])
    n1 = smooth_noise(rng, L.X.shape, 7.0 / L.res)
    n2 = smooth_noise(rng, L.X.shape, 3.0 / L.res)
    out = dict(sea=None, coast=None, lakes=[], rivers=[], woods=[], islands=[], names=[], bridges=[])

    # the sea: everything far enough from the roads on its side, up to a coast that keeps off them
    outer = {"S": L.Y > H, "N": L.Y < 0, "E": L.X > W, "W": L.X < 0}[side]
    depth = {"S": H - L.Y, "N": L.Y, "E": W - L.X, "W": L.X}[side]       # how far inland from the sea's edge
    span = H if side in "SN" else W
    reach = N["band"] + 0.14 * span
    wet = ((L.dist > 3.4 + 1.3 * n1) & (depth < reach * (1 + 0.25 * n1))) | outer
    # only water a broad disc can reach: no fjords between the branches, a coast with a bay or two
    disc = int(round(5.0 / L.res))
    yy, xx = np.mgrid[-disc:disc + 1, -disc:disc + 1]
    st = (xx ** 2 + yy ** 2) <= disc ** 2
    sea = ndimage.binary_opening(wet, structure=st) | outer
    lab, nl = ndimage.label(sea)
    keep = np.unique(lab[outer & (lab > 0)])
    sea = np.isin(lab, keep[keep > 0])
    sea_poly = L.outline(sea & (L.dist > 2.2) | outer, smooth=1.6)
    if sea_poly is not None:
        sm = L.fill(sea_poly)
        if (sm & L.sheet).sum() * L.res ** 2 > 0.02 * W * H:
            out["sea"] = sea_poly
            inside = (sea_poly[:, 0] > -0.4) & (sea_poly[:, 0] < W + 0.4) & (sea_poly[:, 1] > -0.4) & \
                     (sea_poly[:, 1] < H + 0.4)
            # the coast: the longest run of the outline on the sheet
            runs, cur = [], []
            for k_, ok in enumerate(np.r_[inside, inside]):
                if ok:
                    cur.append(k_ % len(sea_poly))
                else:
                    if cur:
                        runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            run = max(runs, key=len) if runs else []
            run = run[:len(sea_poly)]
            out["coast"] = sea_poly[run] if len(run) > 2 else sea_poly
            sea_m = sm
        else:
            sea_m = np.zeros_like(sea)
    else:
        sea_m = np.zeros_like(sea)
    water = sea_m.copy()

    # lakes: the cores of the larger gaps between roads
    edge = np.minimum.reduce([L.X, W - L.X, L.Y, H - L.Y])
    room = np.minimum(L.dist, edge - 1.0)
    interior = (room > 2.6) & ~ndimage.binary_dilation(sea_m, iterations=int(3.0 / L.res)) & L.sheet
    lab, nl = ndimage.label(interior)
    cands = []
    for k_ in range(1, nl + 1):
        comp = lab == k_
        dmax = float(room[comp].max())
        if dmax < 4.6:
            continue
        thr = max(2.8, 0.58 * dmax)
        nn = 0
        ci, cj = np.unravel_index(np.argmax(np.where(comp, room, 0)), room.shape)
        reachR = 1.7 * dmax * (1 + 0.3 * n2)
        core = np.hypot(L.X - L.X[ci, cj], L.Y - L.Y[ci, cj]) < reachR
        for _ in range(6):
            m = comp & core & (room > thr + 0.16 * dmax * n2)
            m = ndimage.gaussian_filter(m.astype(float), 1.3 / L.res) > 0.5
            m &= room > 2.2
            lb, nn = ndimage.label(m)
            if not nn:
                break
            sizes = np.bincount(lb.ravel())[1:]
            m = lb == (1 + int(np.argmax(sizes)))
            area = m.sum() * L.res ** 2
            if area <= 0.06 * W * H:
                break
            thr *= 1.12
        if not nn or area < 14:
            continue
        cands.append((area, m))
    cands.sort(key=lambda t: -t[0])
    total = 0.0
    names_left = list(LAKE_NAMES)
    rng.shuffle(names_left)
    for area, m in cands[:3]:
        if total + area > 0.16 * W * H:
            continue
        poly = L.outline(m, smooth=0.8)
        if poly is None or len(poly) < 8:
            continue
        pm = L.fill(poly)
        if (L.dist[pm] < 1.9).any():
            continue
        out["lakes"].append(poly)
        water |= pm
        total += area

    # the river: from an edge of the sheet to the sea (or the largest lake), along the valleys, over roads only
    # through gates square across a road's middle
    riv = river(G, N, L, water, sea_m, n1, rng)
    if riv:
        out["rivers"].append(riv["spec"])
        out["bridges"] = riv["bridges"]
        water |= riv["mask"]

    # woods: some of the dry land left, never on a road
    dry = ~ndimage.binary_dilation(water, iterations=int(1.2 / L.res)) & L.sheet
    n3 = smooth_noise(rng, L.X.shape, 5.0 / L.res)
    cand = dry & (L.dist > 2.0 + 1.4 * (n3 + 1) / 2)
    lab, nl = ndimage.label(cand)
    comps = []
    for k_ in range(1, nl + 1):
        comp = lab == k_
        area = comp.sum() * L.res ** 2
        if area > 40:
            comps.append((area, comp))
    comps.sort(key=lambda t: -t[0])
    picks = comps[:4]
    rng.shuffle(picks)
    for area, comp in picks[:2]:
        dd = np.where(comp, L.dist, 0)
        i, j = np.unravel_index(np.argmax(dd), dd.shape)
        R = min(13.0, 5.0 + 0.5 * math.sqrt(area))
        disc = np.hypot(L.X - L.X[i, j], (L.Y - L.Y[i, j]) * 1.25) < R * (1 + 0.25 * n2)
        m = comp & disc & (L.dist > 1.6)
        if m.sum() * L.res ** 2 < 25:
            continue
        poly = L.outline(m, smooth=1.0)
        if poly is not None:
            out["woods"].append(poly)

    # an island in the largest lake, if there is room
    if out["lakes"]:
        big = max(out["lakes"], key=lambda p: len(p))
        pm = L.fill(big)
        dd = ndimage.distance_transform_edt(pm) * L.res
        if dd.max() > 3.6:
            i, j = np.unravel_index(np.argmax(dd), dd.shape)
            r = min(1.4, 0.28 * dd.max())
            isl = md.blob((float(L.X[i, j]) + rng.uniform(-1, 1), float(L.Y[i, j]) + rng.uniform(-1, 1)),
                          (r, r * 0.6), rng.uniform(0, 180), int(rng.integers(1000)), 0.15, n=60)
            from matplotlib.path import Path as MPath
            if MPath(big).contains_points(isl).all():
                out["islands"].append(isl)
    # names, for the largest lake and the sea, where the words fit in the water at the demo's size
    k = W / DEMO_PX
    name_px = 10.5
    for kind, poly in ([("lake", max(out["lakes"], key=lambda p: L.fill(p).sum()))] if out["lakes"] else []) + \
            ([("sea", out["sea"])] if out["sea"] is not None else []):
        text = names_left[0] if kind == "lake" else SEA_NAMES[int(rng.integers(len(SEA_NAMES)))]
        pm = L.fill(poly) & L.sheet
        if kind == "sea":
            pm &= (L.X > 1.5) & (L.X < W - 1.5) & (L.Y > 1.5) & (L.Y < H - 1.5)
        w_mm = 0.52 * name_px * len(text) * k + 1.2 * k
        h_mm = 0.95 * name_px * k
        er = ndimage.binary_erosion(pm, structure=np.ones((max(1, int(h_mm / L.res) | 1), max(1, int(w_mm / L.res) | 1))))
        er &= ndimage.distance_transform_edt(pm) * L.res > 0.6
        if out["islands"]:
            for isl in out["islands"]:
                er &= ~ndimage.binary_dilation(L.fill(isl), iterations=int((h_mm / 2 + 0.6) / L.res))
        if er.any():
            dd = ndimage.distance_transform_edt(er)
            i, j = np.unravel_index(np.argmax(dd), dd.shape)
            out["names"].append(dict(text=text, at=[float(L.X[i, j]), float(L.Y[i, j])], rot=0, kind=kind))
    out["land"] = L
    out["water_mask"] = water
    return out


def river(G, N, L, water, sea_m, n1, rng):
    """A river from a sheet edge to the sea, crossing roads only through gates square across a road's middle."""
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    W, H, side = N["W"], N["H"], N["side"]
    roads, pos = N["roads"], N["pos"]
    res = 0.45
    nx, ny = int(W / res), int(H / res)
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * res, (np.arange(ny) + 0.5) * res)
    pts = np.c_[X.ravel(), Y.ravel()]
    fr = L.at(L.dist, pts).reshape(ny, nx)
    ft = L.at(L.tdist, pts).reshape(ny, nx)
    wat = L.at(water, pts).reshape(ny, nx)
    seaR = L.at(sea_m, pts).reshape(ny, nx)
    gates = []
    gate_m = np.zeros((ny, nx), int) - 1
    for e, p in roads.items():
        cum = arclen(p)
        if cum[-1] < 4.6:
            continue
        best = None
        for f in (0.5, 0.42, 0.58, 0.35, 0.65):
            m = np.array([np.interp(f * cum[-1], cum, p[:, 0]), np.interp(f * cum[-1], cum, p[:, 1])])
            dt = min(np.hypot(*(m - pos[t])) for t in G.nodes)
            if dt > 3.4 and (best is None or dt > best[0]):
                best = (dt, m, f)
        if best is None:
            continue
        _, m, f = best
        i0 = int(np.searchsorted(cum, f * cum[-1]))
        t = p[min(i0 + 3, len(p) - 1)] - p[max(i0 - 3, 0)]
        t /= np.hypot(*t)
        n = np.array([-t[1], t[0]])
        rel = np.c_[X.ravel() - m[0], Y.ravel() - m[1]]
        g = ((np.abs(rel @ t) <= 0.5) & (np.abs(rel @ n) <= 2.6)).reshape(ny, nx)
        gate_m[g] = len(gates)
        gates.append(dict(edge=e, at=m, t=t, n=n))
    ok = (ft > 2.8) & ((fr > 1.7) | (gate_m >= 0) | wat)
    edge_d = np.minimum.reduce([X, W - X, Y, H - Y])
    # sources: a few points spread along the edges other than the sea's, away from roads; a river may touch the
    # sheet's edge only near its own source
    src_pts = []
    for s_ in [s_ for s_ in "NSEW" if s_ != side]:
        for f in np.linspace(0.12, 0.88, 7):
            q = {"N": (f * W, 0.2), "S": (f * W, H - 0.2), "W": (0.2, f * H), "E": (W - 0.2, f * H)}[s_]
            src_pts.append(np.array(q))
    entry = np.zeros((ny, nx), bool)
    for q in src_pts:
        entry |= np.hypot(X - q[0], Y - q[1]) < 11.0
    ok &= (edge_d > 8.5) | entry | seaR
    cost = 1.0 + 2.2 * np.exp(-np.clip(fr - 1.7, 0, None) / 2.5) + 0.6 * (L.at(n1, pts).reshape(ny, nx) + 1)
    cost = cost + 6.0 * np.clip((12.0 - edge_d) / 6.0, 0, 1) * ~entry
    cost = np.where(gate_m >= 0, 1.0, cost)
    cost = np.where(wat & ~seaR, 0.6, cost)
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, wts = [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        ys0, ys1 = max(0, -dy), ny - max(0, dy)
        A = (slice(ys0, ys1), slice(0, nx - dx))
        B = (slice(ys0 + dy, ys1 + dy), slice(dx, nx))
        good = ok[A] & ok[B]
        rows.append(idx[A][good])
        cols.append(idx[B][good])
        wts.append((math.hypot(dx, dy) * res * (cost[A] + cost[B]) / 2)[good])
    Gm = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2).tocsr()
    if seaR.any():
        target = idx[seaR & ok]
    else:
        target = None
    edge_cells = []
    for q in src_pts:
        c = (min(ny - 1, int(q[1] / res)), min(nx - 1, int(q[0] / res)))
        if ok[c] and fr[c] > 2.6 and not seaR[c]:
            edge_cells.append(c)
    if not edge_cells:
        return None
    lakeR = wat & ~seaR
    targets = []
    if seaR.any():
        targets.append(idx[seaR & ok])
    if (lakeR & ok).any():                              # a river may also end in a lake
        targets.append(idx[lakeR & ok])
    if not targets:
        return None
    tried = []
    for ti, target in enumerate(targets):
        D, pred = dijkstra(Gm, directed=False, indices=target, return_predecessors=True, min_only=True)[:2]
        for c in edge_cells:
            k = idx[c]
            if not np.isfinite(D[k]):
                continue
            path, kk = [], k
            while kk >= 0 and kk not in path[-1:]:
                path.append(kk)
                if pred[kk] < 0:
                    break
                kk = pred[kk]
            P = np.c_[(np.array(path) % nx + 0.5) * res, (np.array(path) // nx + 0.5) * res]
            cells = np.array(path)
            g_seq = []
            for gid in gate_m.ravel()[cells]:
                if gid >= 0 and (not g_seq or g_seq[-1] != gid):
                    g_seq.append(int(gid))
            cum = arclen(P)
            length = cum[-1]
            nb = len(set(g_seq))
            mid = (cum > 12.0) & (cum < length - 5.0)
            ed = np.minimum.reduce([P[:, 0], W - P[:, 0], P[:, 1], H - P[:, 1]])
            hug = float((ed[mid] < 11.0).mean()) if mid.any() else 1.0
            if DEBUG:
                print("   river cand", c, ti, round(length, 1), nb, round(hug, 2))
            if length < (0.35 if ti == 0 else 0.22) * max(W, H) or nb > 3 or hug > 0.3:
                continue
            sc = (30 if nb == 0 else abs(nb - 1.5) * 10) - min(length, 1.2 * max(W, H)) * 0.15 + 8 * ti
            tried.append((sc + rng.random() * 8, P, g_seq, ti == 0))
    tried.sort(key=lambda t: t[0])
    for _, P, g_seq, to_sea in tried[:10]:
        r = shape_river(G, N, L, P, [gates[g] for g in g_seq], seaR if to_sea else lakeR, X, Y, rng)
        if r:
            return r
    return None


def shape_river(G, N, L, P, gl, seaR, X, Y, rng):
    """Smooth a river path, square across its gates; check it; its outline, bridges and mask."""
    from matplotlib.path import Path as MPath
    W, H = N["W"], N["H"]
    roads, pos = N["roads"], N["pos"]
    # replace the stretch through each gate with two points square across the road
    P = list(P)
    for g in gl:
        d = np.hypot(*(np.array(P) - g["at"]).T)
        i = int(np.argmin(d))
        lo, hi = i, i
        while lo > 0 and d[lo - 1] < 3.4:
            lo -= 1
        while hi < len(P) - 1 and d[hi + 1] < 3.4:
            hi += 1
        a_, b_ = g["at"] + 2.4 * g["n"], g["at"] - 2.4 * g["n"]
        if np.hypot(*(np.array(P[lo]) - a_)) > np.hypot(*(np.array(P[lo]) - b_)):
            a_, b_ = b_, a_
        P = P[:lo] + [a_, b_] + P[hi + 1:]
    P = np.array(P)
    keep = [0]
    fixed = set()
    for g in gl:
        for k_, q in enumerate(P):
            if min(np.hypot(*(q - (g["at"] + 2.4 * g["n"]))), np.hypot(*(q - (g["at"] - 2.4 * g["n"])))) < 1e-6:
                fixed.add(k_)
    simp = [P[0]]
    last = 0
    for k_ in range(1, len(P)):
        if k_ in fixed or k_ == len(P) - 1 or np.hypot(*(P[k_] - P[last])) > 3.2:
            simp.append(P[k_])
            last = k_
    S_ = np.array(simp)
    ext = np.r_[[2 * S_[0] - S_[1]], S_, [2 * S_[-1] - S_[-2]]]
    line = np.concatenate([cr_segment(ext[i - 1], ext[i], ext[i + 1], ext[i + 2], step=0.3)[(1 if i > 1 else 0):]
                           for i in range(1, len(ext) - 2)])
    # extend a little past the sheet edge at the source, and stop a little into the sea
    d0 = line[0] - line[1]
    line = np.r_[[line[0] + d0 / np.hypot(*d0) * 3.0], line]
    if seaR.any():                                         # run on into the water, so the mouth has no seam
        d1 = line[-1] - line[-4]
        line = np.r_[line, [line[-1] + d1 / np.hypot(*d1) * f for f in (1.0, 2.0, 3.0)]]
    w0, w1 = 0.75, 1.35
    cum = arclen(line)
    w = w0 + (w1 - w0) * cum / cum[-1]
    nrm = normals(line)
    outline = np.r_[line + nrm * (w / 2)[:, None], (line - nrm * (w / 2)[:, None])[::-1]]
    # checks: towns stay clear; roads meet the river only at the gates, crossing its centre once there
    T = np.array([pos[v] for v in G.nodes])
    from scipy.spatial import cKDTree
    if cKDTree(T).query(resample(line, 0.3))[0].min() < 2.3:
        return None
    rp = MPath(outline)
    bridges = []
    for e, p in roads.items():
        dense = resample(p, 0.08)
        wet = rp.contains_points(dense)
        xs = crossings(p, line)
        g = next((g for g in gl if tuple(g["edge"]) == tuple(e)), None)
        if g is None:
            if wet.any() or xs:
                return None
            continue
        if len(xs) != 1 or np.hypot(*(dense[wet] - g["at"]).T).max(initial=0) > 2.2:
            return None
        x = xs[0]
        i = int(np.argmin(np.hypot(*(line - x).T)))
        rd = line[min(i + 3, len(line) - 1)] - line[max(i - 3, 0)]
        rd /= np.hypot(*rd)
        j = int(np.argmin(np.hypot(*(p - x).T)))
        td = p[min(j + 2, len(p) - 1)] - p[max(j - 2, 0)]
        td /= np.hypot(*td)
        if abs(float(rd @ td)) > 0.6:                      # cross square-ish (at least about 53 degrees)
            return None
        bridges.append(dict(at=x, dir=td, river_dir=rd, width=float(w[i]), road=list(e)))
    if len(bridges) != len(gl):
        return None
    mask = L.fill(outline)
    return dict(spec=dict(centre=line, width=w, outline=outline), bridges=bridges, mask=mask)


# ---------------------------------------------------------------------------------------------------- labels
def label_size(text, k):
    """A town number's box at the demo's size, mm (k: mm per px)."""
    return (DIGIT_PX * len(text) + 2 * LPAD_PX) * k, (CAP_PX + 2 * LPAD_PX) * k


def label_box(p, ang, gap, rad, w, h):
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    cx = p[0] + ca * (rad + gap + abs(ca) * w / 2)
    cy = p[1] - sa * (rad + gap + abs(sa) * h / 2)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def disc_box_hit(c, r, b):
    qx, qy = np.clip(c[0], b[0], b[2]), np.clip(c[1], b[1], b[3])
    return math.hypot(c[0] - qx, c[1] - qy) < r


def place_labels(G, pos, roads, W, H, water, land, px=DEMO_PX):
    """A spot beside every town for its number at the demo's size: never on another label or town (any town may
    carry the current-town disc), as few roads and as little water under it as possible."""
    k = W / px
    T = {v: np.asarray(pos[v], float) for v in G.nodes}
    rp = np.concatenate([resample(p, 0.3) for p in roads.values()])
    excl = (HUB_PX + 1.8) * k                      # the check allows HUB_PX + 0.5: room for rounding to 0.1 mm
    crowd = {v: sum(1 for w in G.nodes if w != v and np.hypot(*(T[v] - T[w])) < 9.0) for v in G.nodes}
    order = [G.goal, G.start] + sorted((v for v in G.nodes if v not in (G.goal, G.start)),
                                       key=lambda v: (-crowd[v], v))
    out, boxes = {}, {}
    pad = 1.2 * k
    for v in order:
        w, h = label_size(str(v), k)
        rad = (TOWN_PX + RING_PX + 0.5) * k if v == G.goal else (HUB_PX if v == G.start else TOWN_PX) * k
        best = None
        for ang in range(0, 360, 15):
            for gp in (GAP_PX, GAP_PX + 2.0, GAP_PX + 4.5):
                b = label_box(T[v], ang, gp * k, rad, w, h)
                if b[0] < 0.3 or b[1] < 0.3 or b[2] > W - 0.3 or b[3] > H - 0.3:
                    continue
                if any(disc_box_hit(T[u], excl, b) for u in G.nodes if u != v):
                    continue
                if any(b[0] < o[2] + pad and b[2] > o[0] - pad and b[1] < o[3] + pad and b[3] > o[1] - pad
                       for o in boxes.values()):
                    continue
                m = (rp[:, 0] > b[0] - 0.15) & (rp[:, 0] < b[2] + 0.15) & (rp[:, 1] > b[1] - 0.15) & \
                    (rp[:, 1] < b[3] + 0.15)
                sc = 3.0 * m.sum()
                gx = np.linspace(b[0], b[2], 4)
                gy = np.linspace(b[1], b[3], 3)
                wet = land.at(water, np.array([(x, y) for x in gx for y in gy]))
                sc += 2.0 * wet.sum()
                ctr = (b[:2] + b[2:]) / 2
                own = np.hypot(*(ctr - T[v]))
                sc += 25 * sum(1 for u in G.nodes if u != v and np.hypot(*(ctr - T[u])) < own - 0.2)
                sc += 0.015 * abs((ang - 45 + 180) % 360 - 180) + 0.4 * gp / GAP_PX
                if best is None or sc < best[0]:
                    best = (sc, b)
        if best is None:
            raise Fail(f"no room for the label of {v}")
        boxes[v] = best[1]
        c = (best[1][:2] + best[1][2:]) / 2
        out[str(v)] = [round(float(c[0]), 2), round(float(c[1]), 2), str(v)]
    return out


# ---------------------------------------------------------------------------------------------------- the layout
def edge_rank(G, e):
    return int(min(6, max(1, min(G.depth[e[0]], G.depth[e[1]]) + 1)))


def shortest(G, s, t):
    """A shortest directed route from s to t (towns), or None."""
    ch = {v: [] for v in G.nodes}
    for u, v in G.edges:
        ch[u].append(v)
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for w in sorted(ch[u]):
            if w not in prev:
                prev[w] = u
                q.append(w)
    if t not in prev:
        return None
    out = [t]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return out[::-1]


def layout(G, attempts=6, quiet=True, first=0):
    """The map of a question, in map_layout.json's schema (millimetres, y down), every guarantee checked."""
    last = None
    for salt in range(first, first + attempts):
        try:
            N = network(G, salt=salt)
            rng = np.random.default_rng(G.seed + 31 + salt)
            geo = geography(G, N, rng)
            labels = place_labels(G, N["pos"], N["roads"], N["W"], N["H"], geo["water_mask"], geo["land"])
            L = assemble(G, N, geo, labels)
            check(L, G)
            check(compact(L), G)                   # the site's rounded copy must keep every guarantee too
            return L
        except Fail as err:
            last = err
            if not quiet:
                print("  attempt", salt, err)
    raise RuntimeError(f"no map for this question: {last}")


def assemble(G, N, geo, labels):
    r2 = lambda a: np.round(np.asarray(a, float), 2).tolist()  # noqa: E731
    pos, roads = N["pos"], N["roads"]
    over = []
    for e, f, x in N["cross"]:
        hi_, lo_ = (e, f) if edge_rank(G, e) <= edge_rank(G, f) else (f, e)
        over.append(dict(at=r2(x), over=list(hi_), under=list(lo_)))
    W = geo
    cur = G.path[-1]
    route = shortest(G, cur, G.goal)
    L = dict(
        note="A map of one question, written by map_engine.py. Millimetres on a W x H sheet, y down. Every road is "
             "an edge of the question, drawn from its first town to its second.",
        size=[N["W"], N["H"]], seed=G.seed, step=round(N["a"], 3),
        start=G.start, goal=G.goal, current=cur, path=G.path, route=route,
        edges=[list(e) for e in G.edges],
        towns={str(v): r2(pos[v]) for v in G.nodes},
        roads=[dict(edge=list(e), rank=edge_rank(G, e), path=r2(md.rdp(roads[e], 0.03))) for e in G.edges],
        overpasses=over,
        bridges=[dict(at=r2(b["at"]), dir=r2(b["dir"]), river_dir=r2(b["river_dir"]), width=round(b["width"], 2),
                      road=b["road"]) for b in W["bridges"]],
        water=dict(sea=r2(W["sea"]) if W["sea"] is not None else None,
                   coast=r2(W["coast"]) if W["coast"] is not None else None,
                   lakes=[r2(p) for p in W["lakes"]],
                   rivers=[dict(centre=r2(md.rdp(r["centre"], 0.05)), width=[round(float(r["width"][0]), 2),
                                                                           round(float(r["width"][-1]), 2)],
                                outline=r2(md.rdp(r["outline"], 0.04))) for r in W["rivers"]]),
        woods=[r2(p) for p in W["woods"]], ridges=[], islands=[r2(p) for p in W["islands"]],
        names=[dict(text=n["text"], at=r2(n["at"]), rot=0, kind=n["kind"]) for n in W["names"]],
        labels=labels, planar=N["planar"],
        demo=dict(px=DEMO_PX, town=TOWN_PX, hub=HUB_PX, ring=RING_PX, font=FONT_PX, digit=DIGIT_PX, cap=CAP_PX,
                  gap=GAP_PX, pad=LPAD_PX))
    return L


def check(L, G=None):
    """Every guarantee, on the layout as written (so it also checks a layout read back from JSON)."""
    from matplotlib.path import Path as MPath
    A = lambda x: np.asarray(x, float)  # noqa: E731
    Wd, Hd = L["size"]
    towns = {int(t): A(p) for t, p in L["towns"].items()}
    roads = {tuple(r["edge"]): A(r["path"]) for r in L["roads"]}
    Wt = L["water"]
    bodies = ([A(Wt["sea"])] if Wt["sea"] else []) + [A(p) for p in Wt["lakes"]]
    for e, p in roads.items():
        d = resample(p, 0.08)
        for b in bodies:
            if MPath(b).contains_points(d).any() or len(crossings(p, np.r_[b, b[:1]])):
                raise Fail(f"road {e} enters water")
    for t, p in towns.items():
        for b in bodies:
            bb = np.r_[b, b[:1]]
            if MPath(b).contains_point(p) or np.min([seg_dist(np.array([p[0]]), np.array([p[1]]), x, y)[0]
                                                     for x, y in zip(bb[:-1], bb[1:])]) < 1.0:
                raise Fail(f"town {t} in or by the water")
    br = [dict(at=A(b["at"]), width=b["width"], road=tuple(b["road"])) for b in L["bridges"]]
    for r in Wt["rivers"]:
        out, cen = A(r["outline"]), A(r["centre"])
        rp = MPath(out)
        for e, p in roads.items():
            d = resample(p, 0.08)
            wet = rp.contains_points(d)
            xs = crossings(p, cen)
            mine = [b for b in br if b["road"] == e]
            if not mine:
                if wet.any() or xs:
                    raise Fail(f"road {e} crosses the river away from a bridge")
            else:
                if len(xs) != len(mine) or np.hypot(*(d[wet] - mine[0]["at"]).T).max(initial=0) > mine[0]["width"] / 2 + 1.6:
                    raise Fail(f"road {e} meets the river badly at its bridge")
        for t, p in towns.items():
            if np.hypot(*(cen - p).T).min() < 2.0:
                raise Fail(f"town {t} on the river")
    # roads meet only at shared towns or recorded overpasses
    ov = [A(o["at"]) for o in L["overpasses"]]
    keys = list(roads)
    for i, e in enumerate(keys):
        for f in keys[i + 1:]:
            for x in crossings(roads[e], roads[f]):
                sh = set(e) & set(f)
                if sh and min(np.hypot(*(x - towns[s])) for s in sh) < 0.3:
                    continue
                if not any(np.hypot(*(x - o)) < 0.5 for o in ov):
                    raise Fail(f"roads {e} and {f} cross without an overpass")
    if L.get("planar") and ov:
        raise Fail("a planar question drawn with overpasses")
    for e, p in roads.items():
        d = resample(p, 0.2)
        for t, q in towns.items():
            if t not in e and np.hypot(*(d - q).T).min() < 1.2:
                raise Fail(f"road {e} passes town {t}")
    # towns and labels at the demo's size
    k = Wd / L["demo"]["px"]
    excl = (L["demo"]["hub"] + 0.5) * k
    boxes = {}
    for key, (x, y, txt) in L["labels"].items():
        w, h = label_size(txt, k)
        b = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        if b[0] < 0 or b[1] < 0 or b[2] > Wd or b[3] > Hd:
            raise Fail(f"label {key} off the sheet")
        for t, q in towns.items():
            if str(t) != key and disc_box_hit(q, excl, b):
                raise Fail(f"label {key} on town {t}")
        for k2, o in boxes.items():
            if b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1]:
                raise Fail(f"labels {key} and {k2} overlap")
        boxes[key] = b
    tl = list(towns.values())
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            if np.hypot(*(tl[i] - tl[j])) < 2 * (L["demo"]["hub"] + 1) * k:
                raise Fail("two towns overlap at the demo's size")
    assert set(L["labels"]) >= {str(t) for t in towns}, "a town without a label"
    return True


# ---------------------------------------------------------------------------------------------------- drawing
def draw(ax, L, route=True):
    """The map as the demo draws it (marks sized for the sheet drawn DEMO_PX px wide, scaled to this axes): water,
    woods, grey roads with white casings, bridges, overpasses, towns, the start's disc, the goal's ring, numbers."""
    from matplotlib.patches import Circle, Polygon
    A = lambda x: np.asarray(x, float)  # noqa: E731
    Wd, Hd = L["size"]
    ax.set_xlim(0, Wd)
    ax.set_ylim(Hd, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    w_in = ax.get_position().width * ax.figure.get_size_inches()[0]
    pt = w_in * 72 / DEMO_PX                              # points per demo px
    k = Wd / DEMO_PX                                      # mm per demo px
    for p in L["woods"]:
        ax.add_patch(Polygon(A(p), closed=True, fc=md.WOOD, ec="none", zorder=1))
    Wt = L["water"]
    shapes = ([A(Wt["sea"])] if Wt["sea"] else []) + [A(p) for p in Wt["lakes"]] + \
        [A(r["outline"]) for r in Wt["rivers"]]
    for p in shapes:
        ax.add_patch(Polygon(p, closed=True, fc="none", ec=md.SHORE, lw=1.1 * pt, zorder=2, joinstyle="round"))
    for p in shapes:
        ax.add_patch(Polygon(p, closed=True, fc=md.WATER, ec="none", zorder=3))
    for p in L["islands"]:
        ax.add_patch(Polygon(A(p), closed=True, fc="white", ec=md.SHORE, lw=0.9 * pt, zorder=3.2))
    for n in L["names"]:
        ax.text(*n["at"], n["text"], ha="center", va="center", fontsize=10.5 * pt, fontstyle="italic",
                color=md.WATER_INK, zorder=3.5)
    path = L.get("route") or []
    lit = set(zip(path[:-1], path[1:])) if route else set()
    rw, lw_lit = 1.7, 2.6
    for r in sorted(L["roads"], key=lambda r: tuple(r["edge"]) in lit):
        p = A(r["path"])
        on = tuple(r["edge"]) in lit
        z = 5 if on else 4
        ax.plot(p[:, 0], p[:, 1], color="white", lw=((lw_lit if on else rw) + 2.0) * pt, zorder=z,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(p[:, 0], p[:, 1], color=md.ROSE if on else md.ROAD, lw=(lw_lit if on else rw) * pt, zorder=z + 0.1,
                solid_capstyle="round", solid_joinstyle="round")
        c, t, _ = md._along(p, 0.5)
        md.chevron(ax, c, t, 5.0 * k, md._mix(md.ROSE if on else md.ROAD, md.INK, 0.25), z + 0.2)
    for o in L["overpasses"]:                            # the road over: a short white-cased stretch on top
        p = A(next(r["path"] for r in L["roads"] if r["edge"] == o["over"]))
        m = np.hypot(*(p - A(o["at"])).T) < 2.2
        q = p[m]
        on = tuple(o["over"]) in lit
        ax.plot(q[:, 0], q[:, 1], color="white", lw=((lw_lit if on else rw) + 4.0) * pt, zorder=6,
                solid_capstyle="butt")
        ax.plot(q[:, 0], q[:, 1], color=md.ROSE if on else md.ROAD, lw=(lw_lit if on else rw) * pt, zorder=6.1,
                solid_capstyle="butt")
    for b in L["bridges"]:
        at, d, w = A(b["at"]), A(b["dir"]), b["width"]
        nrm = np.array([-d[1], d[0]])
        off = 2.6 * k
        half = w / 2 + 0.5
        for sg in (1, -1):
            q = at + nrm * sg * off
            line = np.array([q - d * half + nrm * sg * 0.3, q - d * (half - 0.3), q + d * (half - 0.3),
                             q + d * half + nrm * sg * 0.3])
            ax.plot(line[:, 0], line[:, 1], color=md._mix(md.ROAD, md.INK, 0.35), lw=0.8 * pt, zorder=5.5)
    for t, p in L["towns"].items():
        t = int(t)
        cur = t == L["current"]
        ax.add_patch(Circle(p, (HUB_PX if cur else TOWN_PX) * k, fc=md.INK if cur else "white", ec=md.INK,
                            lw=1.2 * pt, zorder=8))
        if t == L["goal"]:
            ax.add_patch(Circle(p, (TOWN_PX + RING_PX) * k, fc="none", ec=md.ROSE, lw=1.3 * pt, zorder=8))
        if cur:
            ax.text(*p, str(t), ha="center", va="center_baseline", fontsize=9.5 * pt, color="white", zorder=9,
                    fontweight="medium")
    for key, (x, y, txt) in L["labels"].items():
        if key == str(L["current"]):
            continue
        ax.text(x, y, txt, ha="center", va="center", fontsize=FONT_PX * pt, color=md.INK, zorder=9)


def question(i):
    q = json.loads(GRAPHS.read_text())["questions"][i]
    return Graph(q["edges"], q["start"], q["goal"], q["path"]), q


SHEET = [0, 8, 12, 19, 23, 30, 48, 61, 76, 96, 124, 136, 146, 165, 180, 201, 220, 244, 276, 288, 291, 294, 296, 293]


def contact(indices, out, cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = math.ceil(len(indices) / cols)
    cw = 4.2
    fig = plt.figure(figsize=(cols * cw, rows * cw * 0.86))
    for n, i in enumerate(indices):
        f = MAPS / f"{i}.json"
        L = json.loads(f.read_text()) if f.exists() else layout(question(i)[0])
        r, c = divmod(n, cols)
        ax = fig.add_axes([c / cols + 0.004, 1 - (r + 1) / rows + 0.02 / rows, 1 / cols - 0.008, 0.8 / rows])
        draw(ax, L)
        q = json.loads(GRAPHS.read_text())["questions"][i]
        ax.set_title(f"{i}: lookahead {q['lookahead']}, {len(L['towns'])} towns", fontsize=8, color=md.INK_SOFT
                     if hasattr(md, "INK_SOFT") else "#4C5B78")
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), *L["size"], fill=False, ec="#DDE3EE", lw=0.6))
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print("wrote", out)


def site64():
    """The approved hand-planned map of question 64 (data/map_layout.json, map_draw.py) in the site's form, with
    number spots for the demo's size: the home page draws it, and the demo uses it for question 64."""
    A = json.loads(md.LAYOUT.read_text())
    G, q = question(64)
    assert sorted(map(tuple, A["edges"])) == sorted(G.edges) and A["start"] == G.start and A["goal"] == G.goal
    pos = {int(t): np.asarray(p, float) for t, p in A["towns"].items()}
    roads = {tuple(r["edge"]): np.asarray(r["path"], float) for r in A["roads"]}
    W, H = A["size"]
    land = Land(W, H, list(roads.values()), [pos[v] for v in G.nodes])
    Wt = A["water"]
    water = np.zeros(land.X.shape, bool)
    for poly in [Wt["sea"], *Wt["lakes"], *[r["outline"] for r in Wt["rivers"]]]:
        water |= land.fill(np.asarray(poly, float))
    # the approved map is denser (84 mm wide); the demo draws it at least 800 px wide
    labels = place_labels(G, pos, roads, W, H, water, land, px=800.0)
    L = dict(
        size=[W, H], seed=A.get("seed"), step=6.0, start=G.start, goal=G.goal, current=G.path[-1], path=G.path,
        route=shortest(G, G.path[-1], G.goal), edges=[list(e) for e in G.edges],
        towns={str(t): p for t, p in A["towns"].items()},
        roads=[dict(edge=r["edge"], rank=r.get("rank", 1), path=r["path"]) for r in A["roads"]],
        overpasses=[], bridges=A["bridges"],
        water=dict(sea=Wt["sea"], coast=Wt["coast"], lakes=Wt["lakes"],
                   rivers=[dict(centre=r["centre"], width=[r["width"][0], r["width"][-1]], outline=r["outline"])
                           for r in Wt["rivers"]]),
        woods=A["woods"], ridges=A.get("ridges", []), islands=A.get("islands", []),
        names=[dict(text=n["text"], at=n["at"], rot=n.get("rot", 0), kind="sea" if "Bay" in n["text"] else "lake")
               for n in A.get("names", [])],
        labels=labels, planar=True,
        demo=dict(px=800.0, town=TOWN_PX, hub=HUB_PX, ring=RING_PX, font=FONT_PX, digit=DIGIT_PX, cap=CAP_PX,
                  gap=GAP_PX, pad=LPAD_PX))
    check(L, G)
    C = compact(L)
    check(C, G)
    (MAPS / "64.json").write_text(json.dumps(C, separators=(",", ":")))
    print("wrote", MAPS / "64.json", (MAPS / "64.json").stat().st_size, "bytes")


# ---------------------------------------------------------------------------------------------------- batch
def compact(L):
    """The site's copy: coordinates to 0.1 mm, outlines thinned, nothing the demo does not draw."""
    r1 = lambda a: [[round(x, 1), round(y, 1)] for x, y in np.asarray(a, float).tolist()]  # noqa: E731
    th = lambda a, e: r1(md.rdp(np.asarray(a, float), e))  # noqa: E731
    W = L["water"]
    return dict(
        size=L["size"], step=L["step"], start=L["start"], goal=L["goal"], current=L["current"], path=L["path"],
        route=L["route"],
        towns={t: [round(p[0], 1), round(p[1], 1)] for t, p in L["towns"].items()},
        roads=[dict(edge=r["edge"], rank=r["rank"], path=th(r["path"], 0.04)) for r in L["roads"]],
        overpasses=[dict(at=[round(x, 1) for x in o["at"]], over=o["over"], under=o["under"]) for o in L["overpasses"]],
        bridges=[dict(at=[round(x, 1) for x in b["at"]], dir=[round(x, 3) for x in b["dir"]],
                      river_dir=[round(x, 3) for x in b["river_dir"]], width=b["width"], road=b["road"])
                 for b in L["bridges"]],
        water=dict(sea=th(W["sea"], 0.1) if W["sea"] else None, coast=th(W["coast"], 0.1) if W["coast"] else None,
                   lakes=[th(p, 0.08) for p in W["lakes"]],
                   rivers=[dict(centre=th(r["centre"], 0.08), width=r["width"], outline=th(r["outline"], 0.06))
                           for r in W["rivers"]]),
        woods=[th(p, 0.12) for p in L["woods"]], ridges=[th(p, 0.12) for p in L.get("ridges", [])],
        islands=[th(p, 0.05) for p in L["islands"]],
        names=L["names"], labels={k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in L["labels"].items()},
        planar=L["planar"], demo=L["demo"])


def one(i):
    G, q = question(i)
    t0 = __import__("time").time()
    try:
        L = layout(G)
        C = compact(L)
        check(C, G)                                # the rounded copy keeps every guarantee too
    except Exception as err:                     # reported, not fatal: the batch goes on
        return i, "FAILED", str(err)
    (MAPS / f"{i}.json").write_text(json.dumps(C, separators=(",", ":")))
    return i, round(__import__("time").time() - t0, 1), len(C["water"]["lakes"]), len(C["water"]["rivers"]), \
        len(C["bridges"]), len(C["overpasses"]), C["planar"]


def batch(indices, workers=8):
    from multiprocessing import Pool
    MAPS.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(one, indices):
            rows.append(r)
            print(*r, flush=True)
    total = sum((MAPS / f"{i}.json").stat().st_size for i in indices if (MAPS / f"{i}.json").exists())
    print(f"{len(rows)} maps, {total / 1e6:.2f} MB")
    return rows


if __name__ == "__main__":
    import os
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    if cmd == "all":
        batch(list(range(300)) if len(sys.argv) < 3 else [int(x) for x in sys.argv[2].split(",")])
    elif cmd == "sheet":
        contact(SHEET, HERE / "out" / "engine_sheet.png")
    elif cmd == "site64":
        site64()
    elif cmd == "one":
        i = int(sys.argv[2])
        print(one(i))
        contact([i], HERE / "out" / "engine_one.png", cols=1)
