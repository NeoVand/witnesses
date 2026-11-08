"""Computations and drawing for Fig. 1. Every mark is computed here from the 49-town question's 48 roads and the
eight-card deal; the checks are asserts and run on every build.

The question (data/map_layout.json, data/map_question.json): 49 towns, eight branches of six one-way roads from the
start; the goal ends one branch (lookahead 6). It is a tree, so exactly one walk reaches the goal from any town on that
branch and none from any other town: the propagator's light from the goal covers the one route, and the seven other
branches stay dark.

Hand-set layers (beta = 3, t0 = 0.8, legs of at most 4 roads = Theorem 1's channel after two layers):
- lookup from the start: softmax weights e^{beta A_sv} / Z over the 48 other towns;
- join for (start, goal): Z = sum_k a_sk b_kg, a and b reach in at most 4 roads: witnesses k = 45, 31, 39;
- propagator: X = S (I - G)^{-1}, G = t0 A^T, S = the goal column, solved; X_v = t0^{d(v, goal)} (asserted);
- SET: the card in hand q and the eight table cards, whose four set pairs are asserted by the rule;
  Newton for the character chi_(1,1,1,1): phases phi_j, p1 = sum phi_j, p2 = sum phi_j^2, e2 = (p1^2 - p2)/2 equal to
  the sum of the 28 pair products, and over all 81 characters the count of sets (= 4), both asserted.
"""
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import *  # noqa: E402,F401,F403
from style import house, md, OUT  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

# ---------------------------------------------------------------------------------------------------- the question
BETA, T0, REACH = 3.0, 0.8, 4


def dist_from(s, rev=False):
    M = A.T if rev else A
    d, todo = {s: 0}, [s]
    for u in todo:
        for w in np.flatnonzero(M[IX[u]]):
            w = TOWNS[w]
            if w not in d:
                d[w] = d[u] + 1
                todo.append(w)
    return d


def set_question(L_):
    """Make L_ (map_layout.json's schema: q64, or a map_engine layout) the question every function below draws.
    The current town is the start in both questions used (the path so far is the start alone)."""
    global L, S0, GOAL, TOWNS, IX, N, A, D_START, D_GOAL, KIDS, ROUTE, LIT, SIZE, IS_Q64
    L = L_
    IS_Q64 = "right" in L_
    S0, GOAL = L["current"], L["goal"]
    assert L.get("path", [S0]) == [S0] and L["start"] == S0
    TOWNS = sorted(int(t) for t in L["towns"])
    IX = {t: i for i, t in enumerate(TOWNS)}
    N = len(TOWNS)
    A = np.zeros((N, N))
    for u, v in L["edges"]:
        A[IX[u], IX[v]] = 1
    D_START, D_GOAL = dist_from(S0), dist_from(GOAL, rev=True)
    KIDS = [TOWNS[i] for i in np.flatnonzero(A[IX[S0]])]
    ROUTE = [S0] + L["right"] if IS_Q64 else [int(t) for t in L["route"]]       # one shortest route, start to goal
    assert ROUTE[0] == S0 and ROUTE[-1] == GOAL and len(ROUTE) - 1 == D_START[GOAL]
    LIT = [(int(u), int(v)) for u, v in L["edges"] if int(v) in D_GOAL]      # every road on a walk to the goal
    SIZE = tuple(L.get("size", (md.MW, md.MH)))
    if IS_Q64:
        assert len(KIDS) == 8


def lookup_weights():
    w = {t: math.exp(BETA * A[IX[S0], IX[t]]) for t in TOWNS if t != S0}
    Z = sum(w.values())
    out = {t: x / Z for t, x in w.items()}
    assert len({round(out[k], 12) for k in KIDS}) == 1                  # the start's children weigh the same
    return out


def join_witnesses():
    """Middle towns k for the pair (start, goal) with both legs (at most REACH roads each)."""
    a = {k for k, d in D_START.items() if 0 < d <= REACH}
    b = {k for k, d in D_GOAL.items() if 0 < d <= REACH}
    mids = sorted(a & b, key=lambda k: (D_START[k], k))
    if IS_Q64:
        assert len(a) == 32 and len(b) == 4 and mids == [45, 31, 39], (len(a), len(b), mids)
    Z = sum(A_reach(REACH)[IX[S0], IX[k]] * A_reach(REACH)[IX[k], IX[GOAL]] for k in TOWNS if k not in (S0, GOAL))
    assert Z == len(mids)
    return mids, a, b


def A_reach(r):
    """Reach in 1 to r roads, as 0/1 (Theorem 1's channel without the identity)."""
    R = np.zeros((N, N))
    P = np.eye(N)
    for _ in range(r):
        P = (P @ A > 0).astype(float)
        R = np.maximum(R, P)
    return R


def propagator():
    """X = S (I - G)^{-1}, G = t0 A^T, S = the goal column, by one solve; checked against the walks counted by
    length, X_v = sum_k t0^k (A^k)_{v, goal} (the graphs are acyclic, so the count ends at N)."""
    G = T0 * A.T
    S = np.zeros((N, N))
    S[:, IX[GOAL]] = 1.0
    X = S @ np.linalg.solve(np.eye(N) - G, np.eye(N))
    x = {t: X[IX[S0], IX[t]] for t in TOWNS}
    walks, P = np.zeros(N), np.eye(N)
    for k in range(N + 1):
        walks += T0 ** k * P[:, IX[GOAL]]
        P = P @ A
    assert not P.any()                                                  # acyclic: no walk longer than N
    for t in TOWNS:
        assert abs(x[t] - walks[IX[t]]) < 1e-12, (t, x[t], walks[IX[t]])
        assert (x[t] > 1e-12) == (t in D_GOAL)                         # roundoff aside, light only where the goal is reached
        if IS_Q64:
            assert abs(x[t] - (T0 ** D_GOAL[t] if t in D_GOAL else 0.0)) < 1e-12
    return x


set_question(house.maplayout()[0])


# ---------------------------------------------------------------------------------------------------- SET
# The deal. Card = (number, shape, fill, colour)
# as house.set_card draws it: number + 1 symbols; oval, diamond, squiggle; solid, striped, open; blue, green, purple.
# In hand: two solid blue ovals. The table, found by a seeded search (seed 1) for four cards whose completions with the
# card in hand are four more distinct cards, every attribute taking all three values on the table, each fill at least
# twice: four set pairs, side by side; check_deal() proves them the only sets with the card in hand.
YOURS = (1, 0, 0, 0)
TABLE = [(2, 0, 2, 0), (0, 0, 1, 0), (1, 1, 1, 1), (1, 2, 2, 2), (2, 2, 1, 1), (0, 1, 2, 2), (1, 1, 0, 0), (1, 2, 0, 0)]
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
CHAR = (1, 1, 1, 1)


def check_deal():
    assert len(set(TABLE) | {YOURS}) == 9                                   # nine different cards
    sets = {(j, k) for j in range(8) for k in range(j + 1, 8) if house.is_set(YOURS, TABLE[j], TABLE[k])}
    assert sets == set(PAIRS), sets                                         # exactly the four adjacent pairs
    assert sorted(i for pr in PAIRS for i in pr) == list(range(8))          # every table card in exactly one


def newton():
    phi = [house.chi(CHAR, a) for a in TABLE]
    p1, p2 = sum(phi), sum(f * f for f in phi)
    e2 = (p1 * p1 - p2) / 2
    pairs = [(j, k) for j in range(8) for k in range(j + 1, 8)]
    assert abs(sum(phi[j] * phi[k] for j, k in pairs) - e2) < 1e-9
    count = 0
    for sc in np.ndindex(3, 3, 3, 3):
        ph = [house.chi(sc, a) for a in TABLE]
        P1, P2 = sum(ph), sum(f * f for f in ph)
        count += house.chi(sc, YOURS) * (P1 * P1 - P2) / 2
    assert abs(count / 81 - 4) < 1e-9
    return dict(phi=phi, p1=p1, p2=p2, e2=e2, q=house.chi(CHAR, YOURS), sets=round((count / 81).real))


check_deal()

# ---------------------------------------------------------------------------------------------------- page, text
SIZES = (SMALL, LABEL, LETTER, 8.5, 9.0, 12.0, 13.0, 30.0)
TREE = "#C9C4BA"                                         # tree glyphs in the woods (the woods' warm grey, darker)


def save_v(fig, name):
    """house.save (pdf, svg with text as outlines, 300 dpi png; overflow, overlap and on-line checks) with Fig. 1's
    larger type allowed."""
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(plt.Text) if t.get_text().strip()})
    assert all(s in SIZES for s in sizes), sizes
    over = house.layout.overflow(fig)
    if over:
        print(f"WARNING {name}: content past the page edge (mm): {over}")
    for a, b in house.layout.collisions(fig):
        print(f"WARNING {name}: texts overlap: {a!r} / {b!r}")
    for a, what in house.layout.on_marks(fig):
        print(f"WARNING {name}: text on a {what}: {a!r}")
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------- the map
def xy(t):
    return np.asarray(L["towns"][str(t)], float)


def road(u, v):
    return np.asarray(next(r["path"] for r in L["roads"] if r["edge"] == [u, v]), float)


def qmap(fig, x, y, w, roads=True, hl=None, crop=None):
    ax = md.sheet(fig, x, y, w, crop or (0.0, 0.0, *SIZE))
    s = md.scale(ax)
    base_any(ax, s)
    if roads:
        md.roads(ax, L, hl, s=s)
        overpasses(ax, s, hl)
    return ax, s


def base_any(ax, s):
    """mapdraw's land and water for either kind of layout (a map_engine sheet may have no sea)."""
    W_ = L["water"]
    if W_.get("sea") is None:
        L2 = dict(L, water=dict(W_, sea=[[-50.0, -50.0], [-49.0, -50.0], [-49.0, -49.0]]))
        md.base(ax, L2, s)
    else:
        md.base(ax, L, s)


def overpasses(ax, s, hl=None):
    """Where two roads of a non-planar question cross: the road over drawn again on top, with its white casing."""
    for o in L.get("overpasses", []):
        p = road(*o["over"])
        q = p[np.hypot(*(p - np.asarray(o["at"])).T) < 2.0 / max(s, 0.5)]
        col, a = md._style(o["over"], hl, md.ROAD)
        if a == 0.0:                                     # a lit road: its light, drawn above, is already on top
            continue
        lw = md.road_width(s, col != md.ROAD)
        ax.plot(q[:, 0], q[:, 1], color="white", lw=lw + 2.2 * md._f(s), zorder=5.6, solid_capstyle="butt")
        ax.plot(q[:, 0], q[:, 1], color=col, alpha=a, lw=lw, zorder=5.7, solid_capstyle="butt")


def tree_glyphs(ax, s, step=4.6, avoid=()):
    """Small flat trees on a jittered grid inside each wood (mm on the map), a little darker than the wood."""
    rng = np.random.default_rng(7)
    h = 1.25 / s
    for poly in L["woods"]:
        P = np.asarray(poly)
        path = MPath(P)
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        for gx in np.arange(x0 + step / 2, x1, step):
            for gy in np.arange(y0 + step / 2, y1, step * 0.85):
                c = np.array([gx + rng.uniform(-1, 1), gy + rng.uniform(-0.8, 0.8)])
                clear = all(not (b[0] - 0.8 / s < c[0] < b[2] + 0.8 / s and b[1] - 1.4 / s < c[1] < b[3] + 0.6 / s)
                            for b in avoid)
                if clear and path.contains_point(c, radius=-2.2) and min(np.hypot(*(c - xy(t))) for t in TOWNS) > 2.5:
                    ax.add_patch(Polygon([c + (0, -h), c + (-0.42 * h, 0.35 * h), c + (0.42 * h, 0.35 * h)],
                                         closed=True, fc=TREE, ec="none", zorder=1.5))
                    ax.plot([c[0], c[0]], [c[1] + 0.3 * h, c[1] + 0.62 * h], color=TREE, lw=0.4, zorder=1.5,
                            solid_capstyle="butt")


def disc(ax, p, r, fc, ec=INK, lw=0.45, z=8.0):
    ax.add_patch(Circle(p, r, fc=fc, ec=ec, lw=lw, zorder=z))


def towns(ax, s, fill=None, edge=None, radius=None, skip=(), lw=None):
    """Open circles for the towns (or as given), the start's ink disc and the goal's rose ring."""
    r0, lw = md.town_radius(s), 0.5 * md._f(s) if lw is None else lw
    for t in TOWNS:
        if t in skip or t in (S0, GOAL):
            continue
        disc(ax, xy(t), (radius or {}).get(t, r0), (fill or {}).get(t, "white"), (edge or {}).get(t, INK), lw=lw)


def start_goal(ax, s, start_fc=INK, goal_fc=None, rs=None):
    disc(ax, xy(S0), rs or md.hub_radius(s) * 0.62, start_fc, start_fc, z=9)
    r = md.town_radius(s)
    if goal_fc:
        disc(ax, xy(GOAL), r * 1.35, goal_fc, goal_fc, z=9)
    else:
        disc(ax, xy(GOAL), r, "white", INK, lw=0.5 * md._f(s), z=9)
        ax.add_patch(Circle(xy(GOAL), md.goal_ring(s) * 1.05, fc="none", ec=ROSE, lw=0.7 * md._f(s), zorder=9))


def light(ax, s, x, sigma=1.15, alpha=0.5, core=1.35, res=0.18, roads=None):
    """The propagator's computed light on every road of a walk to the goal: along a road u -> v, at fraction f from u,
    the weight of the walks from that point, t0^(1 - f) X_v (q64: t0^d, d the roads to the goal). Drawn as a soft
    glow (a raster: at each point the largest such weight nearby times a Gaussian of the distance, sigma printed mm)
    under a core line whose tone is the weight (log scale)."""
    roads = LIT if roads is None else roads
    pts, val = [], []
    for u, v in roads:
        p = md._dense(road(u, v), 0.06)
        f = md._arclen(p) / md._arclen(p)[-1]
        pts.append(p)
        val.append(T0 ** (1 - f) * x[v])
    P, Vv = np.concatenate(pts), np.concatenate(val)
    hi = max(x.values())
    lo = min(v for v in Vv)
    assert Vv.max() <= hi + 1e-12 and lo > 0
    if IS_Q64:
        assert abs(Vv.max() - x[GOAL]) < 1e-9 and abs(lo - x[S0]) < 1e-9
    sg = sigma / s
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()
    gx, gy = np.arange(x0, x1, res), np.arange(y0, y1, res)
    X_, Y_ = np.meshgrid(gx, gy)
    I = np.zeros_like(X_)
    for (px, py), v in zip(P[::3], Vv[::3]):
        m = (np.abs(X_ - px) < 3.5 * sg) & (np.abs(Y_ - py) < 3.5 * sg)
        I[m] = np.maximum(I[m], v / hi * np.exp(-((X_[m] - px) ** 2 + (Y_[m] - py) ** 2) / (2 * sg * sg)))
    rgb = np.array([int(ROSE[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    img = np.concatenate([np.broadcast_to(rgb, I.shape + (3,)), (alpha * I)[..., None]], -1)
    ax.imshow(img, extent=(x0, x1, y1, y0), origin="upper", zorder=3.6, interpolation="bilinear")
    span = math.log(hi / lo) if hi > lo else 1.0
    tone = lambda v: cmap_exact(0.38 + 0.62 * math.log(max(v, lo) / lo) / span)  # noqa: E731
    for p, v in zip(pts, val):
        seg = np.stack([p[:-1], p[1:]], 1)
        vm = (v[:-1] + v[1:]) / 2
        ax.add_collection(LineCollection(seg, colors=[tone(a) for a in vm], linewidths=core * md._f(s),
                                         zorder=4.8, capstyle="round"))
    return tone


# ---------------------------------------------------------------------------------------------------- SET drawing
def card(ax, x, y, side, attrs, edge=INK, lw=0.55, z=5, sym_lw=None):
    house.set_card(ax, x, y, side, attrs, edge=edge, lw=lw, z=z, sym_lw=sym_lw or 0.075 * side)


def phase_dial(ax, cx, cy, r, ph, col=ROSE, style="hand", lw=0.9, ring=INK, ring_lw=0.5, ticks=12, z=5.0):
    """A phase on a clock face (canvas mm, y down): 0 at the top, a third of a turn per step clockwise; ticks at the
    twelve hours (the three positions long). style 'hand': a hand of length |ph| r; 'wedge': the turn from 0 to the
    phase filled (phase 0 leaves the face empty)."""
    t_ = np.linspace(0, 2 * np.pi, 160)
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=ring, lw=ring_lw, zorder=z))
    for k in range(ticks):
        a = 2 * math.pi * k / ticks
        major = k % (ticks // 3) == 0
        ln = 0.26 * r if major else 0.12 * r
        ux, uy = math.sin(a), -math.cos(a)
        ax.plot([cx + (r - ln) * ux, cx + r * ux], [cy + (r - ln) * uy, cy + r * uy], color=ring,
                lw=ring_lw * (1.0 if major else 0.7), zorder=z + 0.1, solid_capstyle="butt")
    th = math.atan2(z_im(ph), z_re(ph))                     # complex phase, counter-clockwise from 3 o'clock
    turn = th % (2 * math.pi)                             # drawn clockwise from 12 (house.dial's convention)
    if style == "wedge":                                  # the turn from 0 to the phase, lightly filled
        if turn > 1e-6 and abs(turn - 2 * math.pi) > 1e-6:
            # canvas y is down, so data angles run clockwise on the page: 12 o'clock is -90 degrees
            ax.add_patch(Wedge((cx, cy), r * 0.96 * abs(ph), -90, -90 + math.degrees(turn), fc=col, ec="none",
                               zorder=z + 0.05, alpha=0.32))
    ln = abs(ph) * r * 0.82                               # the hand, always: phase 0 points to 12 (the value 1)
    ax.plot([cx, cx + ln * math.sin(turn)], [cy, cy - ln * math.cos(turn)], color=col, lw=lw, zorder=z + 0.3,
            solid_capstyle="round")
    ax.add_patch(Circle((cx, cy), 0.11 * r + 0.12, fc=col, ec="none", zorder=z + 0.4))


def z_re(z):
    return complex(z).real


def z_im(z):
    return complex(z).imag
