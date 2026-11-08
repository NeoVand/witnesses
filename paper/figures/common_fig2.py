"""Drawing kit and data for Fig. 2.

What is drawn, and from where (every mark is read at build time):
- the question: test question 1 of the benchmark's 250-vertex test set at lookahead 31 (data/q250.json): 250 towns,
  249 one-way roads, a start with eight branches of 31 roads, one of which ends at the goal;
- the map: data/fig2_map_even.json (the question's towns laid out as an even, meandering fill of a planar map),
  drawn with mapdraw.py (roads of one width, white casing, chevrons, bridges), woods with small tree glyphs, italic
  water names;
- the light: the trained propagator s22c-cap99-v49-s0 (one layer, trained on 49-vertex graphs) on this question,
  data/prop250.json: X = S (I - G)^{-1} of its working channel, read at the start's row; X_v is the weight of the walks
  from town v to the goal. A road u -> v carries the walks from u that go on through v, so it is lit by X_v (times its
  gate, about 0.97 on every road). Brightness is log10 X on a fixed scale from 10^LO to the largest X, drawn as rose
  light along each road and blurred: a soft glow only where the flow is;
- the gates: data/fig2_gates.json: the working channel's learned gates in all three seeds;
- the strip: per-lookahead test accuracy at 250 vertices, data/search/results/s11-looptri256-v49-s* (looped triangle
  attention, each run at its best number of runs K) and s22c-cap99-v49-s* (the propagator);
- the convergence chart: rows.json of the runs that trained at each size (pair state over names s2-pairn, the
  Transformer over names with the pair state's read-out s23-sannpr, the benchmark's own Transformer s1t, s2t, s6t).
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *  # noqa: E402,F401,F403
from style import house, md  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb, to_hex  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle, FancyArrowPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
SEARCH = DATA / "search"
Q = json.loads((DATA / "q250.json").read_text())
PROP = json.loads((DATA / "prop250.json").read_text())
GATES = json.loads((DATA / "fig2_gates.json").read_text())
MAPE = json.loads((DATA / "fig2_map_even.json").read_text())
RUN = PROP["map_run"]
R0 = PROP["runs"][RUN]
WORK = R0["working_channel"]
X = np.asarray(R0["channels"][WORK]["X_row"], float)       # X[v]: walks from v to the goal (index = town name)
START, GOAL, ANS = Q["start"], Q["goal"], Q["answer"]
KIDS = R0["kids"]
assert R0["said"] == ANS and Q["path"] == [START]
LO = -3.6                                                    # the light's scale: 10^LO (dark) to max X (full)
HI = float(np.log10(X.max()))
GAMMA = 1.5
WOOD_TREE = "#C9C4BA"
md.WOOD = "#EEECE8"                                          # woods a shade deeper than mapdraw's
HEAD_W = "medium"
ROOM = 1.3                                                   # mm a label keeps from any mark (house rule: 1.2)


def mix(a, b, f):
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return to_hex(a + (b - a) * float(np.clip(f, 0, 1)))


def level(x):
    """Brightness of walk weight x on the figure's fixed log scale."""
    return float(np.clip((np.log10(max(x, 1e-30)) - LO) / (HI - LO), 0, 1) ** GAMMA)


def heading(fig, x, y, letter, s):
    """Panel letter and title on one baseline (y is the top of the letter)."""
    text(fig, x, y + 2.25, letter, LETTER, fontweight="bold", va="baseline")
    return text(fig, x + 4.4, y + 2.25, s, LABEL, fontweight=HEAD_W, va="baseline")


def route(L):
    return [int(t) for t in L["route"]]


def dense(p, step):
    p = np.asarray(p, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    n = max(2, int(math.ceil(cum[-1] / step)) + 1)
    t = np.linspace(0, cum[-1], n)
    return np.c_[np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])], t / max(cum[-1], 1e-9)


def road_levels(L):
    """Per road u -> v: its brightness at both ends; both are X_v's (the walks from u that use this road)."""
    out = {}
    for r in L["roads"]:
        u, v = r["edge"]
        out[(u, v)] = (level(X[v]), level(X[v]))
    return out


# ---------------------------------------------------------------------------------------------------- the light
GLOW_SIG, GLOW_AMP = (0.32, 1.35, 4.2), (0.85, 0.45, 0.27)    # core, halo and wash: widths (print mm), strengths


def glow(ax, L, crop, s, sig=GLOW_SIG, amp=GLOW_AMP, z=3.6, res_mm=0.11, only=None):
    """The walks as light: each road rasterised with its brightness (max where roads meet), blurred at three widths
    (core, halo, wash; print mm), composited as rose over the page."""
    from scipy import ndimage
    x0, y0, x1, y1 = crop
    res = res_mm / s                                        # map mm per pixel
    nx, ny = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
    R = np.zeros((ny, nx))
    for r in L["roads"]:
        u, v = r["edge"]
        if only is not None and (u, v) not in only:
            continue
        p, f = dense(r["path"], res * 0.5)
        I = np.full(len(p), level(X[v]))
        j = ((p[:, 0] - x0) / res).astype(int)
        i = ((p[:, 1] - y0) / res).astype(int)
        ok = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        np.maximum.at(R, (i[ok], j[ok]), I[ok])
    a = np.zeros_like(R)
    for sg, am in zip(sig, amp):
        spx = sg / res_mm
        B = ndimage.gaussian_filter(R, spx) * math.sqrt(2 * math.pi) * spx
        a = 1 - (1 - a) * (1 - np.clip(am * B, 0, 1))
    img = np.zeros((ny, nx, 4))
    img[..., :3] = to_rgb(ROSE)
    img[..., 3] = np.clip(a, 0, 1)
    ax.imshow(img, extent=(x0, x0 + nx * res, y0 + ny * res, y0), origin="upper", zorder=z, interpolation="bilinear")
    return R


# ---------------------------------------------------------------------------------------------------- map pieces
def trees(ax, L, s, per=22.0, size=1.85, z=1.6, col=WOOD_TREE):
    """Small flat conifers in each wood (the map's wood symbol), well inside it and off every road."""
    roads = np.concatenate([dense(r["path"], 0.4)[0] for r in L["roads"]])
    for poly in L["woods"]:
        P = np.asarray(poly, float)
        path = MPath(P)
        xa, ya = P.min(0), P.max(0)
        g = 0.5 / s
        xs, ys = np.meshgrid(np.arange(xa[0], ya[0], g), np.arange(xa[1], ya[1], g))
        pts = np.c_[xs.ravel(), ys.ravel()]
        pts = pts[path.contains_points(pts)]
        if not len(pts):
            continue
        from scipy.spatial import cKDTree
        d_edge = cKDTree(np.r_[P, dense(P, 0.3)[0]]).query(pts)[0]
        d_road = cKDTree(roads).query(pts)[0]
        ok = (d_edge > 1.5 / s) & (d_road > 2.0 / s)
        pts, d_edge = pts[ok], d_edge[ok]
        area = abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])) / 2 * s * s
        n = int(np.clip(round(area / per), 1, 7))
        pick = []
        order = np.argsort(-d_edge)
        for k in order:
            if all(np.hypot(*(pts[k] - pts[q])) > 3.2 / s for q in pick):
                pick.append(k)
            if len(pick) >= n:
                break
        for k in pick:
            conifer(ax, pts[k], size / s, col, z)


def conifer(ax, p, h, col, z):
    cx, cy = p
    w = 0.62 * h
    tri1 = [(cx, cy - 0.55 * h), (cx - 0.38 * w, cy - 0.05 * h), (cx + 0.38 * w, cy - 0.05 * h)]
    tri2 = [(cx, cy - 0.30 * h), (cx - 0.5 * w, cy + 0.28 * h), (cx + 0.5 * w, cy + 0.28 * h)]
    for t in (tri2, tri1):
        ax.add_patch(Polygon(t, closed=True, fc=col, ec="none", zorder=z))
    ax.add_patch(Rectangle((cx - 0.06 * w, cy + 0.26 * h), 0.12 * w, 0.2 * h, fc=col, ec="none", zorder=z))


def place_word(ax, L, s, town, word, size=SMALL, gap=1.0, prefer=None, weight="normal", col=INK, z=10,
               avoid=(), at=None, linespacing=1.0):
    """A word beside a town (or the point `at`), at the side whose box touches the fewest roads and towns, inside
    the map's frame."""
    T = {int(t): np.asarray(p, float) for t, p in L["towns"].items()}
    roads = np.concatenate([dense(r["path"], 0.25)[0] for r in L["roads"]])
    W_ = L["water"]
    shores = [dense(p, 0.3)[0] for p in ([W_["sea"]] if W_["sea"] else []) + list(W_["lakes"])]
    shores += [dense(r["outline"], 0.3)[0] for r in W_["rivers"]]
    pts = np.concatenate([roads, np.array(list(T.values()))] + shores +
                         ([np.asarray(avoid, float)] if len(avoid) else []))
    w, h = md.measure(word, size)
    w, h = w / s, h / s
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    anchor = np.asarray(at, float) if at is not None else T[town]
    best = None
    for a in range(0, 360, 10):
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        for gp in (gap, gap * 1.6, gap * 2.4, gap * 3.4, gap * 4.6):
            cx = anchor[0] + ca * (gp / s + abs(ca) * w / 2 + 1.0 / s)
            cy = anchor[1] - sa * (gp / s + abs(sa) * h / 2 + 1.0 / s)
            pad = ROOM / s                                         # the house's 1.2 mm from any mark
            box = (cx - w / 2 - pad, cy - h / 2 - pad, cx + w / 2 + pad, cy + h / 2 + pad)
            if box[0] < x0 or box[2] > x1 or box[1] < y0 or box[3] > y1:
                continue
            n = int(((pts[:, 0] > box[0]) & (pts[:, 0] < box[2]) & (pts[:, 1] > box[1]) & (pts[:, 1] < box[3])).sum())
            sc = 4 * n + 0.35 * gp + (0.004 * abs(((a - prefer) + 180) % 360 - 180) if prefer is not None else 0)
            if best is None or sc < best[0]:
                best = (sc, cx, cy)
    t = ax.text(best[1], best[2], word, ha="center", va="center", fontsize=size, color=col, zorder=z,
                fontweight=weight, linespacing=linespacing)
    t.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
    t._fig2_score = best[0]
    return t


def goal_mark(ax, p, s, z=9.0, filled=False):
    r = md.town_radius(s)
    if filled:
        ax.add_patch(Circle(p, r * 1.9, fc=ROSE, ec="white", lw=0.9, zorder=z))
    else:
        ax.add_patch(Circle(p, r * 1.05, fc="white", ec=ROSE, lw=0.9, zorder=z))
        ax.add_patch(Circle(p, r * 2.0, fc="none", ec=ROSE, lw=1.0, zorder=z))


def cased_roads(ax, L, s, z=4.0, w=2.7, o=0.36, lit_from=0.30, clip=None, ink=INK, chevrons=True):
    """Roads drawn as a white road between two thin ink edges (lit roads' edges in rose)."""
    lv = road_levels(L)
    arts = []
    for r in L["roads"]:
        e = tuple(r["edge"])
        m = sum(lv[e]) / 2
        col = mix(ink, ROSE, 0.3 + 0.7 * m) if m >= lit_from else ink
        p = np.asarray(r["path"], float)
        arts += ax.plot(p[:, 0], p[:, 1], color=col, lw=w + 2 * o, zorder=z + (0.2 if m >= lit_from else 0),
                        solid_capstyle="round", solid_joinstyle="round")
    for r in L["roads"]:
        p = np.asarray(r["path"], float)
        arts += ax.plot(p[:, 0], p[:, 1], color="white", lw=w, zorder=z + 0.3, solid_capstyle="round",
                        solid_joinstyle="round")
        if chevrons:
            c, t = md.chevron_spot(L, r)
            md.chevron(ax, c, t, 0.62 / s, mix(md.ROAD, INK, 0.05), z + 0.4)
    if clip is not None:
        for a in arts:
            a.set_clip_path(clip)
    return arts


# ---------------------------------------------------------------------------------------------------- the hero map
def hero_map(fig, x, y, w, L, crop=None, words=True, key_at=None, names=False, tree=True, cased=False,
             trace_ring=None, avoid_page=None, step=True, road_w=2.7, town_f=1.25, glow_sig=GLOW_SIG,
             glow_amp=GLOW_AMP):
    """The question on its map with the trained layer's walks as light. Returns (ax, s, to_page)."""
    W_, H_ = L["size"]
    crop = crop or (0.0, 0.0, W_, H_)
    ax = md.sheet(fig, x, y, w, crop=crop)
    s = md.scale(ax)
    md.base(ax, L, s=s)
    if tree:
        trees(ax, L, s)
    k = w / (crop[2] - crop[0])
    if names:
        av = None
        if avoid_page is not None:                     # page-mm points (e.g. a lens's dashed lines) names keep off
            av = np.c_[crop[0] + (np.asarray(avoid_page)[:, 0] - x) / k, crop[1] + (np.asarray(avoid_page)[:, 1] - y) / k]
        lake_names(ax, L, s, avoid=av)
    glow(ax, L, crop, s, sig=glow_sig, amp=glow_amp)
    if cased:
        cased_roads(ax, L, s, w=road_w)
        rr = md.town_radius(s) * town_f
        for t, p in L["towns"].items():
            t = int(t)
            if t == START:
                continue
            lt = level(X[t])
            ec = mix(INK, ROSE, 0.4 + 0.6 * lt) if lt >= 0.3 else INK
            ax.add_patch(Circle(p, rr, fc="white", ec=ec, lw=0.55, zorder=8))
        ax.add_patch(Circle(L["towns"][str(START)], rr * 1.45, fc=INK, ec=INK, lw=0.5, zorder=8.5))
    else:
        hero_roads(ax, L, s)
        hero_towns(ax, L, s)
    if step:
        next_step(ax, L, s)
    goal_mark(ax, np.asarray(L["towns"][str(GOAL)], float), s * (1.25 if cased else 1.0), filled=cased)

    def to_page(p):
        return np.array([x + (p[0] - crop[0]) * k, y + (p[1] - crop[1]) * k])
    if words:
        place_word(ax, L, s, START, "start", prefer=200)
        place_word(ax, L, s, GOAL, "goal", size=LABEL, prefer=0, weight="medium", gap=1.6)
    return ax, s, to_page


# ---------------------------------------------------------------------------------------------------- magnifier


def tangents(c1, r1, c2, r2):
    """The two outer tangent segments of two circles (page mm)."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    d = c2 - c1
    D = np.hypot(*d)
    u = d / D
    out = []
    for sg in (1, -1):
        cos_t = (r1 - r2) / D
        sin_t = sg * math.sqrt(max(0.0, 1 - cos_t * cos_t))
        n = np.array([u[0] * cos_t - u[1] * sin_t, u[0] * sin_t + u[1] * cos_t])
        out.append((c1 + n * r1, c2 + n * r2))
    return out


# ---------------------------------------------------------------------------------------------------- gate scatter


# ---------------------------------------------------------------------------------------------------- strip and chart
def chance(V, L_):
    return house.chance(V, L_)


def converge_rows():
    """Share of runs that converge, by the graph size trained on."""
    def rid(name):
        f = SEARCH / "results" / name / "rows.json"
        return json.loads(f.read_text())[0] if f.exists() else None
    pair = {V: [rid(f"s2-pairn-v{V}-s{s_}") for s_ in range(3)] for V in (49, 75, 100, 125)}
    tnames = {V: [rid(f"s23-sannpr-v{V}-s{s_}") for s_ in range(3)] for V in (49, 125)}
    theirs = {31: [rid(f"s6t-v31-s{s_}") for s_ in (1, 2, 3)], 41: [rid(f"s1t-v41-s{s_}") for s_ in (1, 2, 3)],
              49: [rid(f"s1t-v49-s{s_}") for s_ in (1, 2, 3)], 100: [rid("s2t-v100-s1")]}
    out = {}
    for key, d in (("pair", pair), ("tnames", tnames), ("theirs", theirs)):
        pts = []
        for V, rs in sorted(d.items()):
            rs = [r for r in rs if r]
            pts.append((V, sum(1 for r in rs if r.get("converged_at")), len(rs),
                        [r.get("converged_at") or r.get("examples") for r in rs]))
        out[key] = pts
    return out
