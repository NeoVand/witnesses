"""A benchmark question drawn as a road map: test question 64 of the 49-vertex test set, 49 towns and 48 one-way roads,
a start with eight branches of six roads, one of which reaches the goal. Every road drawn is an edge of the question.

data/map_layout.json holds everything the drawing needs (towns, road polylines with their rank along the branch,
bridges, water, woods, ridge, names, label anchors), so code in any language can redraw it; data/map_question.json
holds the question.

    import mapdraw as md; L = md.load(); ax = md.sheet(fig, x, y, w); md.base(ax, L); md.roads(ax, L)
"""
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
LAYOUT = DATA / "map_layout.json"
QUESTION = DATA / "map_question.json"
INDEX = 64                     # the test question (lookahead 6, 49 towns)

MW, MH = 84.0, 76.0            # the sheet, mm; drawn at full size (s = 1) the map is MW mm wide

# ---------------------------------------------------------------------------------------------------- geometry


def _arclen(p):
    return np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]


def _dense(p, step):
    p = np.asarray(p, float)
    return np.concatenate([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step) + 1)) for a, b in zip(p[:-1], p[1:])])


def load():
    """The map layout (data/map_layout.json)."""
    return json.loads(LAYOUT.read_text())


# ---------------------------------------------------------------------------------------------------- drawing
_PAL = json.loads((DATA / "style" / "palette.json").read_text())
ROSE, COB, INK, GREY = _PAL["rose"], _PAL["cobalt"], _PAL["ink"], _PAL["blue_grey"]
WATER, SHORE, WATER_INK = "#E6ECF8", "#C3CEE5", "#7C90BC"   # water, its shoreline, its names
WOOD, RIDGE_C = "#F2F1EE", "#ECEAE5"
ROAD = "#BAC1CD"
SMALL, LABEL = 6.0, 7.0                              # the paper's text sizes (house.py)
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


R_TOWN = 0.78                                        # town radius, mm, at full size


def town_radius(s):
    """Town radius in map mm for the map at scale s."""
    return R_TOWN * (0.6 + 0.4 * min(s, 1.0)) / s


R_HUB = 1.6                                         # the start's disc, mm at full size, its number inside


def hub_radius(s):
    return R_HUB * (0.6 + 0.4 * min(s, 1.0)) / s if s >= 0.8 else town_radius(s) * 1.3


def goal_ring(s):
    return town_radius(s) + 0.55 * _f(s) / s


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


# ---------------------------------------------------------------------------------------------------- Theorem 1
