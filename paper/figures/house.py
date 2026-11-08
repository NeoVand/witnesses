"""House style for the figures: Cobalt and Rose on white (data/style/palette.json), three type sizes, a 180 mm page laid
out in millimetres from the top-left corner, legends, SET cards, map labels and the readers of the search runs'
results. save() writes figN.pdf (vector, fonts embedded), figN.svg (text as outlines) and figN.png (300 dpi) into
FIG_OUT (default: out/ beside this file).
"""
import json
import math
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402  (house style, fonts, layout helpers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from layout import axes_mm, blank, clean  # noqa: E402
from palette import P  # noqa: E402
import mapdraw as md  # noqa: E402  (the map of Fig. 1)

OUT = pathlib.Path(os.environ.get("FIG_OUT", HERE / "out"))
DATA = HERE / "data"
SEARCH = DATA / "search"

# ---------------------------------------------------------------------------------------------------- house style
# Cobalt and Rose
C = json.loads((DATA / "style" / "palette.json").read_text())
ROSE, ROSE_L, COB, COB_L = C["rose"], C["rose_light"], C["cobalt"], C["cobalt_light"]
INK, SOFT, GREY, MIST, VIOLET, TINT = C["ink"], C["ink_soft"], C["blue_grey"], C["blue_mist"], C["violet"], C["tint"]
ROSE_D = "#9E1F4B"                                   # the deep end of cr_exact, for a third rose tone
P.update({k: v for k, v in C.items() if isinstance(v, str) and v.startswith("#")})
SMALL, LABEL, LETTER = 6.0, 7.0, 8.0                 # ticks and notes; titles and axis labels; panel letters
DOT = (0, (0.9, 1.4))
W = 180.0
plt.rcParams.update({"font.size": SMALL, "axes.labelsize": LABEL, "xtick.labelsize": SMALL,
                     "ytick.labelsize": SMALL, "mathtext.fontset": "cm", "axes.unicode_minus": True,
                     "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
                     "svg.fonttype": "path", "pdf.fonttype": 42, "savefig.bbox": "standard",
                     "path.simplify": False, "svg.hashsalt": "two-things", "axes.labelpad": 2.0,
                     "lines.solid_capstyle": "round", "lines.dash_capstyle": "round"})
cmap_exact = LinearSegmentedColormap.from_list("cr_exact", ["#FFFFFF", "#FBE3EB", "#F1A3BC", "#E3678F", "#D6336C",
                                                            "#B52A5C"])     # cr_exact, stopped at deep rose
cmap_div = LinearSegmentedColormap.from_list("cr_div", ["#1B3F7A", "#2557A7", "#9DB8E0", "#FFFFFF", "#F1A3BC",
                                                        "#D6336C", "#7A1F3D"])
MM = 1 / 25.4


def figure(h):
    fig = plt.figure(figsize=(W * MM, h * MM))
    fig.patch.set_facecolor("white")
    fig._h = h
    return fig


def text(fig, x, y, s, size=SMALL, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("va", "top")
    return fig.text(x / W, 1 - y / fig._h, s, fontsize=size, **kw)


def canvas(fig, x, y, w, h):
    """A blank axes in millimetres, y down, equal aspect: for drawn panels."""
    ax = axes_mm(fig, x, y, w, h)
    blank(ax)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    return ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(plt.Text) if t.get_text().strip()})
    assert all(s in (SMALL, LABEL, LETTER) for s in sizes), sizes
    over = layout.overflow(fig)
    if over:
        print(f"WARNING {name}: content past the page edge (mm): {over}")
    for a, b in layout.collisions(fig):
        print(f"WARNING {name}: texts overlap: {a!r} / {b!r}")
    for a, what in layout.on_marks(fig):
        print(f"WARNING {name}: text on a {what}: {a!r}")
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})        # reproducible bytes on a rebuild
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def hkey(col, ls="-", lw=1.0, marker=None, ms=2.4, mfc=None, mew=0.8):
    """A legend handle drawn like a series: line style and width, marker and its fill."""
    return Line2D([], [], color=col, ls=ls, lw=lw, marker=marker or "None", ms=ms, mfc=mfc or col, mec=col, mew=mew)


def legend(ax, items, loc="best", ncol=1, title=None, **kw):
    """One compact legend per chart: small type, no frame, handles drawn like the series."""
    kw = dict(dict(borderaxespad=0.3), **kw)
    lg = ax.legend([h for h, _ in items], [t for _, t in items], loc=loc, ncol=ncol, frameon=False, fontsize=SMALL,
                   title=title, title_fontsize=SMALL, handlelength=1.9, handleheight=0.6, handletextpad=0.5,
                   borderpad=0.2, labelspacing=0.28, columnspacing=1.1, alignment="left", **kw)
    for t in lg.get_texts():
        t.set_color(INK)
    if lg.get_title():
        lg.get_title().set_color(SOFT)
    return lg


def legend_at(fig, x, y, items, ncol=2, **kw):
    """A legend whose top-left corner is at (x, y) mm on the page: for one above its axes, in one or two rows."""
    lg = fig.legend([h for h, _ in items], [t for _, t in items], loc="upper left",
                    bbox_to_anchor=(x / W, 1 - y / fig._h), ncol=ncol, frameon=False, fontsize=SMALL, handlelength=1.9,
                    handleheight=0.6, handletextpad=0.5, borderaxespad=0.0, borderpad=0.0, labelspacing=0.28,
                    columnspacing=1.1, **kw)
    for t in lg.get_texts():
        t.set_color(INK)
    return lg


# ---------------------------------------------------------------------------------------------------- raw results


def run(rid):
    """A search run's rows.json record (None if the run is not in data/search/results)."""
    f = SEARCH / "results" / rid / "rows.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())[0]
    r["done"] = True
    return r


SEEDS = (0, 1, 2)


def chance(V, L):
    """Chance of the right first step on the benchmark's test questions at lookahead L: one branch in p, with p drawn
    uniformly from 2 to (V - 1) // L, as their generate_eval_data draws it (their train.py, lines 444-445)."""
    top = (V - 1) // L
    return float(np.mean([1 / p for p in range(2, top + 1)])) if top >= 2 else 1.0


# ---------------------------------------------------------------------------------------------------- characters of (Z_3)^4
OMEGA = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))


def tone(col, a):
    """`col` blended with white: a = 1 is the colour itself, a = 0 white."""
    r = np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return tuple(1 - a * (1 - r))


def chi(s, a):
    return OMEGA ** (sum(x * y for x, y in zip(s, a)) % 3)


# ---------------------------------------------------------------------------------------------------- depth runs
DEPTH_RUNS = {49: [(2, "s7-pairn-L2-v49-s{}"), (3, "s5-pairn-L3-v49-s{}"), (4, "s5-pairn-L4-v49-s{}"), (5, "s5-pairn-L5-v49-s{}"),
                   (6, "s5-pairn-L6-v49-s{}"), (7, "s5-pairn-L7-v49-s{}"), (8, "s2-pairn-v49-s{}")],
              100: [(4, "s7-pairn-L4-v100-s{}"), (5, "s7-pairn-L5-v100-s{}"), (6, "s7-pairn-L6-v100-s{}"), (8, "s2-pairn-v100-s{}")]}


def reach(acc, top, bar=0.95):
    """The deepest lookahead up to which every lookahead is answered at least `bar`."""
    r = 0
    for L in range(1, top + 1):
        if acc[str(L)] < bar:
            break
        r = L
    return r


def jitter(i, n, w=0.16):
    return (i - (n - 1) / 2) * w


# ---------------------------------------------------------------------------------------------------- SET cards
CARD_HUE = ["#5B8FC7", "#55A27A", "#8C6BB8"]          # SET's three colours, apart from Cobalt and Rose
# The deal: your card and four pairs of table cards, each pair completing a set with your card (attributes: number,
# shape, fill, colour, each 0..2); chosen by a seeded search for variety, checked below.
YOURS = (1, 0, 2, 2)
TABLE = [(1, 2, 1, 0), (1, 1, 0, 1), (2, 2, 1, 1), (0, 1, 0, 0), (2, 0, 2, 1), (0, 0, 2, 0), (2, 2, 1, 2), (0, 1, 0, 2)]


def is_set(a, b, c):
    return all((x + y + z) % 3 == 0 for x, y, z in zip(a, b, c))


def shape_path(kind, cx, cy, w, h):
    if kind == 0:                                                     # oval
        return Ellipse((cx, cy), w, h)
    if kind == 1:                                                     # diamond
        return Polygon([(cx - w / 2, cy), (cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2)], closed=True)
    t = np.linspace(-1, 1, 61)                                        # squiggle
    centre = 0.26 * w * np.sin(np.pi * t)
    half = 0.30 * w * np.sqrt(np.clip(1 - t ** 2, 0, 1)) ** 0.6 + 0.02 * w
    ys = cy + t * h / 2
    return Polygon(np.r_[np.c_[cx + centre - half, ys], np.c_[cx + centre + half, ys][::-1]], closed=True)


def set_card(ax, x, y, s, attrs, edge=None, lw=0.35, z=3, sym_lw=0.45):
    """A SET card centred at (x, y), side s mm, on a canvas (y down); attrs = (number, shape, fill, colour)."""
    n, kind, fill, col = attrs
    ax.add_patch(FancyBboxPatch((x - s / 2, y - s / 2), s, s, boxstyle=f"round,pad=0,rounding_size={s * 0.08}",
                                fc="white", ec=edge or MIST, lw=lw, zorder=z))
    hue = CARD_HUE[col]
    w, h = s * 0.24, s * 0.62
    for k in range(n + 1):
        p = shape_path(kind, x + (k - n / 2) * (w * 1.18), y, w, h)
        p.set(fc=hue if fill == 0 else "white", ec=hue, lw=sym_lw, zorder=z + 1)
        if fill == 1:
            p.set(hatch="||||||")
        ax.add_patch(p)


def sail(ax, apex, b, c, col, a0=0.10, a1=0.42, z=1):
    """A triangle with a soft fill that deepens from its apex to its base, and a thin edge."""
    poly = Polygon([apex, b, c], closed=True, fc="none", ec=col, lw=0.5, zorder=z + 1, joinstyle="round")
    ax.add_patch(poly)
    ys = np.linspace(0, 1, 200)[:, None, None]
    # on a y-down canvas the image's first row lies on the base, so the alpha falls from a1 there to a0 at the apex
    rgba = np.concatenate([np.broadcast_to(np.array(matplotlib_rgb(col)), (200, 1, 3)), a1 - (a1 - a0) * ys], -1)
    x0, x1 = min(apex[0], b[0], c[0]), max(apex[0], b[0], c[0])
    im = ax.imshow(rgba, extent=(x0, x1, max(b[1], c[1]), apex[1]), origin="lower", aspect="auto", zorder=z,
                   interpolation="bilinear")
    im.set_clip_path(Polygon([apex, b, c], closed=True, transform=ax.transData))


def matplotlib_rgb(h):
    return [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]


# ---------------------------------------------------------------------------------------------------- the map
# Test question 64 as a road map: mapdraw.py's layout (data/map_layout.json) and the question (data/map_question.json).
_MAP = {}


def maplayout():
    if "L" not in _MAP:
        _MAP["L"] = md.load()
        _MAP["Q"] = json.loads(md.QUESTION.read_text())["questions"][str(md.INDEX)]
    return _MAP["L"], _MAP["Q"]


class Labels:
    """Labels on a map at scale s: each goes to the spot around its anchor that touches the fewest roads, towns,
    water and earlier labels."""

    def __init__(self, ax, L, s, extra=()):
        self.ax, self.L, self.s, self.k = ax, L, s, 1.0 / s
        self.roads = np.concatenate([md._dense(r["path"], 0.15) for r in L["roads"]] + [np.asarray(e) for e in extra])
        self.towns = np.array([p for p in L["towns"].values()], float)
        W_ = L["water"]
        self.wet = [MPath(np.asarray(p)) for p in [W_["sea"], *W_["lakes"], *[r["outline"] for r in W_["rivers"]]]]
        x0, x1 = ax.get_xlim()
        y1, y0 = ax.get_ylim()
        self.lim = (x0, y0, x1, y1)
        self.boxes = []

    def score(self, b):
        k = self.k
        pad = 0.25 * k
        rp = self.roads
        sc = 3.0 * int(((rp[:, 0] > b[0] - pad) & (rp[:, 0] < b[2] + pad) & (rp[:, 1] > b[1] - pad)
                        & (rp[:, 1] < b[3] + pad)).sum())
        cx = np.clip(self.towns[:, 0], b[0], b[2])
        cy = np.clip(self.towns[:, 1], b[1], b[3])
        sc += 60.0 * int((np.hypot(self.towns[:, 0] - cx, self.towns[:, 1] - cy)
                          < md.town_radius(self.s) + 0.35 * k).sum())
        grid = np.array([(x_, y_) for x_ in np.linspace(b[0], b[2], 5) for y_ in np.linspace(b[1], b[3], 3)])
        sc += 2.0 * sum(int(m.contains_points(grid).sum()) for m in self.wet)
        p2 = 0.4 * k
        sc += 200.0 * sum(1 for o in self.boxes if o[0] - p2 < b[2] and o[2] + p2 > b[0] and o[1] - p2 < b[3]
                          and o[3] + p2 > b[1])
        x0, y0, x1, y1 = self.lim
        if b[0] < x0 + 0.3 * k or b[2] > x1 - 0.3 * k or b[1] < y0 + 0.3 * k or b[3] > y1 - 0.3 * k:
            sc += 1000
        return sc

    def add(self, anchor, txt, size=SMALL, color=INK, rad=None, prefer=None, gaps=(0.45, 0.8, 1.3, 2.0), at=None,
            angles=range(0, 360, 15), **kw):
        k = self.k
        w, h = md.measure(txt, size)
        w, h = w * k + 0.2 * k, h * k + 0.1 * k
        rad = md.town_radius(self.s) if rad is None else rad
        anchor = np.asarray(anchor, float)
        if at is not None:                                       # a fixed spot, still counted for later labels
            cx, cy = at
            best = (0, cx, cy, np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        else:
            best = None
            for a in angles:
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                for gap in gaps:
                    cx = anchor[0] + ca * (rad + gap * k + abs(ca) * w / 2)
                    cy = anchor[1] - sa * (rad + gap * k + abs(sa) * h / 2)
                    b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                    sc = self.score(b) + 1.5 * gap
                    if prefer is not None:
                        sc += 0.03 * abs(((a - prefer) + 180) % 360 - 180)
                    if best is None or sc < best[0]:
                        best = (sc, cx, cy, b)
        self.boxes.append(best[3])
        kw.setdefault("zorder", 12)
        return self.ax.text(best[1], best[2], txt, ha="center", va="center", fontsize=size, color=color,
                            linespacing=1.05, **kw)
