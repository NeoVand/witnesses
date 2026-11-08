"""Helpers for Fig. 4: data readers, the panel heading (bold letter, plain headline), the two washes, marker shapes per
layer, and label placement with clearance checks.
"""
import json
import math
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import *  # noqa: E402,F401,F403
from style import house, md, OUT  # noqa: E402,F401
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text as Text_  # noqa: E402
from matplotlib.patches import Circle, Polygon, FancyBboxPatch, Wedge, PathPatch  # noqa: E402,F401
from matplotlib.path import Path as MPath  # noqa: E402

DATA = HERE / "data"
ROSE_W, COB_W = "#FBE6EE", "#E6EDF9"        # the washes: rose and cobalt tints, pale enough for points on top
LAYER = {"set4": ("o", "SET, 4 attributes"), "set6": ("D", "SET, 6 attributes"), "or": ("s", "OR join"),
         "prop": ("^", "propagator")}
MS = {"o": 2.9, "D": 2.5, "s": 2.6, "^": 3.1}


def load(name):
    return json.loads((DATA / f"{name}.json").read_text())


# ------------------------------------------------------------------------------------------------ depth law data
def depth_runs():
    """Trained L-layer pair models: per run the deepest lookahead answered at 95% (every lookahead up
    to it) and whether that is every lookahead in the test set (capped)."""
    out = []
    for V in (49, 100):
        top = (V - 1) // 2
        for L, pat in house.DEPTH_RUNS[V]:
            for s in house.SEEDS:
                r = house.run(pat.format(s))
                if not (r and r.get("done") and r.get("test")):
                    continue
                y = house.reach(r["test"], top)
                out.append(dict(V=V, L=L, seed=s, reach=y, capped=y == top, run=pat.format(s)))
    return out


# ------------------------------------------------------------------------------------------------ drawn instruments


# ------------------------------------------------------------------------------------------------ alignment
def ylabel_at(ax, mm):
    """Put the y label's centre `mm` millimetres left of the axes' left spine, so stacked or neighbouring charts
    share one label line whatever their tick labels."""
    W_ = ax.figure.get_size_inches()[0] * 25.4
    ax.yaxis.set_label_coords(-mm / (ax.get_position().width * W_), 0.5)


def xlabel_at(ax, mm):
    """Put the x label's top `mm` millimetres below the axes' bottom spine (shared baselines for labels)."""
    H_ = ax.figure.get_size_inches()[1] * 25.4
    ax.xaxis.set_label_coords(0.5, -mm / (ax.get_position().height * H_))
    ax.xaxis.label.set_va("top")


def rounded_frame(ax, crop, radius_mm=1.6, col=None, lw=0.5):
    """A thin, light, rounded frame on a map axes (map mm, y down) along `crop`, and every map mark clipped to it."""
    from matplotlib.patches import FancyBboxPatch
    x0, y0, x1, y1 = crop
    s = md.scale(ax)
    kw = dict(boxstyle=f"round,pad=0,rounding_size={radius_mm / s}", transform=ax.transData)
    clip = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, fc="none", ec="none", **kw)
    ax.add_patch(clip)
    for a in list(ax.patches) + list(ax.lines) + list(ax.collections):
        if a is not clip:
            a.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, fc="none", ec=col or MIST, lw=lw, zorder=20,
                                clip_on=False, **kw))


# ------------------------------------------------------------------------------------------------ titles and checks
def heading_base(fig, x, y_base, letter, title=None):
    """Panel letter and title (sentence case, at most five words) on one baseline at y_base mm from the top."""
    text(fig, x, y_base, letter, LETTER, fontweight="bold", va="baseline")
    if title:
        assert len(title.split()) <= 5 and title[0].isupper(), title
        text(fig, x + 4.6, y_base, title, LABEL, va="baseline")


def bbox_mm(fig, artist, r=None):
    """An artist's drawn box in mm from the page's top-left corner: (x0, y0, x1, y1), y down."""
    r = r or fig.canvas.get_renderer()
    b = artist.get_window_extent(r)
    W_, H_ = fig.get_size_inches() * 25.4
    px = fig.dpi / 25.4
    return b.x0 / px, H_ - b.y1 / px, b.x1 / px, H_ - b.y0 / px


def marks_px(fig, ax):
    """Every plotted line (sampled every 2 px), marker and disc of `ax` as (x, y, radius) in display pixels."""
    from matplotlib.patches import Circle as C_
    marks = []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        if not len(xy):
            continue
        rad = ln.get_markersize() / 2 * fig.dpi / 72 if ln.get_marker() not in (None, "", "None", " ") else 0.0
        rad += ln.get_linewidth() / 2 * fig.dpi / 72
        if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
            seg = np.diff(xy, axis=0)
            n = np.maximum(2, (np.hypot(*seg.T) / 2).astype(int))
            for a, d, k in zip(xy[:-1], seg, n):
                t = np.linspace(0, 1, k)[:, None]
                marks += [(q[0], q[1], rad) for q in a + t * d]
        else:
            marks += [(q[0], q[1], rad) for q in xy]
    for p in ax.patches:
        if isinstance(p, C_) and p.get_visible():
            tr = p.get_data_transform()                         # the centre is in data units (not the unit circle's)
            c = tr.transform(p.get_center())
            e = tr.transform((p.get_center()[0] + p.get_radius(), p.get_center()[1]))
            marks.append((c[0], c[1], abs(e[0] - c[0]) + p.get_linewidth() / 2 * fig.dpi / 72))
    return np.array(marks) if marks else np.zeros((0, 3))


def box_clear(fig, b, M):
    """Smallest distance in mm from a display-pixel box to the marks M."""
    if not len(M):
        return float("inf")
    dx = np.maximum(0, np.maximum(b.x0 - M[:, 0], M[:, 0] - b.x1))
    dy = np.maximum(0, np.maximum(b.y0 - M[:, 1], M[:, 1] - b.y1))
    return float(((np.hypot(dx, dy) - M[:, 2]) / (fig.dpi / 25.4)).min())


def clearance(fig, ax, extra=()):
    """For every text (and the legend) of `ax`: the smallest distance, in mm at print size, from its box to a plotted
    line, marker or disc of the same axes (a leader's own arrow is a patch and does not count)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    M = marks_px(fig, ax)
    items = [t for t in ax.texts if t.get_visible() and t.get_text().strip()] + list(extra)
    if ax.get_legend():
        items.append(ax.get_legend())
    out = []
    for t in items:
        b = Text_.get_window_extent(t, r) if hasattr(t, "get_text") else t.get_window_extent(r)
        name = t.get_text() if hasattr(t, "get_text") else "legend"
        out.append((name.replace("\n", " "), box_clear(fig, b, M)))
    return out


def place_label(ax, anchor, s, kw, M, taken, room=1.4, touch=0.8, anchor_r=0.6,
                radii=(2.4, 3.4, 4.6, 6.0, 7.5, 9.0), lead_from=4.5):
    """A label beside `anchor` (data units), at the first of 16 directions x radii (printed mm) where it keeps `room`
    mm from the mark it names (a disc of `anchor_r` mm at the anchor), `touch` mm from every other mark in M (nothing
    touches) and from the boxes in `taken`, and stays inside the axes by 0.8 mm; from `lead_from` mm out it gets a
    thin leader to its anchor."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    px = fig.dpi / 25.4
    a = ax.transData.transform(anchor)
    abox = ax.get_window_extent(r)
    dirs = []
    for k in range(32):                                   # 32 directions; the text hangs away from the anchor
        th = 2 * math.pi * (k + 0.5) / 32
        cx, cy = math.cos(th), math.sin(th)
        dirs.append((cx, cy, "left" if cx > 0.3 else "right" if cx < -0.3 else "center",
                     "bottom" if cy > 0.3 else "top" if cy < -0.3 else "center"))
    inv = ax.transData.inverted()
    for rad in radii:
        for ux, uy, ha, va in dirs:
            n = math.hypot(ux, uy)
            pos = inv.transform((a[0] + ux / n * rad * px, a[1] + uy / n * rad * px))
            t = ax.text(pos[0], pos[1], s, ha=ha, va=va, **kw)       # tested as plain text (an annotation's
            b = Text_.get_window_extent(t, r)                           # box is not placed until it is drawn)
            inside = (b.x0 > abox.x0 + 0.8 * px and b.x1 < abox.x1 - 0.8 * px and b.y0 > abox.y0 + 0.8 * px
                      and b.y1 < abox.y1 - 0.8 * px)
            free = all(b.x1 + touch * px < o.x0 or o.x1 + touch * px < b.x0 or b.y1 + touch * px < o.y0
                       or o.y1 + touch * px < b.y0 for o in taken)
            named = np.array([[a[0], a[1], anchor_r * px]])
            if inside and free and box_clear(fig, b, named) >= room and box_clear(fig, b, M) >= touch:
                taken.append(b)
                if rad >= lead_from:
                    t.remove()
                    t = ax.annotate(s, anchor, xytext=pos, ha=ha, va=va, arrowprops=dict(
                        arrowstyle="-", color=kw.get("color", INK), lw=0.4, shrinkA=0.8, shrinkB=2.4), **kw)
                return t
            t.remove()
    raise RuntimeError(f"no room for {s!r}")


def edges_px(fig, *axes, dpi=300):
    """Each axes' (left, top, right, bottom) edges in pixels of the 300-dpi PNG."""
    W_, H_ = fig.get_size_inches()
    out = []
    for ax in axes:
        p = ax.get_position()
        out.append((p.x0 * W_ * dpi, (1 - p.y1) * H_ * dpi, p.x1 * W_ * dpi, (1 - p.y0) * H_ * dpi))
    return out


def place_best(ax, seed, s, kw, M, taken, window=4.0, step=0.25):
    """A centred label at the point within `window` mm of `seed` (data units) that is farthest from every mark in M
    and from the boxes in `taken`; returns the text and its clearance in mm."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    px = fig.dpi / 25.4
    a = ax.transData.transform(seed)
    inv = ax.transData.inverted()
    best = None
    t = ax.text(*seed, s, ha="center", va="center", **kw)
    for dx in np.arange(-window, window + 1e-9, step):
        for dy in np.arange(-window, window + 1e-9, step):
            t.set_position(inv.transform((a[0] + dx * px, a[1] + dy * px)))
            b = Text_.get_window_extent(t, r)
            gap = min([box_clear(fig, b, M)] + [
                max(o.x0 - b.x1, b.x0 - o.x1, o.y0 - b.y1, b.y0 - o.y1) / px for o in taken])
            if best is None or gap > best[0]:
                best = (gap, t.get_position())
    t.set_position(best[1])
    taken.append(Text_.get_window_extent(t, r))
    return t, best[0]
