"""Layout helpers and fonts: Helvetica Neue for every text (bold, italic and medium faces extracted from the macOS
system collection into a local, git-ignored cache), axes placed in millimetres from the top-left corner, and the
overflow, overlap and text-on-mark checks that save() runs on every figure.
"""
import logging
import math
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from palette import P, use  # noqa: E402
from matplotlib import font_manager  # noqa: E402

STAMP = 3876750951                               # the faces' head.modified (s since 1904): fixed, so PDFs rebuild byte for byte
FONTS = HERE / ".fonts"                         # extracted system faces; git-ignored, never committed (Apple's licence)


def _faces():
    """matplotlib reads only the first face of macOS's HelveticaNeue.ttc, so bold and italic silently fall back to
    regular; extract those faces once into .fonts/ and register them."""
    import dataclasses
    import json

    ttc = pathlib.Path("/System/Library/Fonts/HelveticaNeue.ttc")
    want = {1: "HelveticaNeue-Bold.ttf", 2: "HelveticaNeue-Italic.ttf", 10: "HelveticaNeue-Medium.ttf"}
    if not ttc.exists():
        return
    index = FONTS / "entries.json"                    # matplotlib's font properties, cached: addfont takes seconds
    if not index.exists():
        try:
            from fontTools.ttLib import TTCollection
        except ImportError:
            print("fonttools missing: bold and italic text will render regular (build with --with fonttools)")
            return
        FONTS.mkdir(exist_ok=True)
        faces = TTCollection(str(ttc), recalcTimestamp=False).fonts
        for i, f in want.items():
            faces[i]["head"].modified = STAMP
            faces[i].save(str(FONTS / f))
        entries = [dataclasses.asdict(font_manager.ttfFontProperty(font_manager.get_font(str(FONTS / f))))
                   for f in want.values()]
        index.write_text(json.dumps(entries))
    for e in json.loads(index.read_text()):
        font_manager.fontManager.ttflist.append(font_manager.FontEntry(**e))
    font_manager.fontManager._findfont_cached.cache_clear()


_faces()
logging.getLogger("fontTools").setLevel(logging.ERROR)                   # PDF subsetting chatters about timestamps
use()
P.update({"rose": "#D1477A", "rose_light": "#E9A2BC", "rose_deep": "#A8325E",
          "cobalt": "#1F4FD1", "cobalt_light": "#A3B7EE", "cobalt_deep": "#173C9E"})
SMALL, LABEL, LETTER = 5.5, 6.5, 8                     # base type sizes (house.py sets the figures' own)
plt.rcParams.update({"font.size": SMALL, "axes.labelsize": LABEL, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.0, "ytick.major.size": 2.0, "lines.linewidth": 0.9,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom",
                     "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic",
                     "mathtext.bf": "Helvetica Neue:bold", "mathtext.fallback": "stixsans"})
MM = 1 / 25.4


def axes_mm(fig, x, y, w, h, **kw):
    """An axes placed in millimetres from the figure's top-left corner."""
    W_, H_ = fig.get_size_inches() / MM
    return fig.add_axes([x / W_, 1 - (y + h) / H_, w / W_, h / H_], **kw)


def clean(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P["ink"])
    ax.tick_params(colors=P["ink"], length=2.0, width=0.5, pad=1.5)
    if not left:
        ax.set_yticks([])
    if not bottom:
        ax.set_xticks([])
    ax.set_facecolor("none")


def blank(ax):
    ax.set_axis_off()
    ax.set_facecolor("none")


def overflow(fig, tol=0.1):
    """Millimetres by which drawn content runs past each page edge (beyond `tol`); empty if it all fits."""
    fig.canvas.draw()
    b, w, h = fig.get_tightbbox(fig.canvas.get_renderer()), *fig.get_size_inches()
    mm = {"left": -b.x0, "right": b.x1 - w, "bottom": -b.y0, "top": b.y1 - h}
    return {k: round(v / MM, 1) for k, v in mm.items() if v / MM > tol}


def _drawn_texts(fig):
    """Every text the figure draws: free text, axis labels and titles, legends, and the tick labels inside the view."""
    out = list(fig.texts)
    for ax in fig.axes:
        out += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        if ax.get_legend():
            out += list(ax.get_legend().get_texts())
        if not ax.axison:
            continue
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) * 1e-6
            for tick in axis.get_major_ticks():
                if lo - span <= tick.get_loc() <= hi + span:
                    out += [t for t in (tick.label1, tick.label2) if t.get_visible()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def collisions(fig, pad=0.15):
    """Pairs of texts whose boxes overlap by more than `pad` mm on each side: no label may cross another."""
    fig.canvas.draw()
    r, px = fig.canvas.get_renderer(), pad * fig.dpi / 25.4
    boxes = [(t.get_text(), Text.get_window_extent(t, r)) for t in _drawn_texts(fig)]
    hits = []
    for i, (sa, a) in enumerate(boxes):
        for sb, b in boxes[i + 1:]:
            if (a.x0 + px < b.x1 - px and b.x0 + px < a.x1 - px and a.y0 + px < b.y1 - px
                    and b.y0 + px < a.y1 - px):
                hits.append((sa, sb))
    return hits


def touches(fig, ax, t, r, pad=0.1):
    """What `t` sits on among its axes' lines and markers ("line", "marker" or None). Line stretches hidden under an
    opaque patch drawn above them (a node's disc, say) do not count."""
    box = t.get_bbox_patch()
    if box is not None and box.get_facecolor()[3] >= 0.8:          # on its own opaque backing, nothing runs under it
        return None
    px = pad * fig.dpi / 25.4
    bb = Text.get_window_extent(t, r)                              # the words only, not an annotation's arrow
    x0, y0, x1, y1 = bb.x0 + px, bb.y0 + px, bb.x1 - px, bb.y1 - px
    opaque = [p for p in ax.patches if p.get_visible() and p.get_fill() and p.get_facecolor()[3] > 0.99]
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xy = ln.get_transform().transform(ln.get_xydata())
        if not len(xy):
            continue
        ms = (ln.get_markersize() / 2 * fig.dpi / 72) if ln.get_marker() not in (None, "", "None", " ") else 0
        if ms and np.any((xy[:, 0] > x0 - ms) & (xy[:, 0] < x1 + ms) & (xy[:, 1] > y0 - ms) & (xy[:, 1] < y1 + ms)):
            return "marker"
        if ln.get_linestyle() in ("None", "", " ") or len(xy) < 2:
            continue
        u = np.linspace(0, 1, 40)[:, None, None]
        pts = (xy[:-1][None] + u * (xy[1:] - xy[:-1])[None]).reshape(-1, 2)
        pts = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)]
        above = [p for p in opaque if p.get_zorder() > ln.get_zorder()]
        if any(not any(p.contains_point(q) for p in above) for q in pts):
            return "line"
    for c in ax.collections:                     # line collections drawn strongly enough to read as lines, not texture
        if not isinstance(c, LineCollection) or not c.get_visible():
            continue
        cols = c.get_edgecolor()
        alpha = 1.0 if c.get_alpha() is None else float(np.max(c.get_alpha()))
        for k, seg in enumerate(c.get_segments()):
            if len(seg) < 2 or (cols[k % len(cols)][3] if len(cols) else 1.0) * alpha < 0.3:
                continue
            xy = c.get_transform().transform(seg)
            u = np.linspace(0, 1, 40)[:, None, None]
            pts = (xy[:-1][None] + u * (xy[1:] - xy[:-1])[None]).reshape(-1, 2)
            if np.any((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)):
                return "line"
    for c in ax.collections:                                       # scatter markers big enough to read as points
        if not isinstance(c, PathCollection) or not c.get_visible() or not len(c.get_offsets()):
            continue
        offs = c.get_offset_transform().transform(c.get_offsets())
        s = np.broadcast_to(c.get_sizes() if len(c.get_sizes()) else [0.0], (len(offs),))
        rad = np.sqrt(s) / 2 * fig.dpi / 72
        near = np.hypot(offs[:, 0] - np.clip(offs[:, 0], x0, x1), offs[:, 1] - np.clip(offs[:, 1], y0, y1))
        if np.any((s >= 4) & (near < rad)):
            return "marker"
    return None


def on_marks(fig, pad=0.1):
    """Texts in an axes that touch one of its plotted lines or markers: text never sits on a line or a data point."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                what = touches(fig, ax, t, r, pad)
                if what:
                    hits.append((t.get_text(), what))
    return hits


