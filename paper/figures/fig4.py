"""Fig. 4, the two laws, in four quadrants: the size law on the left, the depth law on the right, each law above its
concrete instance. Every mark is read from a data file or computed here from the question's roads.

a  Size law across layers: four hand-set layers (data/fig4_extra.json "collapse"), each tested on a 0.5-nat grid of
   beta * Delta; the two grid values bracketing each flip, rose where every question is right, cobalt where any is
   wrong; x = log of the competitors at the margin over the read-out threshold. Same beta * Delta range as c.
c  Size law on SET: exponential Newton on SET with 6 attributes (729 cards), read with threshold 1/2
   (data/fig4_extra.json "phase"): the boundary is the smallest beta * Delta that answers all 1,000 deals without a set
   at each size (bisection), joined by a monotone cubic in log n (extended linearly past 8 and 364 cards for the wash
   only). Three temperature schedules on the same deals and 200 deals with exactly one set: beta = (2/3) ln 2P (the
   rule; P pairs), beta held at the 8-card rule, beta held at the 32-card rule. A marker is rose when every deal is
   right and cobalt when any is wrong (every error is a deal without a set called a set).
b  Depth law, exact: Theorem 1's reach, lookahead <= 2^L (rose, below the staircase), and the region where no
   L-layer pair network separates a reachable goal from an unreachable one on paths, lookahead >= 2^L + 1 (cobalt,
   above it): 2-WL gives a goal 2^L steps beyond v and one 2^L + 1 steps behind v the same colour after L rounds
   (lookahead counts from the current vertex, one more than from v). No integer lookahead lies between the two, so one
   edge is drawn. Dots: trained L-layer pair models (data/search/results; open: every lookahead in the test set
   answered). Three trained points lie above 2^L, in the region impossible on paths: L = 2 at lookahead 5 (both seeds,
   49 vertices) and L = 4 at 23 (one seed, 49 vertices). The bound is a worst case on paths and the benchmark's graphs
   are not paths; one note with a leader to each group says so ("beats the worst case").
d  Where four layers stop: test question 1 of the benchmark's 250-vertex set (data/q250.json, data/map250.json;
   lookahead 31). Every town on the answer road is the current town of a question whose lookahead is its number of
   roads to the goal; four layers (Theorem 1's construction) answer it when that is at most 2^4 = 16 (rose) and not
   beyond (cobalt). The other branches' roads are drawn faint, without their towns; labels are placed by search so
   that each keeps 1.4 mm from the town it names and 1.25 mm from every other mark.

Layout: rows share top and bottom edges (a with b, c with d), columns share left and right edges (a with c, b's spines
with d's map frame); main() prints the measured offsets and every label's clearance.

    cd paper/figures && uv run --no-project --with numpy --with scipy --with matplotlib==3.10.7 python -I fig4.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common_fig4 import *  # noqa: E402,F401,F403
from common_fig4 import house, md  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

H = 125.0
SCHED = (("rule", "o", r"$\beta = \frac{2}{3}\ln 2P$"), ("fixed32", "s", r"$\beta$ held at 32 cards"),
         ("fixed8", "^", r"$\beta$ held at 8 cards"))
L4 = 4
FAINT, FAINT_T = "#D3D8E1", "#B3BCCB"                  # the other branches' roads and towns, set back


YTOP = 12.6                                                # beta * Delta range shared by a and c


def collapse(fig, x, y, w, h):
    """The size law across four hand-set layers (data/fig4_extra.json "collapse")."""
    rows = load("fig4_extra")["collapse"]
    ax = axes_mm(fig, x, y, w, h)
    lo, hi = 0.0, YTOP
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=ROSE_W, lw=0, zorder=0)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=COB_W, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, zorder=2)
    for r in rows:
        mk = LAYER[r["layer"]][0]
        for g in r["grid"]:
            col = ROSE if g["right"] else COB
            ax.plot([r["x"]], [g["y"]], ls="None", marker=mk, ms=MS[mk] * 1.2, mfc=col, mec="white", mew=0.3,
                    zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks(range(0, 13, 2))
    ax.set_xlabel("log number of competitors at the margin")
    ax.set_ylabel(r"inverse temperature $\times$ margin, $\beta\Delta$")
    clean(ax)
    ax.text(11.0, 10.2, "$y = x$", ha="left", va="top", fontsize=SMALL, color=INK)
    ax.text(0.6, 11.6, "correct", ha="left", va="center", fontsize=LETTER, fontweight="bold", color=ROSE)
    ax.text(12.2, 1.0, "wrong", ha="right", va="center", fontsize=LETTER, fontweight="bold", color=COB)
    items = [(hkey(INK, ls="None", marker=LAYER[k][0], ms=MS[LAYER[k][0]] * 1.2), LAYER[k][1])
             for k in ("set4", "set6", "or", "prop")]
    legend(ax, items, loc="upper left", bbox_to_anchor=(0.0, 0.86), borderaxespad=0.3)
    return ax


def phase(fig, x, y, w, h):
    rows = load("fig4_extra")["phase"]["rows"]
    n = np.array([r["n"] for r in rows], float)
    ys = np.array([r["y_star"] for r in rows])
    ax = axes_mm(fig, x, y, w, h)
    lo_x, hi_x, top = 7.0, 420.0, YTOP
    f = PchipInterpolator(np.log(n), ys)
    xs = np.linspace(np.log(lo_x), np.log(hi_x), 400)
    bnd = f(np.clip(xs, np.log(n[0]), np.log(n[-1])))
    ln = np.log(n)
    s0, s1 = (ys[1] - ys[0]) / (ln[1] - ln[0]), (ys[-1] - ys[-2]) / (ln[-1] - ln[-2])
    bnd = np.where(xs < ln[0], ys[0] + s0 * (xs - ln[0]), bnd)      # linear in log n past the ends (wash only)
    bnd = np.where(xs > ln[-1], ys[-1] + s1 * (xs - ln[-1]), bnd)
    X = np.exp(xs)
    ax.fill_between(X, bnd, top, color=ROSE_W, lw=0, zorder=0)
    ax.fill_between(X, 0, bnd, color=COB_W, lw=0, zorder=0)
    ax.plot(X, bnd, color=INK, lw=0.8, zorder=2)
    for key, mk, _ in SCHED:
        yy = np.array([r[key]["y"] for r in rows])
        ax.plot(n, yy, color=INK, lw=0.6, zorder=3)
        for nn, v, r in zip(n, yy, rows):
            col = COB if r[key]["wrong"] > 0 else ROSE
            ax.plot([nn], [v], ls="None", marker=mk, ms=MS[mk] * 1.35, mfc=col, mec="white", mew=0.3, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(0, top)
    ticks = [8, 16, 32, 64, 128, 256]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(range(0, 13, 2))
    ax.set_xlabel("cards on the table (SET with 6 attributes)")
    ax.set_ylabel(r"inverse temperature $\times$ margin, $\beta\Delta$")
    clean(ax)
    ax.text(8.6, 7.95, "reads correctly", ha="left", va="center", fontsize=LETTER, fontweight="bold", color=ROSE)
    ax.text(390, 1.3, "fails", ha="right", va="center", fontsize=LETTER, fontweight="bold", color=COB)

    def first_wrong(key):
        return next(r for r in rows if r[key]["wrong"] > 0)
    r8, r32 = first_wrong("fixed8"), first_wrong("fixed32")
    note = dict(fontsize=SMALL, color=INK, ha="center", va="bottom", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.4, shrinkA=0.8, shrinkB=2.2))
    ax.annotate(f"fails at\n{r8['n']} cards", (r8["n"], r8["fixed8"]["y"]), xytext=(r8["n"] * 0.42, 1.5),
                **dict(note, va="center"))
    ax.annotate(f"fails at\n{r32['n']} cards", (r32["n"], r32["fixed32"]["y"]), xytext=(r32["n"] * 0.92, 5.3),
                **dict(note, va="center"))
    last = rows[-1]
    ax.annotate(f"reads at {last['n']} cards", (last["n"], last["rule"]["y"]), xytext=(last["n"] * 0.52, 12.05),
                **dict(note, va="center", ha="right"))
    items = [(hkey(INK, lw=0.8), r"boundary: smallest exact $\beta\Delta$")] + [
        (hkey(INK, lw=0.6, marker=mk, ms=MS[mk] * 1.2, mfc="white", mew=0.6), lab) for _, mk, lab in SCHED]
    lg = ax.legend([h for h, _ in items], [t for _, t in items], loc="upper left", frameon=False, fontsize=SMALL,
                   handlelength=1.5, handletextpad=0.5, borderaxespad=0.15, borderpad=0.1, labelspacing=0.28)
    for t in lg.get_texts():
        t.set_color(INK)
    return ax


def staircase(fig, x, y, w, h):
    runs = depth_runs()
    ax = axes_mm(fig, x, y, w, h)
    top, x0, x1 = 128 * 1.08, 1.0, 6.5
    for L in range(1, 7):                                  # answered up to 2^L; impossible on paths from 2^L + 1:
        a, b = L, min(L + 1, x1)                           # no integer lookahead lies between, so one edge serves both
        ax.fill_between([a, b], 0.8, 2.0 ** L, color=ROSE_W, lw=0, zorder=0)
        ax.fill_between([a, b], 2.0 ** L, top, color=COB_W, lw=0, zorder=0)
    sx, sy = [], []
    for L in range(1, 7):
        sx += [L, min(L + 1, x1)]
        sy += [2.0 ** L, 2.0 ** L]
    ax.plot(sx, sy, color=ROSE, lw=1.1, zorder=3, solid_joinstyle="miter")
    for V, col, dx in ((49, ROSE_L, -0.15), (100, ROSE_D, 0.15)):
        rs = [r for r in runs if r["V"] == V and r["L"] <= 6]
        for L in sorted({r["L"] for r in rs}):
            here = [r for r in rs if r["L"] == L]
            for j, r in enumerate(here):
                xx = L + 0.5 + dx + house.jitter(j, len(here), 0.1)
                ax.plot([xx], [r["reach"]], ls="None", marker="o", ms=2.1, mfc="white" if r["capped"] else col,
                        mec=col, mew=0.55, zorder=5)
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.8, top)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(x0, x1)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("layers $L$")
    ax.set_ylabel("lookahead")
    clean(ax)
    ax.text(1.2, 60, "impossible on paths", ha="left", va="center", fontsize=LETTER, color=COB)
    ax.text(1.12, 1.3, "answered", ha="left", va="center", fontsize=SMALL, color=ROSE)
    beat = {}                                              # trained points above 2^L: one note, a leader per group
    for V, dx in ((49, -0.15), (100, 0.15)):
        rs = [r for r in runs if r["V"] == V and r["L"] <= 6]
        for L in sorted({r["L"] for r in rs}):
            here = [r for r in rs if r["L"] == L]
            for j, r in enumerate(here):
                if r["reach"] > 2 ** L and not r["capped"]:
                    beat.setdefault((V, L, r["reach"]), []).append(L + 0.5 + dx + house.jitter(j, len(here), 0.1))
    assert sorted(k[1:] for k in beat) == [(2, 5), (4, 23)], beat
    note = ax.text(2.95, 13.5, "beats the worst case", ha="center", va="center", fontsize=SMALL, color=SOFT)
    fig.canvas.draw()
    bb = note.get_window_extent(fig.canvas.get_renderer())
    px = fig.dpi / 25.4
    inv = ax.transData.inverted()
    for (V, L, y), xs in beat.items():
        tx = float(np.mean(xs))
        if y < 13.5:                                       # below the note: leave from its lower edge
            start = inv.transform((bb.x0 + 0.25 * bb.width, bb.y0 - 0.5 * px))
        else:                                              # above and to the right: leave from its right end
            start = inv.transform((bb.x1 + 0.5 * px, bb.y0 + 0.75 * bb.height))
        ax.annotate("", (tx, y), xytext=start,
                    arrowprops=dict(arrowstyle="-", color=SOFT, lw=0.4, shrinkA=0, shrinkB=2.0))
    items = [(hkey(ROSE_D, ls="None", marker="o", ms=2.1), "trained, 100 vertices"),
             (hkey(ROSE_L, ls="None", marker="o", ms=2.1), "49 vertices"),
             (hkey(GREY, ls="None", marker="o", ms=2.1, mfc="white", mew=0.55), "answered every tested lookahead")]
    lg = ax.legend([h for h, _ in items], [t for _, t in items], loc="lower right", frameon=False, fontsize=SMALL,
                   handlelength=0.6, handletextpad=0.5, borderaxespad=0.1, borderpad=0.1, labelspacing=0.28)
    for t in lg.get_texts():                               # marker-only handles: a short handle keeps the box narrow
        t.set_color(INK)
    return ax


def question_map(fig, x, bottom, w, aspect):
    M = load("map250")
    route = M["route"]
    goal = M["goal"]
    d = {t: len(route) - 1 - i for i, t in enumerate(route)}          # roads to the goal along the answer road
    reach = 2 ** L4
    T = {int(k): np.asarray(v, float) for k, v in M["towns"].items()}
    pts = np.array([T[t] for t in route])
    pad_x = 7.0
    x0, x1 = pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x
    yc = (pts[:, 1].min() + pts[:, 1].max()) / 2
    hh = (x1 - x0) * aspect / 2
    crop = (x0, yc - hh, x1, yc + hh)
    ax = md.sheet(fig, x, bottom - w * aspect, w, crop=crop)
    s = md.scale(ax)
    md.base(ax, M, s)
    hl = {}
    for u, v in zip(route[:-1], route[1:]):
        hl[(u, v)] = ROSE if d[u] <= reach else COB
    md.roads(ax, M, hl, default=FAINT, s=s, chevrons=False)
    r = md.town_radius(s)
    for t in route:
        p = T[t]
        if t == M["start"]:
            ax.add_patch(Circle(p, r * 1.45, fc=INK, ec="none", zorder=9))
            continue
        if t == goal:
            ax.add_patch(Circle(p, r * 1.15, fc=ROSE, ec="none", zorder=9))
            ax.add_patch(Circle(p, md.goal_ring(s) * 1.25, fc="none", ec=ROSE, lw=0.6, zorder=9))
            continue
        col = ROSE if d[t] <= reach else COB
        ax.add_patch(Circle(p, r * 1.15, fc=col, ec="white", lw=0.35, zorder=9))
    kw = dict(fontsize=SMALL, color=INK, zorder=11)
    fig.canvas.draw()
    M = marks_px(fig, ax)
    taken = []
    first = next(t for t in route if d[t] == reach + 1)
    pf = T[first]
    lead = ax.annotate("begins to fail", pf, xytext=(pf[0] + 8.0, pf[1] + 9.5), ha="left", va="center",
                       arrowprops=dict(arrowstyle="-", color=INK, lw=0.4, shrinkA=0.6, shrinkB=2.6), **kw)
    fig.canvas.draw()                                      # an annotation's box is placed only when drawn
    taken.append(lead.get_window_extent(fig.canvas.get_renderer()))
    sc = md.scale(ax)

    def put(*a, **k):                                      # 1.25 mm from every mark if there is room, else 0.8
        try:
            return place_label(*a, touch=1.25, **k)
        except RuntimeError:
            return place_label(*a, touch=0.8, **k)
    put(ax, T[route[0]], "start", kw, M, taken, anchor_r=r * 1.45 * sc, radii=(2.4, 3.4, 4.6, 6.0, 7.5, 9.0, 10.5, 12.0))
    put(ax, T[goal], "goal", kw, M, taken, anchor_r=md.goal_ring(s) * 1.25 * sc, lead_from=3.0)
    left = [T[t] for t in route if reach - 3 <= d[t] <= reach - 1]       # the rose loop's two sides
    right = [T[t] for t in route if 6 <= d[t] <= 8]
    seed = (np.mean([q[0] for q in left + right]), np.mean([q[1] for q in left + right]))
    place_best(ax, seed, f"last {reach} roads", dict(kw, color=ROSE), M, taken)
    rounded_frame(ax, crop)
    return ax


def main():
    fig = figure(H)
    XA0, XA1 = 14.0, 88.0                                  # left column (a, c): shared spines
    XL, XR = 104.0, 177.0                                  # right column (b, d): b's spines are d's frame
    HQ = 47.0                                              # every quadrant's height
    YB1, GAP = 4.2, 3.6                                    # row-1 baseline; gap from row 1's x labels to row 2's letters
    T1 = YB1 + 2.8
    heading_base(fig, 1.0, YB1, "a", "Size law across layers")
    ax_a = collapse(fig, XA0, T1, XA1 - XA0, HQ)
    heading_base(fig, XL - 10.5, YB1, "b", "Depth law")
    ax_b = staircase(fig, XL, T1, XR - XL, HQ)
    for ax in (ax_a, ax_b):
        ylabel_at(ax, 8.5)
        xlabel_at(ax, 4.4)
    fig.canvas.draw()
    lab_bottom = max(bbox_mm(fig, ax.xaxis.label)[3] for ax in (ax_a, ax_b))
    YB2 = lab_bottom + GAP + 2.0
    T2 = YB2 + 2.8
    heading_base(fig, 1.0, YB2, "c", "Size law on SET")
    ax_c = phase(fig, XA0, T2, XA1 - XA0, HQ)
    ylabel_at(ax_c, 8.5)
    xlabel_at(ax_c, 4.4)
    heading_base(fig, XL - 10.5, YB2, "d", "Where four layers stop")
    ax_d = question_map(fig, XL, T2 + HQ, XR - XL, HQ / (XR - XL))
    report(fig, ax_a, ax_b, ax_c, ax_d, lab_bottom)
    save(fig, "fig4")


def report(fig, ax_a, ax_b, ax_c, ax_d, lab_bottom):
    """Shared edges in 300-dpi pixels, the gap between rows, and every label's clearance from the marks."""
    E = dict(zip("abcd", edges_px(fig, ax_a, ax_b, ax_c, ax_d)))     # (left, top, right, bottom)
    pairs = [("top a-b", E["a"][1], E["b"][1]), ("bottom a-b", E["a"][3], E["b"][3]),
             ("top c-d", E["c"][1], E["d"][1]), ("bottom c-d", E["c"][3], E["d"][3]),
             ("left a-c", E["a"][0], E["c"][0]), ("right a-c", E["a"][2], E["c"][2]),
             ("left b-d", E["b"][0], E["d"][0]), ("right b-d", E["b"][2], E["d"][2])]
    print("edges px: " + "; ".join(f"{n} {abs(u - v):.2f}" for n, u, v in pairs))
    lt = min(bbox_mm(fig, x)[1] for x in fig.texts if x.get_text() in ("c", "d"))
    print(f"gap from row 1's x labels to row 2's letters: {lt - lab_bottom:.2f} mm")
    for ax, name in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
        for t, d in clearance(fig, ax):
            flag = "" if d >= 1.2 else "   <-- under 1.2 mm"
            print(f"  {name}: {t!r} {d:.2f} mm{flag}")


if __name__ == "__main__":
    main()
