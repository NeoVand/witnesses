"""Fig. 2: the trained propagator on a 250-vertex search question.

a  The part of the question's map (data/fig2_map_even.json) around the goal, where the trained propagator's walks to
   the goal (run s22c-cap99-v49-s0, common_fig2.py) light the goal's branch; roads drawn white between thin ink
   edges, the map in a thin light rounded frame; a dashed lens on the spot shown in b.
b  That spot enlarged: the roads in the lens, each with a small arrow in its own direction (on the lit branch, toward
   the goal) and a rose bar whose length is its learned gate G[v, u] (the weight a walk keeps stepping back along the
   road u -> v), labelled with the value; two nearby towns with no road between them, dotted, with their gate
   (the larger of G[a, b] and G[b, a]). Values from data/fig2_gates.json (the map run's whole G).
c  The learned gates of the working channel in each of the three trained runs, as a grouped strip plot: every road
   walked back toward the start, the way the walks to the goal travel (rose, 249 per run), the same roads walked
   forward (cobalt), as many pairs of towns with no road, drawn at random from the 61,751 (grey), and the question's
   own pair (start, goal) (open ink). Data: data/fig2_gates.json (back, along, none_sample, query).
d  Runs that succeed against graph size, from each run's rows.json / sizes file: trained at that size, the share of
   runs that converge (the benchmark's Transformer s6t-v31, s1t-v41/49, s2t-v100; the Transformer over names with the
   pair state's read-out s23-sannpr-v49/125; the pair state over names s2-pairn-v49..125); for the propagator
   (s22c-cap99, trained on 49 vertices only), the share of runs answering every lookahead at least 0.99 at that size.
   Sizes a model was not run at are left out.
e  Test accuracy against lookahead at 250 vertices, 200 questions per lookahead, mean of three runs with their range as
   a band: the propagator (s22c-cap99) and looped triangle attention (s11-looptri256, each run at its best number of
   runs K). Chance (one branch in p, p as the generator draws it) dotted.
Run files are under data/search/results/.

    cd paper/figures && uv run --no-project --with numpy --with scipy --with matplotlib==3.10.7 python -I fig2.py
"""
import json
import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common_fig2 import *  # noqa: E402,F401,F403
from common_fig2 import (MAPE, GATES, RUN, Q, X, level, mix, dense, hero_map, heading, tangents,  # noqa: E402
                         place_word, route, GLOW_SIG, GLOW_AMP)
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, PathPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

L = MAPE
T = {int(t): np.asarray(p, float) for t, p in L["towns"].items()}
G = GATES["runs"][RUN]["G"]
MAP_X, MAP_Y, MAP_W = 9.0, 6.6, 95.0                         # the left gutter holds d's tick labels
CROP = (146.0, 0.0, 232.0, 92.6)                            # the goal's branch, its loop and the goal
MAP_H = MAP_W * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0])
ROW2 = MAP_Y + MAP_H + 6.4                                   # d and e: their titles' top
AX2_Y, AX2_H = ROW2 + 5.4, 28.0                              # their axes; the top above 100 % holds the legends
H = AX2_Y + AX2_H + 8.6
GLOW_HALF = dict(glow_sig=tuple(0.5 * v for v in GLOW_SIG), glow_amp=tuple(0.5 * v for v in GLOW_AMP))
LENS = np.array([186.0, 12.0])                              # the lens: two passes of the lit branch, between them no road
R_MAP = 10.5
NO_ROAD = None                                              # two towns in the lens with no road: the closest pair
MAG = (141.0, 34.5, 27.0)                                   # b's centre and radius, page mm
INK_EDGE = INK


def frame(fig, x, y, w, h, r=3.2):
    """A thin light rounded frame round the map; the corners outside it are cleared to the page."""
    ax = canvas(fig, 0, 0, 180.0, H)
    rr = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}")
    inner = rr.get_path().transformed(rr.get_patch_transform())
    outer = MPath([(x - 2, y - 2), (x + w + 2, y - 2), (x + w + 2, y + h + 2), (x - 2, y + h + 2), (x - 2, y - 2)],
                  [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY])
    verts = np.r_[outer.vertices, inner.vertices[::-1]]
    codes = np.r_[outer.codes, [MPath.MOVETO] + [MPath.LINETO] * (len(inner.vertices) - 2) + [MPath.CLOSEPOLY]]
    ax.add_patch(PathPatch(MPath(verts, codes), fc="white", ec="none", zorder=30))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec="#D3D9E4",
                                lw=0.6, zorder=31))
    return ax


def no_road_pair():
    E = {tuple(e) for e in Q["edges"]}
    ins = [t for t, p in T.items() if np.hypot(*(p - LENS)) < 0.8 * R_MAP]
    pairs = [(np.hypot(*(T[a] - T[b])), a, b) for i, a in enumerate(ins) for b in ins[i + 1:]
             if (a, b) not in E and (b, a) not in E]
    return min(pairs)[1:]


def lens_view(fig, cx, cy, R):
    """b: the roads in the lens, white between ink (lit: rose) edges, the light under the lit ones, the towns; per
    road an arrow in its direction and a bar for its learned gate; the no-road pair dotted with its gate."""
    ax = canvas(fig, cx - R - 6, cy - R - 3, min(2 * R + 12, 180.0 - (cx - R - 6)), 2 * R + 6)
    ox, oy = R + 6, R + 3
    k = R / R_MAP

    def P(p):
        p = np.asarray(p, float)
        return np.c_[ox + (p[:, 0] - LENS[0]) * k, oy + (p[:, 1] - LENS[1]) * k]
    clip = Circle((ox, oy), R, transform=ax.transData)
    wpt = 5.2                                                # road width, pt
    pieces = []
    for r in L["roads"]:
        u, v = r["edge"]
        q = P(dense(r["path"], 0.04)[0])
        inside = np.hypot(q[:, 0] - ox, q[:, 1] - oy) <= R + 0.4
        if inside.sum() < 2:
            continue
        m = level(X[v])
        q = q[inside]
        for lw_, al in [(wpt + 1 + 1.1 * j, 0.03 * m) for j in range(10)] if m > 0.05 else []:
            a_ = ax.plot(q[:, 0], q[:, 1], color=ROSE, lw=lw_, alpha=al, zorder=1, solid_capstyle="round")[0]
            a_.set_clip_path(clip)
        col = mix(INK_EDGE, ROSE, 0.3 + 0.7 * m) if m >= 0.3 else INK_EDGE
        for c_, lw_, z in ((col, wpt + 1.1, 2), ("white", wpt, 2.1)):
            a_ = ax.plot(q[:, 0], q[:, 1], color=c_, lw=lw_, zorder=z, solid_capstyle="butt")[0]
            a_.set_clip_path(clip)
        pieces.append((u, v, q, m))
    for t, p in T.items():
        pp = P(p[None])[0]
        if np.hypot(pp[0] - ox, pp[1] - oy) < R - 1.6:
            lt = level(X[t])
            ec = mix(INK, ROSE, 0.4 + 0.6 * lt) if lt >= 0.3 else INK
            ax.add_patch(Circle(pp, 2.0, fc="white", ec=ec, lw=0.8, zorder=3))
    # the pair with no road: the two closest towns in the lens that no road joins, dotted, with their gate
    a, b = NO_ROAD or no_road_pair()
    pa, pb = P(T[a][None])[0], P(T[b][None])[0]
    d = (pb - pa) / np.hypot(*(pb - pa))
    ax.plot([pa[0] + d[0] * 2.4, pb[0] - d[0] * 2.4], [pa[1] + d[1] * 2.4, pb[1] - d[1] * 2.4], color=GREY, lw=0.7,
            ls=(0, (1.0, 1.6)), zorder=2.5)
    g0 = max(G[a][b], G[b][a])
    mid = (pa + pb) / 2
    nrm = np.array([-d[1], d[0]])
    tw, th = md.measure(f"{g0:.2f}", SMALL)
    reach = abs(nrm[0]) * tw / 2 + abs(nrm[1]) * th / 2
    labs = [ax.text(*(mid + nrm * (1.6 + reach)), f"{g0:.2f}", ha="center", va="center", fontsize=SMALL, color=INK,
                    zorder=4)]
    # each road: an arrow on it, a bar beside it, the value beside the bar
    towns_pg = np.array([P(p[None])[0] for p in T.values()])
    road_pts = np.concatenate([q for *_, q, _ in pieces])
    for u, v, q, m in pieces:
        span = np.hypot(*np.diff(q, axis=0).T).sum()
        if span < 0.42 * R:
            continue
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        # the arrow: in the middle of the road's longest run between its towns inside the lens
        mid_i = int(np.searchsorted(cum, cum[-1] / 2))
        tt = q[min(mid_i + 8, len(q) - 1)] - q[max(mid_i - 8, 0)]
        tt /= np.hypot(*tt)
        nn = np.array([-tt[1], tt[0]])
        c0 = q[mid_i]
        if np.min(np.hypot(*(towns_pg - c0).T)) < 4.0:      # keep the arrow off the towns
            for f in (0.35, 0.65, 0.25, 0.75):
                j = int(np.searchsorted(cum, cum[-1] * f))
                if np.min(np.hypot(*(towns_pg - q[j]).T)) >= 4.0:
                    c0 = q[j]
                    break
        acol = ROSE if m >= 0.3 else INK
        ax.add_patch(FancyArrowPatch(c0 - tt * 1.7, c0 + tt * 1.7, arrowstyle="-|>,head_length=1.7,head_width=0.85",
                                     mutation_scale=1, lw=0.9, color=acol, zorder=3.6, shrinkA=0, shrinkB=0))
        g = G[v][u]
        # the bar: parallel to the road, on its emptier side, starting level with the arrow
        best = None
        for sg in (1, -1):
            base = c0 + nn * sg * 4.2 - tt * 0.30 * R * 0.5
            tip = base + tt * 0.30 * R * g
            box = np.array([base, tip, base + nn * sg * 3.5, tip + nn * sg * 3.5])
            lo, hi = box.min(0) - 0.6, box.max(0) + 0.6
            hit = ((road_pts[:, 0] > lo[0]) & (road_pts[:, 0] < hi[0]) & (road_pts[:, 1] > lo[1]) &
                   (road_pts[:, 1] < hi[1])).sum()
            out_ = np.hypot(*(np.r_[tip, base] .reshape(2, 2) - [ox, oy]).T).max() > R - 2.5
            sc = hit + 50 * out_
            if best is None or sc < best[0]:
                best = (sc, sg, base, tip)
        _, sg, base, tip = best
        bw = 0.8
        ax.add_patch(Polygon([base + nn * bw, tip + nn * bw, tip - nn * bw, base - nn * bw], closed=True, fc=ROSE,
                             ec="none", zorder=3.5))
        tw, th = md.measure(f"{g:.2f}", SMALL)
        reach = abs(nn[0]) * tw / 2 + abs(nn[1]) * th / 2          # the label's half extent across the bar
        lab = (base + tip) / 2 + nn * sg * (bw + 1.6 + reach)
        labs.append(ax.text(*lab, f"{g:.2f}", ha="center", va="center", fontsize=SMALL, color=INK, zorder=4))
    ax.add_patch(Circle((ox, oy), R, fc="none", ec=INK, lw=0.9, zorder=5))
    # the key, in the lens's empty top: what the arrows and the bars are
    rows_ = [("arrow", "road direction"), ("bar", "learned gate (walking back)")]
    wtxt = max(md.measure(t_, SMALL)[0] for _, t_ in rows_)
    x0 = ox - (4.0 + 1.3 + wtxt) / 2
    for j, (kind, t_) in enumerate(rows_):
        yk = oy - R + 5.4 + 3.6 * j
        if kind == "arrow":
            ax.add_patch(FancyArrowPatch((x0 + 0.3, yk), (x0 + 3.7, yk), arrowstyle="-|>,head_length=1.7,head_width=0.85",
                                         mutation_scale=1, lw=0.9, color=ROSE, shrinkA=0, shrinkB=0, zorder=4))
        else:
            ax.add_patch(Polygon([(x0, yk - 0.8), (x0 + 4.0, yk - 0.8), (x0 + 4.0, yk + 0.8), (x0, yk + 0.8)],
                                 closed=True, fc=ROSE, ec="none", zorder=4))
        labs.append(ax.text(x0 + 5.3, yk, t_, ha="left", va="center", fontsize=SMALL, color=INK, zorder=4))
    return ax, labs


def gate_values(fig, x, y, w, h):
    """c: one group per trained run; in each, a jittered strip per kind of gate (back along a road, forward along it,
    no road) and the start-goal pair; one legend for the four kinds."""
    from matplotlib.lines import Line2D
    ax = axes_mm(fig, x, y, w, h)
    clean(ax)
    rng = np.random.default_rng(3)
    kinds = [("back", ROSE, -0.27), ("along", COB, -0.09), ("none_sample", "#9AA4B8", 0.09)]
    stats = {}
    for i, (run, d) in enumerate(GATES["runs"].items()):
        for key, col, dx in kinds:
            v = np.asarray(d[key], float)
            ax.scatter(i + dx + rng.uniform(-0.065, 0.065, len(v)), v, s=2.2, c=col, lw=0, alpha=0.75, zorder=3,
                       rasterized=False)
            stats[(i, key)] = (float(v.min()), float(v.max()))
        ax.scatter([i + 0.27], [d["query"]], s=11, facecolors="white", edgecolors=INK, lw=0.7, zorder=4)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.03, 1.02)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["run 1", "run 2", "run 3"])
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylabel("learned gate")
    dot = dict(marker="o", ls="none", ms=2.6, mew=0)
    items = [(Line2D([], [], color=ROSE, mfc=ROSE, **dot), "road, against its arrow"),
             (Line2D([], [], color=COB, mfc=COB, **dot), "road, along its arrow"),
             (Line2D([], [], color="#9AA4B8", mfc="#9AA4B8", **dot), "no road"),
             (Line2D([], [], marker="o", ls="none", ms=3.4, mfc="white", mec=INK, mew=0.7), "start\u2013goal pair")]
    clear_legend(ax, items, [("center", (0.33, 0.45)), ("center", (0.4, 0.40)), ("center", (0.5, 0.36))])
    return ax, stats


def data_points(ax):
    """Display points of an axes' data (lines densified, scatter offsets, filled bands' outlines)."""
    pts = []
    for ln in ax.lines:
        xy = ax.transData.transform(ln.get_xydata())
        if len(xy) > 1:
            t = np.linspace(0, 1, 30)[:, None]
            xy = np.concatenate([a + (b - a) * t for a, b in zip(xy[:-1], xy[1:])])
        pts.append(xy)
    for c in ax.collections:
        off = c.get_offsets()
        if len(off) and len(c.get_paths()) <= 1:
            pts.append(ax.transData.transform(off))
        else:
            for pth in c.get_paths():
                pts.append(c.get_transform().transform(pth.vertices))
    return np.concatenate([p for p in pts if len(p)])


def clear_legend(ax, items, spots, room=ROOM, ncol=1):
    """The house legend at the first spot (loc, anchor) whose box keeps `room` mm from every data mark and off every
    band; the last spot if none does."""
    from matplotlib.path import Path as MP
    fig = ax.figure
    bands = [MP(c.get_transform().transform(p.vertices)) for c in ax.collections for p in c.get_paths()
             if len(p.vertices) > 20]
    P = data_points(ax)
    px = fig.dpi / 25.4
    for i, (loc, anchor) in enumerate(spots):
        lg = legend(ax, items, loc=loc, bbox_to_anchor=anchor, ncol=ncol)
        fig.canvas.draw()
        b = lg.get_window_extent(fig.canvas.get_renderer())
        dx = np.maximum(0, np.maximum(b.x0 - P[:, 0], P[:, 0] - b.x1))
        dy = np.maximum(0, np.maximum(b.y0 - P[:, 1], P[:, 1] - b.y1))
        corners = [(b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1), ((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)]
        on_band = any(bd.contains_point(c_) for bd in bands for c_ in corners)
        if np.hypot(dx, dy).min() / px >= room and not on_band or i == len(spots) - 1:
            return lg
        lg.remove()


SIZES = (31, 41, 49, 75, 100, 125, 150, 200, 250)       # every graph size any search run was trained or tested at


def success_rows():
    """d's points: (size, runs that succeed, runs) per model."""
    from common_fig2 import converge_rows, SEARCH
    d = converge_rows()
    prop = []
    for V in (49, 75, 100, 125, 250):
        ok = 0
        for sd in range(3):
            by = json.loads((SEARCH / "results" / f"s22c-cap99-v49-s{sd}" / f"sizes_s{sd}.json").read_text())[str(V)]
            acc = [v for k_, v in by["1"].items() if k_.isdigit()]
            ok += min(acc) >= 0.99
        prop.append((V, ok, 3))
    return dict(theirs=[p[:3] for p in d["theirs"]], tnames=[p[:3] for p in d["tnames"]],
                pair=[p[:3] for p in d["pair"]], prop=prop)


def panel_d(fig, x, y, w, h):
    from matplotlib.lines import Line2D
    ax = axes_mm(fig, x, y, w, h)
    clean(ax)
    rows = success_rows()
    style = [("theirs", GREY, "s", 1.1, "benchmark Transformer"), ("tnames", COB, "o", 1.1, "Transformer over names"),
             ("pair", ROSE_L, "o", 3.2, "pair state over names"), ("prop", ROSE, "D", 1.1, "propagator, trained on 49")]
    items = []
    for key, col, mk, lw, lab in style:                     # the pair state wide and pale under the propagator
        xs = [p[0] for p in rows[key]]
        ys = [100.0 * p[1] / p[2] for p in rows[key]]
        ms = 4.6 if key == "pair" else 2.8
        z = 3 + (key == "prop")
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            gap_ = any(x0 < v < x1 for v in SIZES)           # a size between them that this model never ran at
            ax.plot([x0, x1], [y0, y1], color=col, lw=0.7 if gap_ else lw, ls=DOT if gap_ else "-", zorder=z,
                    clip_on=False, solid_capstyle="butt")
        ax.plot(xs, ys, ls="none", marker=mk, ms=ms, mfc=col, mec=col, zorder=z + 0.1, clip_on=False)
        items.append((Line2D([], [], color=col, lw=lw, marker=mk, ms=ms, mfc=col, mec=col), lab))
    items.append((Line2D([], [], color=SOFT, lw=0.7, ls=DOT), "intermediate sizes not run"))
    ax.set_xscale("log")
    ax.set_xlim(29, 265)
    ax.set_xticks([31, 41, 49, 75, 100, 125, 250])
    ax.set_xticklabels(["31", "41", "49", "75", "100", "125", "250"])
    ax.minorticks_off()
    ax.set_ylim(-4, 136)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("vertices (training size; propagator: test size)", labelpad=1.5)
    ax.set_ylabel("runs (%)")
    clear_legend(ax, items, [("upper center", (0.5, 1.02))], ncol=3)
    return ax, rows


def panel_e(fig, x, y, w, h):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from common_fig2 import SEARCH, chance
    ax = axes_mm(fig, x, y, w, h)
    clean(ax)
    looks = np.arange(1, 125)
    out = {}
    for arm, col, lab in (("s11-looptri256", COB, "looped triangle attention"), ("s22c-cap99", ROSE, "propagator")):
        A = []
        for sd in range(3):
            by = json.loads((SEARCH / "results" / f"{arm}-v49-s{sd}" / f"sizes_s{sd}.json").read_text())["250"]
            means = {K: np.mean([by[K][str(l_)] for l_ in looks]) for K in by}
            K = max(means, key=lambda k_: (means[k_], -int(k_)))
            A.append([by[K][str(l_)] for l_ in looks])
        A = 100 * np.array(A)
        ax.fill_between(looks, A.min(0), A.max(0), color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(looks, A.mean(0), color=col, lw=1.1, zorder=3)
        out[lab] = (A.mean(1), A.min())
    ch = 100 * np.array([chance(250, int(l_)) for l_ in looks])
    ax.plot(looks, ch, color=GREY, lw=0.8, ls=DOT, zorder=2)
    ax.set_xlim(1, 124)
    ax.set_xticks([1, 32, 64, 96, 124])
    ax.set_ylim(-4, 136)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("lookahead", labelpad=1.5)
    ax.set_ylabel("accuracy (%)")
    items = [(Line2D([], [], color=ROSE, lw=1.1), "propagator"),
             (Line2D([], [], color=COB, lw=1.1), "looped triangle attention"),
             (Line2D([], [], color=GREY, lw=0.8, ls=DOT), "chance")]
    clear_legend(ax, items, [("upper center", (0.5, 1.02))], ncol=2)
    return ax, out


def main():
    fig = figure(H)
    heading(fig, 0.0, 0.6, "a", "Learned gates guide the walks")
    cx, cy, R = MAG
    ax, s, to_page = hero_map(fig, MAP_X, MAP_Y, MAP_W, L, crop=CROP, cased=True, words=False, step=False,
                              road_w=3.4, town_f=1.35, **GLOW_HALF)
    ring = T[GOAL] + (md.town_radius(s * 1.25) * 1.9 + 0.15 / s) * np.c_[np.cos(np.linspace(0, 6.3, 48)),
                                                                        np.sin(np.linspace(0, 6.3, 48))]
    km = MAP_W / (CROP[2] - CROP[0])                         # the lens's dashed lines, in map mm, kept clear too
    lp_ = np.array([MAP_X + (LENS[0] - CROP[0]) * km, MAP_Y + (LENS[1] - CROP[1]) * km])
    lines = [np.c_[np.linspace(a_[0], b_[0], 300), np.linspace(a_[1], b_[1], 300)]
             for a_, b_ in tangents(lp_, R_MAP * km, (cx, cy), R)]
    lines = np.concatenate(lines)
    lines = np.c_[CROP[0] + (lines[:, 0] - MAP_X) / km, CROP[1] + (lines[:, 1] - MAP_Y) / km]
    place_word(ax, L, s, GOAL, "goal", size=LABEL, prefer=0, weight="medium", gap=1.8, avoid=np.r_[ring, lines])
    # "walks (to goal)" inside the big loop, with a short leader to the lit road on its left
    k = MAP_W / (CROP[2] - CROP[0])
    at = np.array([160.5, 11.5])                             # map mm: inside the big loop, by its top-left corner
    ax.text(*at, "walks\n(to goal)", fontsize=SMALL, color=ROSE, ha="center", va="center", linespacing=1.0, zorder=10)
    rt = np.array([T[t] for t in route(L)])
    near = rt[np.argmin(np.hypot(*(rt - (at - [6.0 / k, 0])).T))]
    ax.plot([at[0] - 5.2 / k, near[0] + 1.6 / k], [at[1], near[1]], color=ROSE, lw=0.5, zorder=10)
    frame(fig, MAP_X, MAP_Y, MAP_W, MAP_H)
    # the lens on the map and its lines to b
    k = MAP_W / (CROP[2] - CROP[0])
    lp = np.array([MAP_X + (LENS[0] - CROP[0]) * k, MAP_Y + (LENS[1] - CROP[1]) * k])
    over = canvas(fig, 0, 0, 180.0, H)
    r1 = R_MAP * k
    over.add_patch(Circle(lp, r1, fc="none", ec=INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=40))
    for a, b in tangents(lp, r1, (cx, cy), R):
        over.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.5, ls=(0, (2.2, 1.6)), zorder=40)
    heading(fig, 108.0, 0.6, "b", "The lit road, up close")
    lens_view(fig, cx, cy, R)
    # c: its spines on b's circle (left and right) and on map a's bottom edge
    heading(fig, 108.0, 65.2, "c", "Learned gate values")
    gx, gw = cx - R, 2 * R
    gy = 72.5
    gh = MAP_Y + MAP_H - gy
    sax, q = gate_values(fig, gx, gy, gw, gh)
    # d and e: d under map a (its spines on the map's left and right edges), e under c (on c's spines)
    heading(fig, MAP_X - 9.0, ROW2, "d", "Runs that learn search")
    heading(fig, 108.0, ROW2, "e", "Lookahead at 250 vertices")
    dax, drows = panel_d(fig, MAP_X, AX2_Y, MAP_W, AX2_H)
    eax, eout = panel_e(fig, gx, AX2_Y, gw, AX2_H)
    save(fig, "fig2")
    print("lens gates:", {f"{u}->{v}": G[v][u] for u, v in Q["edges"] if np.hypot(*(T[u] - LENS)) < R_MAP * 1.4},
          "no road", no_road_pair())
    for k_, v in sorted(q.items()):
        print("  run", k_[0] + 1, k_[1], "min %.2e max %.2e" % v)
    print("d:", drows)
    print("e:", {k_: (np.round(v[0], 1).tolist(), round(float(v[1]), 1)) for k_, v in eout.items()})


if __name__ == "__main__":
    main()
