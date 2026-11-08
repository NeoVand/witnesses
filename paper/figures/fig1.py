"""Fig. 1: what a layer sums over, as three maps of the 49-town question over three drawings of an eight-card SET
deal. Columns: a lookup (single witnesses), a join (pairs), a closed form (everything at once). Every mark is
computed in common_fig1.py (hand-set layers, asserted).

a  the start's lookup over the 48 other towns: fan lines, circle area = softmax weight e^{3 A_sv} / Z (the eight
   children 0.100 each, every other town 0.005); below, the card in hand's lookup: one term per table card, 1/8 each.
b  the join for (start, goal) over middle towns k with both legs (at most 4 roads each): the three witnesses as their
   two-step walks start -> k -> goal; below, the join over the 28 pairs of table cards: the four set pairs as sails.
c  the propagator X = S (I - G)^{-1} from the goal (t0 = 0.8): light t0^d along the one route that reaches the goal,
   none on the seven other branches; below, Newton: the eight cards' phases for the character (1,1,1,1) as dials,
   flowing into the two sums p1 and p2, from which e2 = (p1^2 - p2)/2 gives all 28 pair terms.

    cd paper/figures && uv run --no-project --with numpy --with scipy --with matplotlib==3.10.7 python -I fig1.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common_fig1 import *  # noqa: F401,F403
from common_fig1 import (L, S0, GOAL, KIDS, ROUTE, D_GOAL, T0, TOWNS, SIZE, TABLE, YOURS, PAIRS, xy, qmap, towns, start_goal,  # noqa: E402
                         light, card, phase_dial, lookup_weights, join_witnesses, propagator, newton,
                         tree_glyphs, save_v, md, house, np, math, LineCollection, Circle)

from matplotlib.patheffects import withStroke  # noqa: E402

TITLE, BIG, SUB = 13.0, 13.0, 8.5
XC, CW = [1.0, 61.0, 121.0], 57.5
Y_HEAD, Y_MAP = 8.4, 19.6
MAP_H = CW * SIZE[1] / SIZE[0]
Y_SET = Y_MAP + MAP_H + 4.0
H = 120.0
SUBS = [("Sum over single towns", "(one road from the start)"),
        ("Sum over two-step walks", "(from the start to the goal)"),
        ("Sum over every walk", "(light flowing back from the goal)")]
GAP_MM = 1.3                                            # printed mm between a label and the mark it names (rule: 1.2)


def heads(fig):
    text(fig, 1.0, 0.6, "What a layer sums over", TITLE, fontweight="bold")
    yb = Y_HEAD + 2.6                                   # one baseline for each panel's letter and title
    for x, l_, (a, b) in zip(XC, "abc", SUBS):
        text(fig, x + 0.3, yb, l_, BIG, fontweight="bold", va="baseline")
        text(fig, x + 6.4, yb, a, SUB, va="baseline")
        text(fig, x + 6.4, yb + 3.9, b, SUB, color=SOFT, va="baseline")


def mark_labels(lab, s, start_angles=range(0, 360, 15), far=False):
    """'start' and 'goal' at least GAP_MM printed mm from the outer edge of their marks (disc; ring and its stroke)."""
    r_s = md.hub_radius(s) * 0.7 + 0.25 / s
    r_g = md.goal_ring(s) * 1.05 + 0.4 / s
    gaps = (GAP_MM, GAP_MM + 0.4, GAP_MM + 0.9, GAP_MM + 1.6)
    sg = gaps + ((GAP_MM + 2.4, GAP_MM + 3.2, GAP_MM + 4.0) if far else ())   # a: room between the lookup's rays
    t = lab.add(xy(S0), "start", size=LABEL, rad=r_s, prefer=180, gaps=sg, angles=start_angles)
    t.set_path_effects([withStroke(linewidth=1.6, foreground="white")])
    lab.add(xy(GOAL), "goal", size=LABEL, rad=r_g, prefer=0, gaps=gaps)


def map_a(fig):
    ax, s = qmap(fig, XC[0], Y_MAP, CW)
    w = lookup_weights()
    hub = xy(S0)
    ax.add_collection(LineCollection([[hub, xy(t)] for t in w], colors=COB, linewidths=0.45, alpha=0.85,
                                     zorder=6.5, capstyle="round"))
    rmax = 1.05 / s
    wmax = max(w.values())
    rad = {t: max(rmax * math.sqrt(x / wmax), md.town_radius(s) * 0.55) for t, x in w.items()}
    towns(ax, s, radius=rad, edge={t: INK for t in w}, lw=0.55 * md._f(s))
    start_goal(ax, s, rs=md.hub_radius(s) * 0.7)
    lab = house.Labels(ax, L, s, extra=[np.array([hub, xy(t)]) for t in w])
    mark_labels(lab, s, far=True)
    tree_glyphs(ax, s, avoid=lab.boxes)
    return w


def map_b(fig):
    ax, s = qmap(fig, XC[1], Y_MAP, CW)
    mids, a, b = join_witnesses()
    towns(ax, s, edge={k: ROSE for k in mids}, fill={k: "#FBE3EB" for k in mids})
    start_goal(ax, s, rs=md.hub_radius(s) * 0.7)
    r_s, r_g, r_k = md.hub_radius(s) * 0.7, md.goal_ring(s) * 1.1, md.town_radius(s) * 1.1
    paths = witness_walks(ax, s, mids, r_g, r_k)
    lab = house.Labels(ax, L, s, extra=paths)
    mark_labels(lab, s, start_angles=range(120, 241, 15))
    for k in mids:
        lab.add(xy(k), "$k$", size=LABEL, color=ROSE_D, rad=r_k, prefer=250, gaps=(GAP_MM, GAP_MM + 0.4, GAP_MM + 0.9))
    tree_glyphs(ax, s, avoid=lab.boxes)
    return mids


def bezier(p0, c, p1, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def witness_walks(ax, s, mids, r_g, r_k):
    """Each witness k as its two-step walk start -> k -> goal, two quadratic curves. Their bows are chosen by a small
    coordinate search that maximises the clearance between curves and from every other town, with the walks into the
    goal arriving at least 25 degrees apart; the first leg leaves the start town's centre. Asserted: no two curves
    come within 0.5 printed mm of each other or of a town they do not join."""
    S, G = xy(S0), xy(GOAL)
    T = np.array([xy(t) for t in TOWNS])
    legs = [(S0, k) for k in mids] + [(k, GOAL) for k in mids]

    def curve(e, h, n=48):
        P, Q = xy(e[0]), xy(e[1])
        d = np.hypot(*(Q - P))
        u = (Q - P) / d
        return bezier(P, (P + Q) / 2 + np.array([-u[1], u[0]]) * h * d, Q, n)

    def trimmed(e, p):
        m = np.ones(len(p), bool)
        for t in e:
            m &= np.hypot(*(p - xy(t)).T) > 2.2 / s
        return p[m]

    def score(hs):                                       # lower is better: soft penalties, each term on its own
        cs = [trimmed(e, curve(e, h)) for e, h in zip(legs, hs)]
        pen = 0.0
        for i in range(len(cs)):
            others = np.array([xy(t) for t in TOWNS if t not in legs[i]])
            c = (np.min(np.hypot(cs[i][:, None, 0] - others[None, :, 0], cs[i][:, None, 1] - others[None, :, 1]))
                 - md.town_radius(s)) * s
            pen += max(0.0, 1.1 - c) ** 2
            for j in range(i + 1, len(cs)):
                d = np.min(np.hypot(cs[i][:, None, 0] - cs[j][None, :, 0], cs[i][:, None, 1] - cs[j][None, :, 1])) * s
                pen += max(0.0, 1.3 - d) ** 2
        ang = [np.degrees(np.arctan2(*(curve(e, h)[-2] - G)[::-1])) for e, h in zip(legs[3:], hs[3:])]
        sep = min(abs((x - y + 180) % 360 - 180) for i, x in enumerate(ang) for y in ang[i + 1:])
        return pen + 0.02 * sum(h * h for h in hs) + (0 if sep >= 28 else 0.01 * (28 - sep) ** 2)

    rng = np.random.default_rng(1)
    hs, best = None, None
    for _ in range(400):                                 # random starts, then coordinate refinement of the best
        trial = list(rng.uniform(-0.7, 0.7, 6))
        sc = score(trial)
        if best is None or sc < best:
            best, hs = sc, trial
    grid = np.linspace(-0.8, 0.8, 33)
    for _ in range(4):
        for i in range(len(hs)):
            for h in grid:
                trial = hs[:i] + [h] + hs[i + 1:]
                sc = score(trial)
                if sc < best:
                    best, hs = sc, trial
    cs = [curve(e, h, 160) for e, h in zip(legs, hs)]
    tr = [trimmed(e, c) for e, c in zip(legs, cs)]
    for i in range(len(tr)):
        others = np.array([xy(t) for t in TOWNS if t not in legs[i]])
        assert np.min(np.hypot(tr[i][:, None, 0] - others[None, :, 0], tr[i][:, None, 1] - others[None, :, 1])) * s \
            - md.town_radius(s) * s > 0.5, ("a walk passes a town", legs[i])
        for j in range(i + 1, len(tr)):
            dmin = np.min(np.hypot(tr[i][:, None, 0] - tr[j][None, :, 0], tr[i][:, None, 1] - tr[j][None, :, 1]))
            assert dmin * s > 0.5, ("two walks touch", legs[i], legs[j], dmin * s)
    ang = [np.degrees(np.arctan2(*(c[-2] - G)[::-1])) for c in cs[3:]]
    assert min(abs((x - y + 180) % 360 - 180) for i, x in enumerate(ang) for y in ang[i + 1:]) >= 25, ang
    out = []
    for e, c in zip(legs, cs):
        end = r_g + 0.3 / s if e[1] == GOAL else r_k + 0.25 / s
        c = c[np.hypot(*(c - xy(e[1])).T) > end]
        ax.plot(c[:, 0], c[:, 1], color=ROSE, lw=0.85, zorder=7, solid_capstyle="butt")
        md.chevron(ax, c[-1], c[-1] - c[-6], 1.3 / s, ROSE, 7.1)
        out.append(md._dense(c, 0.3))
    print("witness walks: bows", np.round(hs, 2).tolist(), "arrival angles at the goal", np.round(ang, 0).tolist())
    return out


def map_c(fig):
    lit = {(u, v): (md.ROAD, 0.0) for u, v in zip(ROUTE[:-1], ROUTE[1:])}
    ax, s = qmap(fig, XC[2], Y_MAP, CW, hl=lit)
    x = propagator()
    tone = light(ax, s, x, sigma=1.3, alpha=0.55, core=1.4)
    towns(ax, s, fill={t: tone(x[t]) for t in ROUTE[1:-1]}, edge={t: ROSE_D for t in ROUTE[1:-1]})
    start_goal(ax, s, rs=md.hub_radius(s) * 0.7)
    lab = house.Labels(ax, L, s)
    mark_labels(lab, s)
    tree_glyphs(ax, s, avoid=lab.boxes)
    return x


# ---------------------------------------------------------------------------------------------------- SET row
QS, TS = 7.4, 5.8
GAP = (CW - 8 * TS) / 7                                 # the card row spans the map frame, edge to edge


def row_x(w):
    xs = np.arange(8) * (TS + GAP)
    return xs - xs.mean() + w / 2


def set_a(fig):
    cv = canvas(fig, XC[0], Y_SET, CW, H - Y_SET)
    xs = row_x(CW)
    qx, qy, ty = CW / 2, QS / 2 + 0.4, 30.5
    card(cv, qx, qy, QS, YOURS, lw=0.7)
    for x_, c in zip(xs, TABLE):
        card(cv, x_, ty, TS, c)
        cv.plot([qx, x_], [qy + QS / 2, ty - TS / 2], color=COB, lw=0.6, zorder=2, solid_capstyle="butt")
    cv.text(qx + QS / 2 + 1.4, qy - 1.6, "card in hand", fontsize=LABEL, color=INK, va="center")
    cv.text(qx + QS / 2 + 1.4, qy + 1.6, "(a SET card)", fontsize=LABEL, color=SOFT, va="center")
    side_label(cv, "weight 1/8\nto each card", COB, (qx, qy + QS / 2), (xs[7], ty - TS / 2))
    cv.text(CW / 2, ty + TS / 2 + 2.0, "eight table cards", fontsize=LABEL, color=INK, ha="center", va="top")


LABEL_Y = 10.4                                          # the side labels of d and e, centred at one height


def side_label(cv, txt, col, a, b, pad=GAP_MM):
    """A two-line label at the panel's right edge, centred at LABEL_Y, asserted clear of the outermost line or
    wedge edge from a to b (canvas mm) by `pad` mm."""
    w = max(md.measure(t, LABEL)[0] for t in txt.split("\n"))
    h = sum(md.measure(t, LABEL)[1] for t in txt.split("\n")) * 1.25
    x1 = CW - 0.6
    yb = LABEL_Y + h / 2
    edge = a[0] + (b[0] - a[0]) * (yb - a[1]) / (b[1] - a[1])          # the line's x at the label's bottom
    assert edge + pad < x1 - w, (txt, edge, x1 - w)
    cv.text(x1, LABEL_Y, txt, fontsize=LABEL, color=col, ha="right", va="center", linespacing=1.15)


def set_b(fig):
    cv = canvas(fig, XC[1], Y_SET, CW, H - Y_SET)
    xs = row_x(CW)
    qx, qy, ty = CW / 2, QS / 2 + 0.4, 30.5
    card(cv, qx, qy, QS, YOURS, lw=0.7)
    for x_, c in zip(xs, TABLE):
        card(cv, x_, ty, TS, c)
    apex = qx + (np.arange(4) - 1.5) * QS * 0.24        # each pair its own apex on the card's bottom edge
    for (j, k), ax_ in zip(PAIRS, apex):                # narrow wedges: base from one card's centre to the other's
        house.sail(cv, (ax_, qy + QS / 2), (xs[j] - TS * 0.12, ty - TS / 2), (xs[k] + TS * 0.12, ty - TS / 2), ROSE,
                a0=0.12, a1=0.5, z=1)
    cv.text(qx + QS / 2 + 1.4, qy - 1.6, "card in hand", fontsize=LABEL, color=INK, va="center")
    cv.text(qx + QS / 2 + 1.4, qy + 1.6, "(a SET card)", fontsize=LABEL, color=SOFT, va="center")
    side_label(cv, "sum over pairs\n28 terms, 4 sets", ROSE, (apex[-1], qy + QS / 2),
               (xs[7] + TS * 0.12, ty - TS / 2))
    cv.text(CW / 2, ty + TS / 2 + 2.0, "eight table cards (four set pairs)", fontsize=LABEL, color=INK, ha="center",
            va="top")


def set_c(fig):
    """Newton beside the cards: the eight table cards in two rows of four, each with its phase dial for the
    character (1,1,1,1) below it; each row's dials feed a short bus that reaches both sums, p1 (above) and p2
    (below), stacked on the right with their formulas beside them, and e2 between them. The block spans the band of
    panels a and b (card in hand to table cards); the caption sits on their captions' baseline."""
    nw = newton()
    cv = canvas(fig, XC[2], Y_SET, CW, H - Y_SET)
    top, bot = 0.4, 30.5 + TS / 2                       # the band of panels a and b, canvas mm
    cs, r, R = 5.2, 3.0, 3.9
    labs = (r"$p_1=\sum_j \varphi_j$", r"$p_2=\sum_j \varphi_j^{\,2}$")
    wlab = max(md.measure(t, LABEL)[0] for t in labs)
    xsum = CW - wlab - GAP_MM - R                       # the block spans the map frame: formulas end at its edge
    bus = xsum - R - 1.6
    xs = np.linspace(r, bus - 2.0 - max(cs / 2, r), 4)
    ydial = [top + cs + 0.9 + r, bot - r]
    ycard = [y - r - 0.9 - cs / 2 for y in ydial]
    ysum = [ydial[0], bot - R]
    for row in range(2):
        cv.plot([xs[0], bus], [ydial[row], ydial[row]], color=ROSE, lw=0.55, alpha=0.75, zorder=1,
                solid_capstyle="round")
        cv.plot([bus, xsum - R], [ysum[row], ysum[row]], color=ROSE, lw=0.55, alpha=0.75, zorder=1)
        md.chevron(cv, np.array([xsum - R - 0.55, ysum[row]]), (1, 0), 1.1, ROSE, 1.2)
        for i in range(4):
            j = 4 * row + i
            card(cv, xs[i], ycard[row], cs, TABLE[j])
            phase_dial(cv, xs[i], ydial[row], r, nw["phi"][j], col=ROSE, lw=0.9, ring_lw=0.55)
    cv.plot([bus, bus], [ydial[0], ydial[1]], color=ROSE, lw=0.55, alpha=0.75, zorder=1)   # both rows reach both sums
    big = max(abs(nw["p1"]), abs(nw["p2"]), 1.0)          # hands to one scale: the larger sum fills its dial
    for y, val, lab_ in zip(ysum, (nw["p1"], nw["p2"]), labs):
        phase_dial(cv, xsum, y, R, val / big, col=ROSE, lw=1.2, ring_lw=0.7)
        cv.text(xsum + R + GAP_MM, y, lab_, fontsize=LABEL, color=INK, ha="left", va="center")
    ym = (ysum[0] + ysum[1]) / 2
    cv.text(bus + 1.6, ym - 1.9, r"$e_2=\frac{1}{2}\,(p_1^{2}-p_2)$", fontsize=LABEL, color=INK, ha="left",
            va="center")
    cv.text(bus + 1.6, ym + 2.4, "all 28 pairs at once", fontsize=LABEL, color=ROSE, ha="left", va="center")
    cv.text(CW / 2, 30.5 + TS / 2 + 2.0, "eight table cards (as clock dials)", fontsize=LABEL, color=INK,
            ha="center", va="top")
    return nw


FRAME = "#CDD4E0"                                       # the thin, light map frame


def framed(fig, border):
    """With border: every map axes clipped to a rounded rectangle and outlined by a thin, light stroke."""
    if not border:
        return
    from matplotlib.patches import FancyBboxPatch
    for ax in fig.axes[-3:]:
        x0, x1 = ax.get_xlim()
        y1, y0 = ax.get_ylim()
        rr = 1.6 / md.scale(ax)
        clip = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={rr}", fc="none",
                              ec="none", transform=ax.transData)
        for a in ax.get_children():
            if hasattr(a, "set_clip_path") and a is not ax.patch and a not in ax.spines.values():
                a.set_clip_path(clip)
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={rr}",
                                    fc="none", ec=FRAME, lw=0.6, zorder=20, clip_on=False))


def main(name="fig1", border=True):
    fig = figure(H)
    heads(fig)
    map_a(fig)
    map_b(fig)
    map_c(fig)
    framed(fig, border)
    set_a(fig)
    set_b(fig)
    nw = set_c(fig)
    save_v(fig, name)
    print("Newton: p1 %s, p2 %s, e2 %s, sets counted over 81 characters: %d" % (
        np.round(nw["p1"], 3), np.round(nw["p2"], 3), np.round(nw["e2"], 3), nw["sets"]))


if __name__ == "__main__":
    main()
