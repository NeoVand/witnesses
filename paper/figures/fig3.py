"""Fig. 3, SET: who learns, how the join decides, why it is cheap, why it holds as tables grow.

a  Learning curves on new deals (join, Newton, exponential Newton with four clocks, two Transformers;
   common_fig3.curves), and beside them accuracy with a card in hand never held in training (27 cards held in
   training): exponential Newton with six clocks, 729 channels (data/connected/expnewton/results/exp8-rule-a, picked
   rate, test cards), Newton (newton_data m27_new_cards), join and Transformers (newton_data comparison m27) at 55,296
   deals; the Transformers also at 221,184 (open).
b  One trained join (data/clocks_m81.json): for a card in hand and one table card, every third card on one axis (the
   card z = q + j + k in balanced ternary, number slowest, colour fastest, the set in the middle); rows: each
   attribute's clock factor exp(v_a(z_a) - v_a(0)) from the join's mean centred scores; last row: the join's own
   softmax weights over the 81 third cards from its full pair scores.
c  Four clocks (one per attribute: a card's value sits at one of three hours) multiply into exp(beta sum_a cos theta_a),
   a single bright card on the 9 x 9 map of the 81 cards (beta = 4.86, learned by the six-clock layer of e), and that
   product is exactly the 81 characters of (Z_3)^4 (Jacobi-Anger on Z_3), drawn in their 3^4 arrangement, mirror pairs
   in two tones of rose, the four clocks outlined.
d  Newton-type layers sum characters over the table in linear time: one layer's time and peak memory per training step
   against tokens (A10G, 16,384 tokens per batch; newton_data cost). Exponential Newton was not benchmarked.
e  Real one-set deals at 8, 16, 20 and 40 table cards (40 is the largest table that can lack a set), the completing
   pair outlined. Bars: solid = trained on 8-card tables, new cards in hand, mean of 3 seeds (ticks: seeds); hatched =
   hand-set layer with four exact clocks (data/fig3_tables.json: 800 deals per size, half with one set): exponential
   Newton at 40 cards with the rule beta (no trained run there) and the same layer with the 8-card beta fixed (the size
   law's prediction). Trained bars are the six-clock layer (exp8-rule-a). Newton only where trained.

    cd paper/figures && uv run --no-project --with numpy --with scipy --with matplotlib==3.10.7 python -I fig3.py
"""
import json
import math
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import (figure, canvas, axes_mm, clean, legend, legend_at, hkey, save, text, INK, SOFT, GREY, MIST, ROSE, ROSE_L,  # noqa: E402
                   ROSE_D, COB, COB_L, SMALL, LABEL, LETTER)
from common_fig3 import (head, curves, cost_rows, tables, join_clocks, newton_data, CON, ATTRS, CARDS, CARD_COL,  # noqa: E402
                         card_index, char_image, representative, symbol, card2, one_set_deal, fmt_k, C_EXP, C_NEWTON,
                         C_JOIN, C_TBIG, C_TSMALL, C_SOFTMAX)

plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Helvetica Neue",
                     "mathtext.it": "Helvetica Neue:italic", "mathtext.bf": "Helvetica Neue:bold",
                     "mathtext.fallback": "stixsans"})
H = 166.0
TOP, BOT = 10.0, 49.0          # row 1: top and bottom spines of a's charts and b's rows
L1, R1 = 11.0, 92.0            # left column: a's curves (left spine) to a's dot plot (right spine); c's clocks to grid
L2, R2 = 107.5, 178.0          # right column: b's rows and d's charts, spine to spine
Y2 = 57.0                      # row 2 title line
Y3 = 121.0                     # row 3 title line
BETA = 4.86                                     # exponential Newton, six clocks, seed 5 (picked rate)


# ------------------------------------------------------------------------------------------------ data
def new_card_rows():
    """Accuracy (%) with a new card in hand, per seed: (name, colour, [seeds at 55,296], [seeds at 221,184] or None)."""
    nd = newton_data()
    fit = json.loads((CON / "expnewton" / "results" / "exp8-rule-a" / "fits.jsonl").read_text().splitlines()[0])
    rate = fit["lrs"].index(0.003)
    exp8 = [100 * v for v in fit["test8"]["A"][rate]]
    newton = [r["out8"] for r in nd["m27_new_cards"]["rows"] if r["rerun"] == "a"]
    m27 = {(r["arm"], r["N"]): r["out8"] for r in nd["comparison"]["m27"]}
    return [("exponential Newton,\n6 clocks", C_EXP, exp8, None), ("Newton", C_NEWTON, newton, None),
            ("join", C_JOIN, m27[("c1", 55296)], None),
            ("Transformer 535k", C_TBIG, m27[("sa_big", 55296)], m27[("sa_big", 221184)]),
            ("Transformer 66k", C_TSMALL, m27[("sa4", 55296)], m27[("sa4", 221184)])]


def trained_tables():
    """{layer: {table cards: [seeds, %]}} for models trained on 8-card tables, new cards in hand."""
    fit = json.loads((CON / "expnewton" / "results" / "exp8-rule-a" / "fits.jsonl").read_text().splitlines()[0])
    rate = fit["lrs"].index(0.003)
    exp = {n: [100 * v for v in fit[f"test{n}"]["A"][rate]] for n in (8, 16, 20)}
    rows = [r for r in newton_data()["m27_new_cards"]["rows"] if r["rerun"] == "a"]
    newton = {8: [r["out8"] for r in rows], 16: [r["out16"] for r in rows]}
    return dict(exp=exp, newton=newton)


# ------------------------------------------------------------------------------------------------ a
def row_legend(fig, x, y, items, ncol, handlelength=1.5, columnspacing=1.0):
    """One compact legend row whose top-left corner sits at (x, y) mm."""
    lg = fig.legend([h for h, _ in items], [t for _, t in items], loc="upper left", bbox_to_anchor=(x / 180, 1 - y / H),
                    ncol=ncol, frameon=False, fontsize=SMALL, handlelength=handlelength, handleheight=0.6,
                    handletextpad=0.45, borderaxespad=0.0, borderpad=0.0, labelspacing=0.28,
                    columnspacing=columnspacing)
    for t in lg.get_texts():
        t.set_color(INK)
    return lg


def ax_legend(ax, items, loc, handlelength=1.6, ncol=1, borderaxespad=0.3, **kw):
    """One compact legend inside an axes (handles drawn like the series)."""
    lg = ax.legend([h for h, _ in items], [t for _, t in items], loc=loc, ncol=ncol, frameon=False, fontsize=SMALL,
                   handlelength=handlelength, handleheight=0.6, handletextpad=0.45, borderpad=0.15,
                   labelspacing=0.3, columnspacing=0.9, borderaxespad=borderaxespad, **kw)
    for t in lg.get_texts():
        t.set_color(INK)
    return lg


def panel_a(fig):
    head(fig, 1.0, 1.5, "a", "Who learns the rule")
    cs = curves()
    for c in cs:
        c["name"] = {"exp": "exponential Newton, 4 clocks", "sa_big": "Transformer 535k",
                     "sa4": "Transformer 66k"}.get(c["key"], c["name"])
    EDGES["a_curves"] = ax = axes_mm(fig, L1, TOP, 37.0, BOT - TOP)
    for c in cs:
        for n, ss in zip(c["N"], c["seeds"]):
            ax.scatter([n] * len(ss), ss, s=2.4, color=c["col"], lw=0, alpha=0.5, zorder=3)
        ax.plot(c["N"], c["mean"], color=c["col"], ls=c["ls"], lw=1.1, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1700, 640000)
    ticks = [2048, 8192, 32768, 131072, 524288]
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt_k(v) for v in ticks])
    ax.minorticks_off()
    ax.set_ylim(46, 102)
    ax.set_yticks([50, 75, 100])
    ax.set_xlabel("training deals", labelpad=1.5)
    ax.set_ylabel("new deals right (%)")
    clean(ax)
    EDGES["a_legend"] = row_legend(fig, 1.0, 6.2, [(hkey(c["col"], ls=c["ls"], lw=1.1), c["name"]) for c in cs],
                                   ncol=5, handlelength=1.0, columnspacing=0.6)

    # beside: a new card in hand
    rows = new_card_rows()
    EDGES["a_dots"] = ax = axes_mm(fig, R1 - 19.0, TOP, 19.0, BOT - TOP)
    for i, (name, col, s55, s221) in enumerate(rows):
        text(fig, 51.5, TOP + (BOT - TOP) * (i + 0.8) / (len(rows) + 0.4), name, SMALL, color=col if col != C_TSMALL
             else INK, va="center", linespacing=1.1)
        ax.scatter(s55, [i] * len(s55), s=6, color=col, lw=0, zorder=4)
        ax.plot([np.mean(s55)] * 2, [i - 0.3, i + 0.3], color=col, lw=1.1, zorder=5)
        if s221:
            ax.scatter(s221, [i + 0.0] * len(s221), s=7, facecolor="white", edgecolor=col, lw=0.7, zorder=4)
    ax.set_ylim(len(rows) - 0.4, -0.8)
    ax.set_xlim(44, 106)
    ax.set_xticks([50, 100])
    ax.set_xlabel("new card in hand (%)", labelpad=1.5)
    clean(ax, left=False)
    ax_legend(ax, [(hkey(INK, ls="None", marker="o", ms=2.5), "55k deals"),
                   (hkey(INK, ls="None", marker="o", ms=2.6, mfc="white", mew=0.7), "221k deals")], loc="upper left",
              handlelength=0.8, borderaxespad=0.2)


# ------------------------------------------------------------------------------------------------ b
PLACE = {0: 27, 1: 9, 2: 3, 3: 1}


def balanced(t):
    d = {}
    for a in (3, 2, 1, 0):
        r = ((t + 1) % 3) - 1
        d[a] = r
        t = (t - r) // 3
    return d


def panel_b(fig):
    head(fig, 94.5, 1.5, "b", "Four clocks, one peak")
    by_attr, pair, ck = join_clocks()
    assert ck["query"] == 0
    ts = np.arange(-40, 41)
    Z = [balanced(t) for t in ts]
    x0, w, hgt = L2, R2 - L2, 5.4
    ys = [TOP, TOP + 7.4, TOP + 14.8, TOP + 22.2]
    for i, a in enumerate(range(4)):
        ax = axes_mm(fig, x0, ys[i], w, hgt)
        EDGES.setdefault("b_rows", []).append(ax)
        v = by_attr[a]
        f = np.array([math.exp(v[z[a] % 3] - v[0]) for z in Z])
        ax.fill_between(ts, f, step="mid", color=C_JOIN, alpha=0.12, lw=0, zorder=1)
        ax.step(ts, f, where="mid", color=C_JOIN, lw=0.8, zorder=2)
        ax.set_xlim(-40.5, 40.5)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 1])
        ax.set_xticks([])
        clean(ax, bottom=False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(INK)
        cyc = 81 // (3 * PLACE[a])
        text(fig, 95.0, ys[i] + hgt / 2 - 1.15, ATTRS[a], SMALL, va="center")
        text(fig, 95.0, ys[i] + hgt / 2 + 1.15, f"×{cyc}", SMALL, color=SOFT, va="center")
    yP = BOT - hgt
    EDGES["b_product"] = ax = axes_mm(fig, x0, yP, w, hgt)
    j = card_index((1, 2, 0, 1))
    zs = [card_index(tuple(z[a] % 3 for a in range(4))) for z in Z]
    ks = [card_index(tuple(-CARDS[j][a] + CARDS[zc][a] for a in range(4))) for zc in zs]
    sc = np.array([pair[j][k] for k in ks])
    wt = np.exp(sc - sc.max())
    assert int(np.argmax(wt)) == 40
    ax.fill_between(ts, wt, step="mid", color=C_JOIN, alpha=0.12, lw=0, zorder=1)
    ax.step(ts, wt, where="mid", color=C_JOIN, lw=0.9, zorder=2)
    ax.set_xlim(-40.5, 40.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_xticks([0])
    ax.set_xticklabels(["set"])
    clean(ax)
    ax.set_xlabel("the 81 possible third cards", labelpad=1.0)
    text(fig, 95.0, yP + hgt / 2, "join's\nweights", SMALL, linespacing=1.1, va="center")
    m_, e_ = f"{np.sort(wt)[-2]:.0e}".split("e")
    text(fig, x0 + w * 0.56, yP + 0.2, rf"next largest {int(m_)}$\times10^{{{int(e_)}}}$", SMALL, color=SOFT)
    cv = canvas(fig, x0, ys[0] - 0.8, w, yP + hgt - ys[0] + 0.8)
    xs = 40.5 / 81 * w
    cv.plot([xs, xs], [0, yP + hgt - ys[0] + 0.8], color=INK, lw=0.5, ls=(0, (2.0, 1.6)), zorder=0)
    for xx in (0.3, 0.7):
        cv.text(xx * w, (ys[3] + hgt + yP) / 2 - ys[0] + 0.8, "×", fontsize=LABEL, color=INK, ha="center",
                va="center")
    cv.text(xs + 2.4, (ys[3] + hgt + yP) / 2 - ys[0] + 0.8, "=", fontsize=LABEL, color=INK, ha="center", va="center")


# ------------------------------------------------------------------------------------------------ c
def dial(ax, cx, cy, R, attr):
    t = np.linspace(0, 2 * np.pi, 241)
    ax.plot(cx + R * np.cos(t), cy + R * np.sin(t), color=INK, lw=0.6, zorder=2)
    for v in range(3):
        a = 2 * math.pi * v / 3
        ux, uy = math.sin(a), -math.cos(a)
        ax.add_patch(Circle((cx + R * ux, cy + R * uy), 0.45, fc=ROSE, ec="none", zorder=4))
        gx, gy = cx + 0.5 * R * ux, cy + 0.5 * R * uy
        if attr == 0:
            for k in range(v + 1):
                symbol(ax, 1, 0, INK, gx + (k - v / 2) * 0.75, gy, 0.55, 1.15, lw=0.25)
        elif attr == 1:
            symbol(ax, v, 0, INK, gx, gy, 0.85, 1.6, lw=0.25)
        elif attr == 2:
            symbol(ax, 1, v, INK, gx, gy, 1.0, 1.8, lw=0.3)
        else:
            symbol(ax, 1, 0, CARD_COL[v], gx, gy, 1.0, 1.8, lw=0.3)


def panel_c(fig, y0):
    head(fig, 1.0, y0, "c", "Four clocks are 81 characters")
    S0, mx, my = R1 - 44.0, 44.0, 7.0                    # the grid: right edge on the left column's edge
    cv = canvas(fig, 0, y0, 100.0, my + S0 + 1.0)
    EDGES["c_canvas"] = cv
    R = 4.6
    cxd = L1 + R                                           # the clocks' left edge on a's left spine
    ys = [my + R + k * (S0 - 2 * R) / 3 for k in range(4)]   # top clock's top and bottom clock's bottom on the grid's
    for a, y in enumerate(ys):
        dial(cv, cxd, y, R, a)
        cv.text(1.0, y, ATTRS[a], fontsize=SMALL, color=INK, ha="left", va="center")
        if a < 3:
            cv.text(cxd, (ys[a] + ys[a + 1]) / 2, "×", fontsize=SMALL, color=INK, ha="center", va="center")
    EDGES["c_clocks"] = (cxd - R, ys[0] - R, ys[-1] + R)
    # the product on the 81 cards, centred on the grid's middle
    ps = 13.0
    px, py = 26.5, my + S0 / 2 - ps / 2
    img = np.zeros((9, 9))
    for x in np.ndindex(3, 3, 3, 3):
        img[3 * x[0] + x[1], 3 * x[2] + x[3]] = math.exp(BETA * (sum(math.cos(2 * math.pi * v / 3) for v in x) - 4))
    cv.imshow(img, cmap=ListedColormap(["#FFFFFF"]) if img.max() == 0 else
              plt.matplotlib.colors.LinearSegmentedColormap.from_list("w_r", ["#FFFFFF", ROSE]), vmin=0, vmax=1,
              interpolation="nearest", extent=(px, px + ps, py + ps, py), zorder=2)
    cv.add_patch(Rectangle((px, py), ps, ps, fc="none", ec=INK, lw=0.45, zorder=3))
    cv.text(px + ps / 2, py - 1.6, "product", fontsize=SMALL, color=INK, ha="center", va="bottom")
    cv.text(px + ps / 2, py + ps + 1.4, rf"$\beta$ = {BETA:.1f}", fontsize=SMALL, color=SOFT, ha="center", va="top")
    cv.text(px + ps / 2, py + ps + 4.6, "one bright card:\nthe set", fontsize=SMALL, color=ROSE, ha="center",
            va="top", linespacing=1.1)
    cv.annotate("", xy=(px - 1.0, py + ps / 2), xytext=(cxd + R + 1.3, py + ps / 2),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=INK, lw=0.6))
    cv.text(px + ps + 3.0, py + ps / 2, "=", fontsize=LABEL, color=INK, ha="center", va="center")
    # the 81 characters
    t = S0 / 9
    m1 = ListedColormap([ROSE, "#FBE3EB", "#FBE3EB"])
    m2 = ListedColormap([ROSE_L, "#FDF1F5", "#FDF1F5"])
    for s in np.ndindex(3, 3, 3, 3):
        r, c = 3 * s[0] + s[1], 3 * s[2] + s[3]
        rep = representative(s) or not any(s)
        cv.imshow(char_image(s), cmap=m1 if rep else m2, vmin=0, vmax=2, interpolation="nearest",
                  extent=(mx + c * t, mx + (c + 1) * t, my + (r + 1) * t, my + r * t), zorder=2)
    for i in range(10):
        cv.plot([mx, mx + S0], [my + i * t] * 2, color=INK, lw=0.35, zorder=3, solid_capstyle="butt")
        cv.plot([mx + i * t] * 2, [my, my + S0], color=INK, lw=0.35, zorder=3, solid_capstyle="butt")
    for a in range(4):
        s = [0, 0, 0, 0]
        s[a] = 1
        r, c = 3 * s[0] + s[1], 3 * s[2] + s[3]
        x_, y_ = mx + c * t, my + r * t
        cv.plot([x_, x_ + t, x_ + t, x_, x_], [y_, y_, y_ + t, y_ + t, y_], color=INK, lw=1.3, zorder=5,
                solid_joinstyle="miter")
    cv.text(mx, my - 1.4, "81 characters; outlined: the four clocks", fontsize=SMALL, color=INK,
            ha="left", va="bottom")
    EDGES["c_grid"] = (mx, mx + S0, my, my + S0)

    # d: cost of one layer, its two charts spanning the mosaic's height
    head(fig, 94.5, y0, "d", "Linear cost")
    rows = cost_rows()
    ticks = [9, 64, 256, 1024, 4096]
    top, bot = y0 + my, y0 + my + S0
    hh = (bot - top - 4.0) / 2
    for k, (col_i, lab, yl) in enumerate(((1, "ms per step", (0.4, 4000)), (2, "peak MiB", (60, 60000)))):
        ax = axes_mm(fig, L2, top + k * (hh + 4.0), R2 - L2, hh)
        EDGES.setdefault("d_charts", []).append(ax)
        for m, col, name in (("join", C_JOIN, "join"), ("newton", C_NEWTON, "Newton"),
                             ("softmax", C_SOFTMAX, "attention")):
            ok = [(r[0], r[col_i]) for r in rows[m] if r[3] == "ok"]
            ax.plot([n for n, _ in ok], [v for _, v in ok], color=col, lw=1.0, marker="o", ms=2.2, zorder=4)
            for n in [r[0] for r in rows[m] if r[3] != "ok"]:
                ax.plot([n], [yl[1] * 0.5], marker="x", ms=3.4, mew=0.9, color=col, zorder=5)
        if k == 0:
            ax_legend(ax, [(hkey(C_JOIN, lw=1.0, marker="o", ms=2.2), "join"),
                        (hkey(C_NEWTON, lw=1.0, marker="o", ms=2.2), "Newton"),
                        (hkey(C_SOFTMAX, lw=1.0, marker="o", ms=2.2), "attention"),
                        (hkey(C_JOIN, ls="None", marker="x", ms=3.4, mew=0.9), "out of memory")],
                      loc="upper left", ncol=2, borderaxespad=0.25)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(["9", "64", "256", "1,024", "4,096"] if k == 1 else [])
        ax.minorticks_off()
        ax.set_xlim(7, 5200)
        ax.set_ylim(*yl)
        ax.set_yticks([1, 10, 100] if k == 0 else [100, 1000, 10000])
        ax.set_yticklabels(["1", "10", "100"] if k == 0 else ["100", "1k", "10k"])
        ax.set_ylabel(lab, labelpad=1.2)
        clean(ax)
        if k == 1:
            ax.set_xlabel("tokens per sequence", labelpad=1.5)


# ------------------------------------------------------------------------------------------------ d
SIZES = [8, 16, 20, 40]
LAYOUT = {8: (2, 4, 5.8, 7.8), 16: (4, 4, 5.8, 3.9), 20: (4, 5, 5.6, 3.9), 40: (5, 8, 5.4, 3.5)}
X0 = {8: 33.0, 16: 64.0, 20: 95.0, 40: R2 - (8 * (5.4 + 0.8) - 0.8)}


def panel_e(fig, y0):
    head(fig, 1.0, y0, "e", "Larger tables, same rule")
    rng = np.random.default_rng(3)
    q = card_index((2, 1, 1, 0))
    cv = canvas(fig, 0, y0 + 5.0, 180.0, 27.0)
    cy = 14.5
    cv.text(15.0, 2.0, "in hand", fontsize=SMALL, color=INK, ha="center", va="center")
    card2(cv, 15.0, cy, 5.8, 7.8, CARDS[q], edge=INK, lw=0.7, sym_lw=0.35)
    for n in SIZES:
        rows, cols, w, h = LAYOUT[n]
        table, pair = one_set_deal(rng, q, n)
        px, py = w + 0.8, h + 0.7
        tw, th = cols * px - 0.8, rows * py - 0.7
        x0, y_ = X0[n], cy - th / 2 + 1.0
        cv.text(x0 + tw / 2, 2.0, f"{n} cards", fontsize=SMALL, color=INK, ha="center", va="center")
        for i, c in enumerate(table):
            r, k = divmod(i, cols)
            on = c in pair
            card2(cv, x0 + k * px + w / 2, y_ + r * py + h / 2, w, h, CARDS[c], edge=ROSE if on else INK,
                  lw=0.8 if on else 0.35, sym_lw=0.3)
    # bars
    T = {r["n"]: r for r in tables()}
    tr = trained_tables()
    spec = [("exponential Newton", C_EXP, lambda n: tr["exp"].get(n), "exp_rule"),
            ("hand-set, β fixed at 8 cards", ROSE_L, lambda n: None, "exp_fixed"),
            ("Newton", C_NEWTON, lambda n: tr["newton"].get(n), "mean")]
    yb = y0 + 35.0
    for jrow, (name, col, trained, key) in enumerate(spec):
        y = yb + jrow * 3.7
        text(fig, 1.0, y - 1.2, name, SMALL)
        for n in SIZES:
            rows, cols, w, h = LAYOUT[n]
            tw = cols * (w + 0.8) - 0.8
            L = min(tw, 30.0) - 8.0
            ax = canvas(fig, X0[n], y - 1.3, L + 9.0, 2.6)
            seeds = trained(n)
            if seeds or key != "mean":
                ax.plot([0.6, L], [1.3, 1.3], color=MIST, lw=3.0, solid_capstyle="butt", zorder=1, alpha=0.4)
            if not seeds and key == "mean":
                ax.text(0.6, 1.3, "not trained at this size", fontsize=SMALL, color=SOFT, ha="left", va="center")
                continue
            if seeds:
                v = float(np.mean(seeds))
                ax.add_patch(Rectangle((0.6, 0.15), (L - 0.6) * v / 100, 2.3, fc=col, ec="none", zorder=2))
                for s_ in seeds:
                    xx = 0.6 + (L - 0.6) * s_ / 100
                    ax.plot([xx, xx], [-0.1, 2.7], color=INK, lw=0.4, zorder=3)
            else:
                v = 100 * T[n][key]
                ax.add_patch(Rectangle((0.6, 0.15), (L - 0.6) * v / 100, 2.3, fc="white", ec=col, lw=0.6,
                                       hatch="/////", zorder=2))
            ax.text(L + 1.2, 1.3, f"{v:.0f}%", fontsize=SMALL, color=INK, ha="left", va="center")
    # key: one row on the title line, right of the title
    kv = canvas(fig, 62.0, y0, 116.0, 3.4)
    yc = 1.95
    kv.add_patch(Rectangle((0.0, yc - 1.0), 3.2, 2.0, fc=C_EXP, ec="none"))
    kv.plot([2.6, 2.6], [yc - 1.25, yc + 1.25], color=INK, lw=0.4)
    kv.text(4.4, yc, "trained on 8-card tables, 6 clocks (ticks: seeds)", fontsize=SMALL, color=INK, ha="left",
            va="center")
    kv.add_patch(Rectangle((62.0, yc - 1.0), 3.2, 2.0, fc="white", ec=C_EXP, lw=0.6, hatch="/////"))
    kv.text(66.4, yc, "hand-set, 4 exact clocks", fontsize=SMALL, color=INK, ha="left", va="center")


EDGES = {}


def report_extent(fig):
    """Right end of a's legend row and of e's row labels (mm), against the edges they must stay inside."""
    r = fig.canvas.get_renderer()
    bb = EDGES["a_legend"].get_window_extent(r)
    print(f"a legend: {bb.x0 / fig.dpi * 25.4:.1f} to {bb.x1 / fig.dpi * 25.4:.1f} mm (left column ends {R1} mm)")


def report_edges(fig):
    """Pixel positions (300 dpi, from the top-left) of every edge that should be shared, and their offsets."""
    px = 300 / 25.4
    H_ = fig.get_size_inches()[1] * 300

    def box(ax):
        b = ax.get_position()
        W_ = fig.get_size_inches()[0] * 300
        return b.x0 * W_, b.x1 * W_, H_ - b.y1 * H_, H_ - b.y0 * H_          # left, right, top, bottom (px)

    def mm(v):
        return v * px

    a, ad, br, bp = box(EDGES["a_curves"]), box(EDGES["a_dots"]), [box(x) for x in EDGES["b_rows"]], box(EDGES["b_product"])
    d0, d1 = box(EDGES["d_charts"][0]), box(EDGES["d_charts"][1])
    gx0, gx1, gy0, gy1 = EDGES["c_grid"]
    cl, ct, cb = EDGES["c_clocks"]
    out = {
        "row1 top: a curves / a dots / b first row": (a[2], ad[2], br[0][2]),
        "row1 bottom: a curves / a dots / b product": (a[3], ad[3], bp[3]),
        "left column left: a curves spine / c clocks edge": (a[0], mm(cl)),
        "left column right: a dots spine / c grid edge": (ad[1], mm(gx1)),
        "right column left: b rows / d charts": (br[0][0], d0[0], d1[0]),
        "right column right: b rows / d charts": (br[0][1], d0[1], d1[1]),
        "row2 top: c grid / c top clock / d time chart": (mm(Y2 + gy0), mm(Y2 + ct), d0[2]),
        "row2 bottom: c grid / c bottom clock / d memory chart": (mm(Y2 + gy1), mm(Y2 + cb), d1[3]),
    }
    for k, v in out.items():
        print(f"{k}: {', '.join(f'{x:.1f}' for x in v)} px  (spread {max(v) - min(v):.2f} px)")


def main():
    fig = figure(H)
    panel_a(fig)
    panel_b(fig)
    panel_c(fig, Y2)
    panel_e(fig, Y3)
    report_edges(fig)
    report_extent(fig)
    save(fig, "fig3")


if __name__ == "__main__":
    main()
