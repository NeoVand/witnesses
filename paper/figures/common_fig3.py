"""Shared pieces of Fig. 3: colours, SET cards, the characters of (Z_3)^4, the learning curves read from the result
files, and the panel heading (bold letter, plain title).

Sources: data/connected/newton_paper/newton_data.json (join, Newton, the two Transformers, the cost bench),
data/connected/expnewton/summary.json (exponential Newton's curve), data/clocks_m81.json (one trained join's scores),
data/fig3_tables.json (hand-set layers at 8 to 40 table cards).
"""
import json
import math

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon

from style import (DATA, INK, SOFT, GREY, MIST, ROSE, ROSE_L, ROSE_D, COB, COB_L, VIOLET, TINT, SMALL, LABEL, LETTER,
                   W, text)

CON = DATA / "connected"
ATTRS = ["number", "shape", "fill", "colour"]           # card attribute a has weight 3^a in a card's index
CARD_COL = [ROSE, COB, VIOLET]                          # SET's three colours, drawn in the palette's three hues

# series colours
C_EXP, C_NEWTON, C_JOIN, C_TBIG, C_TSMALL, C_SOFTMAX = ROSE, COB, ROSE_D, GREY, SOFT, GREY


def head(fig, x, y, letter, title):
    """Panel heading: bold lowercase letter, then a plain title."""
    text(fig, x, y, letter, LETTER, fontweight="bold")
    text(fig, x + 4.6, y + 0.35, title, LABEL)


# ------------------------------------------------------------------------------------------------ data
def newton_data():
    return json.loads((CON / "newton_paper" / "newton_data.json").read_text())


def curves():
    """Learning curves on new deals (%), every seed, for the five layers of the SET comparison. Exponential Newton:
    expnewton/summary.json curve_a (picked rate per cell); Newton: newton_data fixed_curve rerun a; the join and both
    Transformers: newton_data comparison fig5a."""
    nd = newton_data()
    out = []
    ex = json.loads((CON / "expnewton" / "summary.json").read_text())["curve_a"]["cells"]
    out.append(dict(key="exp", name="exponential Newton", col=C_EXP, ls="-", marker="o",
                    N=[int(k) for k in ex], seeds=[[100 * v for v in ex[k]["test"]] for k in ex]))
    fc = [c for c in nd["fixed_curve"]["cells"] if c["rerun"] == "a"]
    out.append(dict(key="newton", name="Newton", col=C_NEWTON, ls="-", marker="o",
                    N=[c["N"] for c in fc], seeds=[c["test"] for c in fc]))
    cmp_ = nd["comparison"]["fig5a"]
    for key, name, col, ls in (("c1", "join", C_JOIN, "-"), ("sa_big", "Transformer, 535k", C_TBIG, "-"),
                               ("sa4", "Transformer, 66k", C_TSMALL, (0, (2.2, 1.4)))):
        cs = cmp_[key]["cells"]
        out.append(dict(key=key, name=name, col=col, ls=ls, marker="o", N=[c["N"] for c in cs],
                        seeds=[c["test"] for c in cs], params=cmp_[key]["params"]))
    for c in out:
        c["mean"] = [float(np.mean(s)) for s in c["seeds"]]
    return out


def cost_rows():
    """The cost bench (A10G, one layer, 16,384 tokens per batch): {method: [(tokens per sequence, ms, MiB, status)]}."""
    rows = {}
    for r in newton_data()["cost"]["rows"]:
        if r["method"] == "join" and r["tokens"] != 16380 and r["tokens"] != 16384:
            continue                                        # the reduced-batch rows are not the same budget
        rows.setdefault(r["method"], []).append((r["n"], r.get("ms_median"), (r.get("peak_bytes") or 0) / 2 ** 20,
                                                 r["status"]))
    return rows


def tables():
    return json.loads((DATA / "fig3_tables.json").read_text())["rows"]


def join_clocks():
    """One trained join (clocks_m81.json): per attribute the mean centred score of the pairs whose sum is 0, 1, 2 mod 3,
    and the full score of every pair (j, k) for the query card 0."""
    ck = json.loads((DATA / "clocks_m81.json").read_text())
    return np.array(ck["by_attribute"]), np.array(ck["pair_scores"]), ck


# ------------------------------------------------------------------------------------------------ cards
CARDS = np.array([[(c // 3 ** a) % 3 for a in range(4)] for c in range(81)])   # card index -> (number, shape, fill, colour)


def card_index(x):
    return int(sum((int(v) % 3) * 3 ** a for a, v in enumerate(x)))


def shape_path(kind, cx, cy, w, h):
    """Upright symbol on a y-down canvas: 0 oval, 1 diamond, 2 squiggle."""
    if kind == 0:
        return FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={w / 2 * 0.98}")
    if kind == 1:
        return Polygon([(cx - w / 2, cy), (cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2)], closed=True)
    t = np.linspace(-1, 1, 61)
    centre = 0.26 * w * np.sin(np.pi * t)
    half = 0.30 * w * np.sqrt(np.clip(1 - t ** 2, 0, 1)) ** 0.6 + 0.04 * w
    ys = cy + t * h / 2
    return Polygon(np.r_[np.c_[cx + centre - half, ys], np.c_[cx + centre + half, ys][::-1]], closed=True)


def symbol(ax, kind, fill, col, cx, cy, w, h, lw=0.45, z=4):
    p = shape_path(kind, cx, cy, w, h)
    p.set(fc=col if fill == 0 else "white", ec=col, lw=lw, zorder=z, joinstyle="round")
    if fill == 1:
        p.set(hatch="////////")
        p.set_linewidth(lw)
    ax.add_patch(p)
    return p


# ------------------------------------------------------------------------------------------------ characters
def hour(s, x):
    """The hour <s, x> mod 3 of character s at card x."""
    return int(sum(a * b for a, b in zip(s, x)) % 3)


def char_image(s):
    """9 x 9 image of character s over the 81 cards: rows (number, shape), columns (fill, colour); value = hour."""
    img = np.zeros((9, 9), int)
    for x in np.ndindex(3, 3, 3, 3):
        img[3 * x[0] + x[1], 3 * x[2] + x[3]] = hour(s, x)
    return img


def representative(s):
    """True for the member of the conjugate pair {s, -s} whose last non-zero coordinate is 1 (one of each pair: the
    forty characters Newton reads; which member is a convention, the real parts are equal)."""
    nz = [v for v in s if v]
    return bool(nz) and nz[-1] == 1


def fmt_k(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


# ------------------------------------------------------------------------------------------------ cards, both orientations
def _stadium(w, h, n=24):
    """A pill of width w and height h (w >= h) centred at the origin, as a polygon."""
    r = h / 2
    a = np.linspace(-np.pi / 2, np.pi / 2, n)
    right = np.c_[w / 2 - r + r * np.cos(a), r * np.sin(a)]
    left = np.c_[-(w / 2 - r) - r * np.cos(a), -r * np.sin(a)]
    return np.r_[right, left]


def symbol_poly(kind, cx, cy, w, h, horizontal=False):
    """Vertices of a SET symbol of bounding box w x h centred at (cx, cy): 0 oval, 1 diamond, 2 squiggle. The squiggle
    is drawn upright and turned a quarter turn when horizontal."""
    W_, H_ = (h, w) if horizontal else (w, h)              # draw upright in a W_ x H_ box (H_ the long side)
    if kind == 0:
        v = _stadium(H_, W_)[:, ::-1]                       # an upright pill
    elif kind == 1:
        v = np.array([(-W_ / 2, 0), (0, -H_ / 2), (W_ / 2, 0), (0, H_ / 2)])
    else:
        t = np.linspace(-1, 1, 61)
        centre = 0.26 * W_ * np.sin(np.pi * t)
        half = 0.30 * W_ * np.sqrt(np.clip(1 - t ** 2, 0, 1)) ** 0.6 + 0.04 * W_
        ys = t * H_ / 2
        v = np.r_[np.c_[centre - half, ys], np.c_[centre + half, ys][::-1]]
    if horizontal:
        v = np.c_[v[:, 1], -v[:, 0]]
    return v + np.array([cx, cy])


def symbol2(ax, kind, fill, col, cx, cy, w, h, horizontal=False, lw=0.45, z=4):
    p = Polygon(symbol_poly(kind, cx, cy, w, h, horizontal), closed=True, fc=col if fill == 0 else "white", ec=col,
                lw=lw, zorder=z, joinstyle="round")
    if fill == 1:
        p.set_hatch("//////")
    ax.add_patch(p)
    return p


def card2(ax, cx, cy, w, h, attrs, edge=INK, lw=0.5, z=3, sym_lw=0.45):
    """A SET card on a y-down canvas. Portrait (h > w): symbols lie across and stack, as on real cards; landscape:
    symbols stand side by side."""
    n, kind, fill, colour = (int(v) for v in attrs)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.16}",
                                fc="white", ec=edge, lw=lw, zorder=z))
    col = CARD_COL[colour]
    if h > w:
        sw, sh = w * 0.62, h * 0.16
        for k in range(n + 1):
            symbol2(ax, kind, fill, col, cx, cy + (k - n / 2) * sh * 1.5, sw, sh, horizontal=True, lw=sym_lw, z=z + 1)
    else:
        sw, sh = w * 0.18, h * 0.64
        for k in range(n + 1):
            symbol2(ax, kind, fill, col, cx + (k - n / 2) * sw * 1.45, cy, sw, sh, lw=sym_lw, z=z + 1)


def one_set_deal(rng, q, n):
    """A table of n cards holding exactly one pair that completes a set with the card in hand q; returns the table and
    the pair."""
    xq = CARDS[q]
    seen, pairs = set(), []
    for c in range(81):
        if c != q and c not in seen:
            m = card_index(-xq - CARDS[c])
            seen |= {c, m}
            pairs.append((c, m))
    pick = rng.choice(len(pairs), n - 1, replace=False)
    table = list(pairs[pick[0]]) + [pairs[i][rng.integers(2)] for i in pick[1:]]
    order = rng.permutation(n)
    return [table[i] for i in order], set(pairs[pick[0]])
