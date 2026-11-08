"""Fig. 2b and 2d's data: a real 250-vertex test question, its map, and the trained one-layer propagator on it.

Computed on a CPU from saved files only; nothing is trained and nothing is downloaded.

1. question  Test question 1 of the benchmark's own 250-vertex test set at lookahead 31: Saparov et al.'s
             generate_eval_data (their train.py at the commit every run pins, 2c3f32b), reseeded with 12345
             before the lookahead exactly as l2s_eval.test_sets does, so this is the question the s22c runs were scored
             on. Rule, fixed before the model was run on it: the first question at lookahead 31 that names all 250
             vertices (question 0 names 218). Lookahead 31 lies past the longest in 49-vertex training (24); the question
             has eight branches of 31 roads: q64's shape (Fig. 1) at five times the size.
2. map       map_engine.py (this folder), unchanged except a larger sheet (its 112 mm cap raised to 220, so the sheet takes the size its own formula asks for) and marks sized for
             the sheet drawn DEMO_PX px wide, so 250 towns fit; every guarantee map_engine asserts is asserted.
3. prop      The trained propagator of s22c-cap99-v49-s0 (one layer, four channels, t0 = 0.99, rows capped) through
             search_train.propagator itself: per channel the gates G, the source S and X = S (I - G)^{-1}; the read-out;
             the working channel (largest log X margin of the answer over the start's other children). Also the
             hand-built layer (G = t0 A^T plus a background eps on every non-edge, rows capped, S = the
             goal column) and, by bisection, the largest background eps at which it still answers this question: the
             size law's bound for this graph.

    U=uv; T=<their train.py>
    $U run --offline --quiet --no-project --with numpy==2.3.4 nice -n 19 python -I search250.py question $T
    $U run --offline --quiet --no-project --with matplotlib==3.10.7 --with numpy==2.3.4 --with scipy --with networkx \
        nice -n 19 python -I search250.py map
    $U run --offline --quiet --no-project --with numpy==2.3.4 --with jax --with optax nice -n 19 python -I search250.py prop
Writes paper/figures/data/q250.json, map250.json, prop250.json.
"""
import hashlib
import json
import math
import pathlib
import random
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # repository root
SEARCH = HERE
DATA = ROOT / "paper" / "figures" / "data"
V, LOOK, INDEX, SEED = 250, 31, 1, 12345
RUN = "s22c-cap99-v49-s0"                   # the map's run (Fig. 2b); (d) shows all three seeds
RUNS = [f"s22c-cap99-v49-s{k}" for k in range(3)]
SHEET_CAP, DEMO_PX = 220.0, 1800.0


# ---------------------------------------------------------------------------------------------------- 1. question
def question(train_py):
    sys.path.insert(0, str(SEARCH))
    import l2s_eval
    ns = l2s_eval.load(train_py)
    n, pad = 3 * V + 5, V + 3
    random.seed(SEED)
    np.random.seed(SEED)                       # as l2s_eval.test_sets, which reseeds before every lookahead
    xs, ys = ns["generate_eval_data"](n, min_path_length=2, distance_from_start=-1, distance_from_end=-1,
                                      lookahead_steps=LOOK, num_paths_at_fork=None, num_samples=INDEX + 1,
                                      max_prefix_vertices=None)
    out = []
    for x, y in zip(xs, ys):
        toks = [int(t) for t in x if t != pad]
        q = toks.index(V + 4)
        edges = [[toks[k + 1], toks[k + 2]] for k in range(0, q, 3) if toks[k] == V + 2]
        names = sorted({a for e in edges for a in e})
        out.append(dict(tokens=[int(t) for t in x], answer=int(y), edges=edges, names=names, start=toks[q + 1],
                        goal=toks[q + 2], path=toks[q + 4:]))
    assert len(out[0]["names"]) < V and len(out[INDEX]["names"]) == V, "the rule picks question 1"
    Q = out[INDEX]
    assert Q["path"] == [Q["start"]]
    kids = {}
    for u, v in Q["edges"]:
        kids.setdefault(u, []).append(v)
    assert len(kids[Q["start"]]) == 8 and Q["answer"] in kids[Q["start"]]
    Q.update(vertices=V, lookahead=LOOK, index=INDEX, seed=SEED, tokens_per_question=n,
             source="generate_eval_data of their train.py (github.com/asaparov/learning_to_search, commit 2c3f32b), "
                    "sha1 of the file " + hashlib.sha1(pathlib.Path(train_py).read_bytes()).hexdigest())
    (DATA / "q250.json").write_text(json.dumps(Q, separators=(",", ":")))
    print("wrote data/q250.json:", len(Q["names"]), "towns,", len(Q["edges"]), "roads, start", Q["start"], "goal",
          Q["goal"], "answer", Q["answer"])


# ---------------------------------------------------------------------------------------------------- 2. map
def plan_wide(G, rng):
    """map_engine.plan with the sheet's cap raised from 112 to SHEET_CAP mm (everything else as there)."""
    import map_engine as me
    n = len(G.nodes)
    d0 = len(G.adj[G.start])
    a = float(np.clip(7.6 - 0.032 * n, 5.9, 7.4))
    W = math.sqrt(n * a * a * 3.4 / 0.8) + 14.0
    if d0 > 6:
        Rh = max(a, d0 * max(3.8, 0.66 * a) / (2 * math.pi))
        deep = max(G.size[k] for k in G.kids[G.start])
        W = max(W, (2 * (Rh * 1.12 + min(deep - 1, 3) * a * 0.9 + 6.0) + 14.0) / 0.8)
    W = float(np.clip(W, 62.0, SHEET_CAP))
    H = round(0.8 * W, 1)
    W = round(W, 1)
    side = me.SIDES[rng.choice(4, p=[0.34, 0.28, 0.26, 0.12])]
    band = (0.13 + 0.05 * rng.random()) * (H if side in "SN" else W)
    m = 3.2
    box = [m, m, W - m, H - m]
    if side == "S":
        box[3] -= band
    elif side == "N":
        box[1] += band
    elif side == "E":
        box[2] -= band
    else:
        box[0] += band
    return a, W, H, side, band, box


def make_map(first=0, attempts=6):
    sys.path.insert(0, str(HERE))
    import map_engine as me
    me.plan = plan_wide
    me.DEMO_PX = DEMO_PX
    me.place_labels.__defaults__ = (DEMO_PX,)
    Q = json.loads((DATA / "q250.json").read_text())
    G = me.Graph(Q["edges"], Q["start"], Q["goal"], Q["path"])
    L = me.layout(G, attempts=attempts, quiet=False, first=first)
    L["question"] = dict(vertices=V, lookahead=LOOK, index=INDEX, source="data/q250.json")
    (DATA / "map250.json").write_text(json.dumps(L, separators=(",", ":")))
    print("wrote data/map250.json: sheet", L["size"], "lakes", len(L["water"]["lakes"]), "rivers",
          len(L["water"]["rivers"]), "bridges", len(L["bridges"]))


# ---------------------------------------------------------------------------------------------------- 3. propagator
def load_run(run):
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(SEARCH))
    import search_train as st
    d = SEARCH / "results" / run
    row = json.loads((d / "rows.json").read_text())[0]
    cfg, P, Vt = row["cfg"], row["name_pool"], row["vertices"]
    tmpl = st.init(jax.random.PRNGKey(0), cfg, P + 5, 3 * Vt + 5)
    Wt = dict(np.load(d / f"params_s{row['seed']}.npz"))
    flat, tree = jax.tree_util.tree_flatten_with_path(tmpl)
    p = jax.tree_util.tree_unflatten(tree, [jnp.array(Wt[jax.tree_util.keystr(k)]) for k, _ in flat])
    return p, cfg, P


def hand_layer(A, goal, present, t0, eps):
    """The hand-built task layer: gates t0 on reversed edges and eps on every other pair of present names,
    rows capped at t0 (as the trained layer), source = the goal column; X_v = walks from v to the goal (float64)."""
    pm = np.outer(present, present)
    G = (t0 * A.T + eps * (1 - A.T)) * pm
    np.fill_diagonal(G, 0.0)
    G = G * (t0 / np.maximum(G.sum(1, keepdims=True), t0))
    N = len(A)
    R = np.linalg.solve(np.eye(N) - G, np.eye(N))     # R[k, v]: walks k -> v in G
    return R[goal]                                     # X_cv = sum_k S_ck R_kv with S_ck = [k = goal]


def prop():
    out = dict(runs={r: prop_run(r) for r in RUNS}, map_run=RUN)
    (DATA / "prop250.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote data/prop250.json")


def prop_run(run):
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(SEARCH))
    import search_train as st
    import loop_train as lt
    Q = json.loads((DATA / "q250.json").read_text())
    p, cfg, P = load_run(run)
    pad = V + 3
    x = np.asarray(Q["tokens"])[None]
    e = lt.marker_ids(x, V, P)
    ecfg = dict(cfg, bf16=False, remat=False)
    logits = np.asarray(lt.logits(p, ecfg, jnp.array(x), jnp.array(e), pad))[0]
    z, present, nv, Nn = st.pairn_start(p, jnp.array(x), pad, p["tok"][jnp.array(e)], cfg.get("pool_norm", False))
    t = (z - z.mean(1, keepdims=True)) * jax.lax.rsqrt(z.var(1, keepdims=True) + 1e-6)   # the block's cnorm
    b = p["blocks"][0]
    X = np.asarray(st.propagator(t, b, present, ecfg))[0]                                # [H, N, N]
    # G and S as the layer forms them (search_train.propagator, written out to keep them)
    mm = np.asarray(present[:, None, :, None] * present[:, None, None, :])[0]
    proj = lambda w, c: np.asarray(jnp.einsum("bcij,ch->bhij", t, w) + c[None, :, None, None])[0]  # noqa: E731
    t0 = float(cfg["t0"])
    sig = lambda a: 1 / (1 + np.exp(-a))  # noqa: E731
    G = t0 * sig(proj(b["pg"], b["pgb"])) * mm
    Graw = G.copy()
    G = G * (t0 / np.maximum(G.sum(-1, keepdims=True), t0))
    S = sig(proj(b["ps"], b["psb"])) * mm
    I = np.eye(G.shape[-1])
    X2 = np.stack([np.maximum(np.linalg.solve((I - g).T, s.T).T, 0) for g, s in zip(G.astype(np.float64), S)])
    assert np.allclose(X2, X, rtol=2e-3, atol=1e-6), float(np.abs(X2 - X).max())
    N = G.shape[-1]
    A = np.zeros((N, N))
    for u, v in Q["edges"]:
        A[u, v] = 1
    pres = np.asarray(present)[0] > 0
    cur, goal, ans = Q["path"][-1], Q["goal"], Q["answer"]
    kids = [int(v) for v in np.flatnonzero(A[cur])]
    said = int(np.argmax(logits))
    assert said == ans, ("the model answers", said, "not", ans)
    lx = np.log(X[:, cur] + 1e-20)
    margin = [float(lx[h, ans] - max(lx[h, c] for c in kids if c != ans)) for h in range(len(X))]
    work = int(np.argmax(margin))
    pm = np.outer(pres, pres) & ~np.eye(N, dtype=bool)
    rev, fwd = A.T.astype(bool) & pm, A.astype(bool) & pm
    non = pm & ~rev & ~fwd
    non[cur, goal] = False                     # the question's own pair (start, goal), kept apart: see query_gate
    gm = np.zeros((N, N), bool)
    gm[:, goal] = True
    ch = []
    bins = np.arange(-12, 0.01, 0.25)                                          # log10 gate histogram, non-edges
    for h in range(len(X)):
        g = G[h]
        ch.append(dict(
            rev=g[rev].tolist(), fwd=g[fwd].tolist(),
            non_hist=np.histogram(np.log10(np.maximum(g[non], 1e-12)), bins)[0].tolist(),
            non_quant={q: float(np.quantile(g[non], q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
            src_goal=S[h][pres & gm].tolist(), src_other_quant={q: float(np.quantile(S[h][pm & ~gm], q))
                                                                   for q in (0.5, 0.99, 1.0)},
            rowsum_max=float(Graw[h].sum(-1).max()), spectral_radius=float(np.abs(np.linalg.eigvals(g)).max()),
            margin_nats=margin[h], X_row=X[h, cur].tolist(), query_gate=float(G[h][cur, goal]),
            query_src=float(S[h][cur, goal])))
    # the hand-built layer's background bound on this question: largest eps with the answer still the arg max
    def right(eps):
        x_ = hand_layer(A, goal, pres.astype(float), t0, eps)
        return x_[ans] > max(x_[c] for c in kids if c != ans)
    lo, hi = -12.0, 0.0
    assert right(10 ** lo) and not right(10 ** -0.5)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if right(10 ** mid) else (lo, mid)
    hand0 = hand_layer(A, goal, pres.astype(float), t0, 0.0)
    out = dict(run=run, cfg=cfg, t0=t0, current=cur, goal=goal, answer=ans, said=said, kids=kids, names=Q["names"],
               logits_kids={str(c): float(logits[c]) for c in kids}, working_channel=work, channels=ch,
               non_bins=bins.tolist(), hand_X=hand0.tolist(), hand_eps_star_log10=lo,
               note="G, S and X as search_train.propagator forms them for the question of data/q250.json; X_row = "
                    "X[h, current, :] over names 0..V (absent names 0); rev/fwd = gates on reversed / forward edges; "
                    "hand_eps_star_log10: the largest background gate at which the hand-built layer still answers.")
    print(run, "said", said, "answer", ans, "working channel", work, "margins (nats)",
          np.round(margin, 2).tolist(), "eps* 10^%.2f" % lo)
    for h, c in enumerate(ch):
        print(h, "rev median %.4f min %.4f" % (np.median(c["rev"]), np.min(c["rev"])), "fwd max %.2e" % max(c["fwd"]),
              "non q50 %.2e q999 %.2e max %.2e" % (c["non_quant"][0.5], c["non_quant"][0.999], c["non_quant"][1.0]),
              "src goal median %.3f other q99 %.2e" % (np.median(c["src_goal"]), c["src_other_quant"][0.99]),
              "rho %.3f" % c["spectral_radius"])
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "question":
        question(sys.argv[2])
    elif what == "map":
        make_map(*(int(a) for a in sys.argv[2:]))
    elif what == "prop":
        prop()
