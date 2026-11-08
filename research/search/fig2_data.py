"""Fig. 2's own data, beyond q250.json, map250.json and prop250.json (search250.py), computed on a CPU from saved
files only; nothing is trained and nothing is downloaded. Reads and writes paper/figures/data/.

1. map     The same 250-vertex question laid out by map_engine.py (this folder) on a wide sheet (W x H below) 
           whose map spans the page: map_engine unchanged except plan() (the sheet's size, the sea's side and band,
           as search250.plan_wide changes it) and marks sized for the sheet drawn DEMO_PX px wide. Every guarantee
           map_engine asserts is asserted. Writes data/fig2_map_wide.json.
2. gates   The trained propagators of s22c-cap99-v49-s{0,1,2} on the question, through search250.load_run and
           search_train itself: the working channel's gates G (rows capped, as the layer forms them) on every
           reversed road, every forward road, the pair (start, goal), and every pair of present names without a road
           (all 61,752 kept as a log10 histogram and as a fixed random draw of 249 per seed for the scatter); checked
           against prop250.json's stored values; for the map's run (s0) also the whole matrix G, to 3 digits.
           Writes data/fig2_gates.json.

    U=uv
    $U run --offline --quiet --no-project --with matplotlib==3.10.7 --with numpy==2.3.4 --with scipy --with networkx \
        --with contourpy nice -n 19 python -I fig2_data.py map [W H side band salt]
    $U run --offline --quiet --no-project --with numpy==2.3.4 --with jax --with optax nice -n 19 python -I fig2_data.py gates
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # repository root
DATA = ROOT / "paper" / "figures" / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
DEMO_PX = 1800.0
WIDE = dict(W=265.0, H=120.0, side="S", band=0.12, salt=3)    # chosen from 8 sheets by eye: the clearest answer road


def plan_sheet(W, H, side, band_frac):
    def plan(G, rng):
        import map_engine as me
        n = len(G.nodes)
        a = float(np.clip(7.6 - 0.032 * n, 5.9, 7.4))
        rng.choice(4, p=[0.34, 0.28, 0.26, 0.12])       # keep the engine's draws in step
        rng.random()
        band = band_frac * (H if side in "SN" else W)
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
    return plan


def make_map(W, H, side, band, salt, out):
    import map_engine as me
    me.plan = plan_sheet(W, H, side, band)
    me.DEMO_PX = DEMO_PX
    me.place_labels.__defaults__ = (DEMO_PX,)
    Q = json.loads((DATA / "q250.json").read_text())
    G = me.Graph(Q["edges"], Q["start"], Q["goal"], Q["path"])
    L = me.layout(G, attempts=4, quiet=False, first=salt)
    L["question"] = dict(vertices=250, lookahead=31, index=1, source="data/q250.json",
                         sheet=dict(W=W, H=H, side=side, band=band, salt=salt))
    pathlib.Path(out).write_text(json.dumps(L, separators=(",", ":")))
    print("wrote", out, "sheet", L["size"], "lakes", len(L["water"]["lakes"]), "rivers", len(L["water"]["rivers"]),
          "woods", len(L["woods"]), "bridges", len(L["bridges"]))


def gates(n_sample=249, seed=20261107):
    import jax
    import jax.numpy as jnp
    import search250 as s250
    sys.path.insert(0, str(s250.SEARCH))
    import search_train as st
    import loop_train as lt
    Q = json.loads((DATA / "q250.json").read_text())
    P = json.loads((DATA / "prop250.json").read_text())
    V = s250.V
    out = dict(note="Working channel's gates G[u, v] (walk u -> v; a road p -> c is walked back as G[c, p]) of the "
                    "trained propagators on data/q250.json, rows capped as search_train.propagator forms them. "
                    "back: G[c, p] for every road p -> c, in data/q250.json's edge order; along: G[p, c]; "
                    "none: a fixed random draw of the pairs of present names with no road either way (the pair "
                    "(start, goal) excluded), with the full log10 histogram of all of them; query: G[start, goal].",
               runs={})
    rng = np.random.default_rng(seed)
    for run in s250.RUNS:
        p, cfg, Pn = s250.load_run(run)
        pad = V + 3
        x = np.asarray(Q["tokens"])[None]
        e = lt.marker_ids(x, V, Pn)
        z, present, nv, Nn = st.pairn_start(p, jnp.array(x), pad, p["tok"][jnp.array(e)], cfg.get("pool_norm", False))
        t = (z - z.mean(1, keepdims=True)) * jax.lax.rsqrt(z.var(1, keepdims=True) + 1e-6)
        b = p["blocks"][0]
        mm = np.asarray(present[:, None, :, None] * present[:, None, None, :])[0]
        proj = lambda w, c: np.asarray(jnp.einsum("bcij,ch->bhij", t, w) + c[None, :, None, None])[0]  # noqa: E731
        t0 = float(cfg["t0"])
        sig = lambda a: 1 / (1 + np.exp(-a))  # noqa: E731
        G = t0 * sig(proj(b["pg"], b["pgb"])) * mm
        G = G * (t0 / np.maximum(G.sum(-1, keepdims=True), t0))
        h = P["runs"][run]["working_channel"]
        g = G[h].astype(np.float64)
        N = g.shape[0]
        A = np.zeros((N, N), bool)
        for u, v in Q["edges"]:
            A[u, v] = True
        pres = np.asarray(present)[0] > 0
        pm = np.outer(pres, pres) & ~np.eye(N, dtype=bool)
        cur, goal = Q["start"], Q["goal"]
        back = [float(g[v, u]) for u, v in Q["edges"]]
        along = [float(g[u, v]) for u, v in Q["edges"]]
        non = pm & ~A & ~A.T
        non[cur, goal] = False
        vals = g[non]
        ch = P["runs"][run]["channels"][h]
        assert np.allclose(sorted(back), sorted(ch["rev"]), rtol=1e-4, atol=1e-7), run
        assert np.allclose(sorted(along), sorted(ch["fwd"]), rtol=1e-4, atol=1e-7), run
        assert abs(g[cur, goal] - ch["query_gate"]) < 1e-5 * max(1.0, ch["query_gate"]), run
        bins = np.asarray(P["runs"][run]["non_bins"])
        hist = np.histogram(np.log10(np.maximum(vals, 1e-12)), bins)[0]
        assert hist.tolist() == ch["non_hist"], run
        pick = rng.choice(len(vals), n_sample, replace=False)
        full = {}
        if run == P["map_run"]:                 # the map's run: every gate, for the magnifier's pairs (3 digits)
            full = dict(G=[[float("%.3g" % v) for v in row] for row in g])
        out["runs"][run] = dict(full, channel=h, back=back, along=along, none_sample=vals[np.sort(pick)].tolist(),
                                none_count=int(non.sum()), none_hist=hist.tolist(), bins=bins.tolist(),
                                none_max=float(vals.max()), none_median=float(np.median(vals)),
                                query=float(g[cur, goal]), t0=t0,
                                eps_star_log10=P["runs"][run]["hand_eps_star_log10"])
        print(run, "channel", h, "back median %.4f" % np.median(back), "along max %.2e" % max(along),
              "none", int(non.sum()), "max %.2e" % vals.max(), "query %.3f" % g[cur, goal])
    (DATA / "fig2_gates.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote data/fig2_gates.json")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "map":
        a = sys.argv[2:]
        if a:
            W, H, side, band, salt = float(a[0]), float(a[1]), a[2], float(a[3]), int(a[4])
            out = a[5] if len(a) > 5 else DATA / "fig2_map_wide.json"
        else:
            W, H, side, band, salt = WIDE["W"], WIDE["H"], WIDE["side"], WIDE["band"], WIDE["salt"]
            out = DATA / "fig2_map_wide.json"
        make_map(W, H, side, band, salt, out)
    elif what == "gates":
        gates()
