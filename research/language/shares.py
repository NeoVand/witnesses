"""Interaction shares of trained and untrained joins, recomputed from saved weights for Fig. 5d (CPU only).

The interaction share of a score over the partner pairs (j, k) a join scores is the part of its variance that is not a
sum f(j) + g(k), divided by its value for independent random scores: (n - 1)^2 / (n^2 - 1) on the full grid (joins
that allow j = k, as chess joins beside attention), (n^2 - 3n + 1) / (n^2 - n - 1) on the off-diagonal pairs (SET
joins and chess joins replacing attention, which score j != k only; additive part fitted by least squares). Sums of
squares are taken about each grid's own mean and pooled over positions or deals, heads and queries.
- SET: every saved one-layer join of set/results8 (7 files x 2 rates x 3 seeds = 42 fits), score
  A(q, j, k) = sum_c Q(q)_c K1(j)_c K2(k)_c / sqrt(R) as in set/mechanism.py, on 2,000 random deals of 8 table
  cards, the card you hold drawn from all 81; and the same join at initialisation (set/bench2.py init, seeds
  0 to 2).
- Chess: each join block of the saved seed-0 chess models (results/c24-cub8-s0, c25-full-hyb8c, c26-full-hyb8r), on
  the first 512 validation positions of results/c24-data, over the 64 x 64 grid of squares; the same configurations
  at initialisation (seeds 0 and 1). The parameters and validation arrays are on the witnesses-language volume.
Writes results/shares.json; name parts to recompute only those (set, set_init, chess).

    JAX_PLATFORMS=cpu nice -n 19 python -I shares.py [set set_init chess]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # research/: paths in shares.json are relative to it
CON, LAB = ROOT / "set", HERE
OUT = HERE / "results" / "shares.json"


def pooled(s, n):
    """s [..., n, n]; pooled interaction share relative to random scores."""
    r = s - s.mean(-1, keepdims=True) - s.mean(-2, keepdims=True) + s.mean((-1, -2), keepdims=True)
    t = s - s.mean((-1, -2), keepdims=True)
    return float((r ** 2).sum() / (t ** 2).sum()) / ((n - 1) ** 2 / (n ** 2 - 1))


def offdiag_sums(s, n, xp=np, iters=60):
    """Interaction and total sums of squares over the off-diagonal pairs j != k only (the pairs a join that excludes
    j = k scores): the additive part f(j) + g(k) is fitted to those cells by least squares (backfitting)."""
    off = 1.0 - xp.eye(n)
    cnt = n * (n - 1)
    mu = (s * off).sum((-1, -2), keepdims=True) / cnt
    a = xp.zeros(s.shape[:-1] + (1,))
    b = xp.zeros(s.shape[:-2] + (1, n))
    for _ in range(iters):
        a = ((s - mu - b) * off).sum(-1, keepdims=True) / (n - 1)
        b = ((s - mu - a) * off).sum(-2, keepdims=True) / (n - 1)
    r = (s - mu - a - b) * off
    t = (s - mu) * off
    return (r ** 2).sum(), (t ** 2).sum()


OFF_RANDOM = {}


def offdiag_random(n):
    """The off-diagonal interaction share of independent random scores, (n^2 - 3n + 1) / (n^2 - n - 1)."""
    return (n * n - 3 * n + 1) / (n * n - n - 1)


def pooled_off(s, n):
    a, b = offdiag_sums(s, n)
    return float(a / b) / offdiag_random(n)


def set_shares(deals=2000):
    sys.path.insert(0, str(CON))
    from mechanism import CARDS, R, features  # noqa: F401
    rng = np.random.default_rng(0)
    qs = rng.integers(0, 81, deals)
    tables = np.stack([rng.choice(np.setdiff1d(np.arange(81), [q]), 8, replace=False) for q in qs])
    out = []
    for f in sorted(CON.glob("results8/q-m*/params_*_c1.npz")):
        i = int(f.stem.split("_")[1])
        fit = [json.loads(x) for x in (f.parent / "fits.jsonl").open()][i]
        m, _, N, _ = fit["stages"][0]
        P = dict(np.load(f))
        get = lambda name, li, s: P[f"['{name}']"][li, s]  # noqa: E731
        nlr, nseed = P["['q']"].shape[:2]
        for li in range(nlr):
            for s in range(nseed):
                hq = np.tanh(features(True) @ get("input", li, s))
                ht = np.tanh(features(False) @ get("input", li, s))
                A = np.einsum("qc,jc,kc->qjk", hq @ get("q", li, s), ht @ get("k1", li, s), ht @ get("k2", li, s)) / R ** .5
                grids = A[qs[:, None, None], tables[:, :, None], tables[:, None, :]]
                out.append(dict(file=str(f.relative_to(ROOT)), m=m, deals=N, rate_index=li, seed=s,
                                share=pooled_off(grids, 8), share_full_grid=pooled(grids, 8)))
    return out


def set_init_shares(deals=2000):
    """The same score for the one-layer join at initialisation (set/bench2.py init, seeds 0 to 2 as trained)."""
    import jax  # noqa: F401
    sys.path.insert(0, str(CON))
    import bench2
    from mechanism import R, features
    rng = np.random.default_rng(0)
    qs = rng.integers(0, 81, deals)
    tables = np.stack([rng.choice(np.setdiff1d(np.arange(81), [q]), 8, replace=False) for q in qs])
    out = []
    for seed in (0, 1, 2):
        p = {k: np.asarray(v) for k, v in bench2.init(seed, bench2.ARMS["c1"]).items() if k in ("input", "q", "k1", "k2")}
        hq, ht = np.tanh(features(True) @ p["input"]), np.tanh(features(False) @ p["input"])
        A = np.einsum("qc,jc,kc->qjk", hq @ p["q"], ht @ p["k1"], ht @ p["k2"]) / R ** .5
        g_ = A[qs[:, None, None], tables[:, :, None], tables[:, None, :]]
        out.append(dict(seed=seed, share=pooled_off(g_, 8), share_full_grid=pooled(g_, 8)))
    return out


def chess_shares(npos=512, bs=8):
    import jax
    import jax.numpy as j
    sys.path.insert(0, str(LAB))
    import chess_train as c1
    import chess_train2 as c2
    D = LAB / "results" / "c24-data"
    sq, ex = np.load(D / "val_sq.npy")[:npos], np.load(D / "val_extra.npy")[:npos]
    base = dict(d=256, H=8)
    arms = [("cub8", c1, dict(base, layers=["sa", "sa", "cubic", "sa", "sa", "cubic", "sa", "sa"]),
             "language/results/c24-cub8-s0/params_s0.npz"),
            ("hyb8c", c2, dict(base, layers=["sa", "sa", "hyb", "sa", "sa", "hyb", "sa", "sa"], connected=True,
                               diag=True, impl="xla"), "language/results/c25-full-hyb8c/params_s0.npz"),
            ("hyb8r", c2, dict(base, layers=["sa", "sa", "hyb", "sa", "sa", "hyb", "sa", "sa"], diag=True, impl="xla"),
             "language/results/c26-full-hyb8r/params_s0.npz")]

    def load(mod, cfg, path, seed):
        p = mod.init(jax.random.PRNGKey(seed), cfg)
        if path is None:
            return p
        P = dict(np.load(ROOT / path))
        flat, tree = jax.tree_util.tree_flatten_with_path(p)
        return jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])

    def scores(mod, cfg, p, s_, e_):
        """Per join block: sum over positions, heads and queries of the interaction and total sums of squares."""
        x = p["piece"][s_.astype(j.int32)] + p["square"][None]
        e = e_[:, :4].astype(j.float32) @ p["castle"] + p["ep"][e_[:, 4].astype(j.int32)] + p["extra"]
        h = j.concatenate([x, e[:, None]], 1)
        res = {}
        for li, (kind, b) in enumerate(zip(cfg["layers"], p["blocks"])):
            xn = mod.norm(h, b["g1"])
            if kind in ("cubic", "hyb"):
                c = b if kind == "cubic" else {k[2:]: v for k, v in b.items() if k.startswith("c_")}
                B, n, d = xn.shape
                dh = d // cfg["H"]
                sp = lambda t: t.reshape(B, t.shape[1], cfg["H"], dh)  # noqa: E731
                q = sp(xn @ c["q"])
                k1, k2 = sp(xn[:, :64] @ c["k1"]), sp(xn[:, :64] @ c["k2"])
                if cfg.get("center"):
                    k1, k2 = k1 - k1.mean(1, keepdims=True), k2 - k2.mean(1, keepdims=True)
                s = j.einsum("bihc,bjhc,bkhc->bhijk", q, k1, k2) / dh ** .5
                if kind == "cubic":       # joins replacing attention score j != k only
                    a_, b_ = offdiag_sums(s[:, :, :, :64, :64], 64, xp=j)
                    res[li] = (float(a_) / offdiag_random(64) * (63 ** 2 / 4095), float(b_))
                else:                     # joins beside attention allow j = k
                    r = s - s.mean(-1, keepdims=True) - s.mean(-2, keepdims=True) + s.mean((-1, -2), keepdims=True)
                    t = s - s.mean((-1, -2), keepdims=True)
                    res[li] = (float((r ** 2).sum()), float((t ** 2).sum()))
            mix = mod.mixer(kind, b, xn, cfg["H"]) if mod is c1 else mod.mixer(kind, b, xn, cfg["H"], cfg)
            h = h + mix
            h = h + jax.nn.gelu(mod.norm(h, b["g2"]) @ b["f1"]) @ b["f2"]
        return res

    out = []
    for name, mod, cfg, path in arms:
        for state, seeds in (("trained", (0,)), ("init", (0, 1))):
            for seed in seeds:
                p = load(mod, cfg, path if state == "trained" else None, seed)
                fn = jax.jit(lambda p, s_, e_, mod=mod, cfg=cfg: scores(mod, cfg, p, s_, e_)) if False else None
                acc = {}
                for i in range(0, npos, bs):
                    for li, (a, b_) in scores(mod, cfg, p, j.array(sq[i:i + bs]), j.array(ex[i:i + bs])).items():
                        acc.setdefault(li, [0.0, 0.0])
                        acc[li][0] += a
                        acc[li][1] += b_
                for li, (a, b_) in acc.items():
                    out.append(dict(arm=name, state=state, seed=seed, block=li, params=path if state == "trained" else None,
                                    share=a / b_ / (63 ** 2 / 4095)))
                print(name, state, seed, {li: round(a / b_ / (63 ** 2 / 4095), 3) for li, (a, b_) in acc.items()},
                      flush=True)
    return out


if __name__ == "__main__":
    parts = sys.argv[1:] or ["set", "set_init", "chess"]          # recompute only the parts named
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["how"] = __doc__.split("Writes")[0].strip()
    for part, fn in (("set", set_shares), ("set_init", set_init_shares), ("chess", chess_shares)):
        if part in parts:
            res[part] = fn()
            v = [r["share"] for r in res[part]]
            print(part, len(v), "values:", round(min(v), 3), "to", round(max(v), 3), flush=True)
            OUT.write_text(json.dumps(res, indent=1))
