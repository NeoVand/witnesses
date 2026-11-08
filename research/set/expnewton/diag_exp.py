"""Where do the four-clock layers' errors fall (no training)? Runs on Modal next to the saved weights
(volume path connected-expnewton/exp-rule-{a,b}/params_rule0.npz) and writes diag.json (results/exp-diag/).

A pseudo-set pair is a table pair on which every learned clock character vanishes (score at its maximum) but which is
not a set, i.e. it completes the attributes the clocks read and misses another. For each model (rerun, rate, seed) and
test set: the clocks' attributes, and the error rate on deals with and without a pseudo-set pair, by label."""
import json, sys
from pathlib import Path
import numpy as np, jax.numpy as j
import bench8, bench_exp as E

TESTS = dict(in8=(20000, None, 8), out8=(20001, bench8.HELD, 8), in16=(20002, None, 16), out16=(20003, bench8.HELD, 16),
             in20=(21200, None, 20), out20=(21201, bench8.HELD, 20))


def attrs_of(x):
    return np.stack([x[..., 3 * a:3 * a + 3].argmax(-1) for a in range(4)], -1)          # (B, T, 4)


def main():
    job = json.loads(sys.argv[1])
    out = Path(job["dest"])
    S = bench8.query_set(27)
    data = {k: bench8.deals(seed, 1024, S if S_ is None else S_, T) for k, (seed, S_, T) in TESTS.items()}
    res = []
    for r in "ab":
        d = Path(f"/vol/connected-expnewton/exp-rule-{r}")
        row = json.loads((d / "fits.jsonl").read_text().splitlines()[0])
        P = dict(np.load(d / "params_rule0.npz"))
        for i, lr in enumerate(row["lrs"]):
            for s in range(3):
                p = {k: j.array(v[i, s]) for k, v in P.items()}
                svec = np.array(row["clocks"][i][s]["table"]["s"])                      # (4 clocks, 4)
                item = dict(rerun=r, lr=lr, seed=s, s=svec.tolist(), rank=row["clocks"][i][s]["table"]["rank"],
                            attributes=sorted({int(a) for v in svec for a in np.flatnonzero(v)}))
                for name, (x, y) in data.items():
                    pred = np.asarray(E.apply(p, j.array(x)).argmax(-1))
                    A = attrs_of(x)
                    T = x.shape[1] - 2
                    tab, q = A[:, 1:T + 1], A[:, T + 1]
                    jj, kk = np.triu_indices(T, 1)
                    tot = (q[:, None] + tab[:, jj] + tab[:, kk]) % 3                        # (B, P, 4)
                    is_set = (tot == 0).all(-1)
                    clock_zero = ((tot @ svec.T) % 3 == 0).all(-1)
                    pseudo = (clock_zero & ~is_set).any(1)
                    err = pred != y
                    cell = {}
                    for lab in (0, 1):
                        for ps in (False, True):
                            m = (y == lab) & (pseudo == ps)
                            cell[f"y{lab}_pseudo{int(ps)}"] = dict(deals=int(m.sum()), errors=int((err & m).sum()))
                    cell["errors"] = int(err.sum())
                    cell["errors_on_pseudo_deals"] = int((err & pseudo).sum())
                    item[name] = cell
                res.append(item)
    (out / "diag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(models=len(res))), flush=True)


if __name__ == "__main__":
    main()
