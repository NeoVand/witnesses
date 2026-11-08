"""Six-clock exponential Newton: score results/exp6-{rule,curve8k}-{a,b} and the registered predictions G1-G3
(Supplementary Table stbl-register2); writes summary6.json. Then prints the held-out-card runs
results/exp8-rule-{a,b} (seeds 3-5, rate picked on validation cards; predictions H1, H2), the source of Fig. 3a
(right) and 3e and of the six-clock rows of Supplementary Table stbl-set."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
pick = lambda row: int(np.argmax(np.asarray(row["val"]).sum(1)))
pc = lambda v: ", ".join(f"{100 * x:.1f}" for x in v)


def rank3(V):
    V = [list(np.asarray(v) % 3) for v in V]
    r = 0
    for c in range(4):
        piv = next((i for i in range(r, len(V)) if V[i][c] % 3), None)
        if piv is None:
            continue
        V[r], V[piv] = V[piv], V[r]
        inv = 1 if V[r][c] == 1 else 2
        V[r] = [(e * inv) % 3 for e in V[r]]
        for i in range(len(V)):
            if i != r and V[i][c]:
                f = V[i][c]
                V[i] = [(a - f * b) % 3 for a, b in zip(V[i], V[r])]
        r += 1
    return r


out, cover = {}, []
for r in "ab":
    row = load(R / f"exp6-rule-{r}" / "fits.jsonl")[0]
    i = pick(row)
    res = dict(lr=row["lrs"][i], params=row["params"], val_all=row["val"], beta=row["beta"][i], train=row["train"][i], seconds=row["seconds"])
    for T in (8, 16, 20):
        for side in ("in", "out"):
            k = f"{side}{T}"
            for rd in ("A", "B", "A64", "Apairs"):
                res[f"{k}_{rd}"] = row[k][rd][i]
            res[f"{k}_A_other"] = row[k]["A"][1 - i]
    res["clocks"] = []
    for s, c in enumerate(row["clocks"][i]):
        t = c["table"]
        good = [sv for sv, f in zip(t["s"], t["fit"]) if f >= .95]
        rk = rank3(good) if good else 0
        res["clocks"].append(dict(fit=[round(f, 3) for f in t["fit"]], s=t["s"], rank_good=rk, conf=[round(v, 3) for v in t["confidence"]],
                                  attrs=sorted({int(a) for sv in good for a in np.flatnonzero(sv)})))
        cover.append(rk == 4)
    res["clocks_other_rate_rank"] = [rank3([sv for sv, f in zip(c["table"]["s"], c["table"]["fit"]) if f >= .95]) for c in row["clocks"][1 - i]]
    out[f"rule_{r}"] = res
    row = load(R / f"exp6-curve8k-{r}" / "cells.jsonl")[0]
    i = pick(row)
    out[f"cell_{r}"] = dict(lr=row["lrs"][i], test=row["test"][i], mean=float(np.mean(row["test"][i])), test64=row["test64"][i],
                            test_trainform=row["test_trainform"][i], beta=row["beta"][i], train=row["train"][i], test_all=row["test"], val_all=row["val"])
a = out["rule_a"]
g = dict(G1=dict(covered=int(sum(cover)), of=len(cover), passed=bool(sum(cover) >= 5)),
         G2=dict(lowest=min(min(a[f"{k}_A"]) for k in ("in16", "out16", "in20", "out20")),
                 passed=bool(all(min(a[f"{k}_A"]) >= .95 for k in ("in16", "out16", "in20", "out20")))),
         G3=dict(mean=out["cell_a"]["mean"], passed=bool(out["cell_a"]["mean"] >= .99)))
g["G2_rerun_b_lowest"] = min(min(out["rule_b"][f"{k}_A"]) for k in ("in16", "out16", "in20", "out20"))
out["grades"] = g
(HERE / "summary6.json").write_text(json.dumps(out, indent=1))
for r in "ab":
    a = out[f"rule_{r}"]
    print(f"rule {r}: lr {a['lr']} val {np.round(np.array(a['val_all']) * 100, 1).tolist()} beta {np.round(a['beta'], 2).tolist()} train {pc(a['train'])} params {a['params']} {a['seconds']:.0f} s")
    for T in (8, 16, 20):
        for side in ("in", "out"):
            k = f"{side}{T}"
            print(f"  {k}: A {pc(a[k + '_A'])} | B {pc(a[k + '_B'])} | A64 {pc(a[k + '_A64'])} | pairs {pc(a[k + '_Apairs'])} | other rate {pc(a[k + '_A_other'])}")
    for c in a["clocks"]:
        print("  clocks", c)
    print("  other rate ranks", a["clocks_other_rate_rank"])
    c = out[f"cell_{r}"]
    print(f"cell {r} 8192: lr {c['lr']} test {pc(c['test'])} (mean {100 * c['mean']:.1f}) test64 {pc(c['test64'])} trainform {pc(c['test_trainform'])} beta {np.round(c['beta'], 2).tolist()} all {np.round(np.array(c['test_all']) * 100, 1).tolist()}")
print(json.dumps(g))

for r in "ab":
    f = R / f"exp8-rule-{r}" / "fits.jsonl"
    if not f.exists():
        continue
    row = load(f)[0]
    i = pick(row)                                            # validation cards in hand (val); val_own is the old rule
    print(f"held-out-card rule {r}: seeds {row['seeds']} lr {row['lrs'][i]} (own-card rule would pick {row['lrs'][pick(dict(val=row['val_own']))]}),"
          f" val cards {pc(row['val'][i])} | other rate {pc(row['val'][1 - i])}; beta {np.round(row['beta'][i], 2).tolist()}")
    for T in (8, 16, 20):
        t = row[f"test{T}"]
        print(f"  test cards, {T} table cards: A {pc(t['A'][i])} | A64 {pc(t['A64'][i])} | pairs {pc(t['Apairs'][i])} | B {pc(t['B'][i])}"
              f" | other rate A {pc(t['A'][1 - i])}")
    lo8 = min(row["test8"]["A"][i]); lo = min(min(row[f"test{T}"]["A"][i]) for T in (16, 20))
    print(f"  H1 (8 cards >= 99% on 3 of 3): {'PASS' if lo8 >= .99 else 'FAIL'} (lowest {100 * lo8:.1f});"
          f" H2 (16 and 20 cards >= 95% on 3 of 3): {'PASS' if lo >= .95 else 'FAIL'} (lowest {100 * lo:.1f})")
