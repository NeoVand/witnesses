"""Four-clock exponential Newton: score results/exp-{rule,curve}-{a,b} and the registered predictions E1-E4
(Supplementary Table stbl-register2). Writes summary.json (curve_a is Fig. 3a's curve).
Rate per fit: the highest validation accuracy summed over the three seeds (first rate wins a tie), as for every arm."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pick(row):
    return int(np.argmax(np.asarray(row["val"]).sum(1)))


out = {}
for r in "ab":
    f = R / f"exp-rule-{r}" / "fits.jsonl"
    if not f.exists():
        continue
    row = load(f)[0]
    i = pick(row)
    res = dict(lr=row["lrs"][i], params=row["params"], seconds=row["seconds"], final_loss=row["final_loss"][i], train=row["train"][i],
               val=row["val"][i], beta=row["beta"][i], val_all=row["val"])
    for T in (8, 16, 20):
        for side in ("in", "out"):
            k = f"{side}{T}"
            for reading in ("A", "B", "A64", "B64"):
                res[f"{k}_{reading}"] = row[k][reading][i]
            res[f"{k}_A_other_rate"] = row[k]["A"][1 - i]
    res["clocks"] = [dict(beta=c["beta"], fit=[round(v, 3) for v in c["table"]["fit"]], rank=c["table"]["rank"], s=c["table"]["s"],
                          conf=[round(v, 3) for v in c["table"]["confidence"]], query_fit=[round(v, 3) for v in c["query"]["fit"]]) for c in row["clocks"][i]]
    out[f"rule_{r}"] = res
for r in "ab":
    f = R / f"exp-curve-{r}" / "cells.jsonl"
    if not f.exists():
        continue
    cells = {}
    for row in load(f):
        i = pick(row)
        cells[row["N"]] = dict(lr=row["lrs"][i], test=row["test"][i], mean=float(np.mean(row["test"][i])), test64=row["test64"][i],
                               train=row["train"][i], beta=row["beta"][i], test_all=row["test"], val_all=row["val"], final_loss=row["final_loss"][i])
    first = next((n for n in sorted(cells) if cells[n]["mean"] >= .9), None)
    out[f"curve_{r}"] = dict(cells=cells, first90=first)

g = {}
a = out.get("rule_a")
if a:
    g["E1"] = dict(out8_mean=float(np.mean(a["out8_A"])), every_seed=bool(min(a["out8_A"]) >= .99), passed=bool(np.mean(a["out8_A"]) >= .99))
    for reading in ("A", "B"):
        vals = {k: a[f"{k}_{reading}"] for k in ("in16", "out16", "in20", "out20")}
        g["E2" if reading == "A" else "E2B"] = dict(lowest=min(min(v) for v in vals.values()), passed=all(min(v) >= .95 for v in vals.values()))
    exact = [c for c, o in zip(a["clocks"], a["out8_A"]) if o >= .99]
    g["E4"] = dict(runs=len(exact), passed=bool(exact) and all(min(c["fit"]) >= .95 and c["rank"] == 4 for c in exact)) if exact else dict(runs=0, passed=None)
c = out.get("curve_a")
if c:
    g["E3"] = dict(first90=c["first90"], passed=c["first90"] is not None and c["first90"] <= 8192)
out["grades"] = g
(HERE / "summary.json").write_text(json.dumps(out, indent=1))


def pc(v):
    return ", ".join(f"{100 * x:.1f}" for x in v)


for r in "ab":
    a = out.get(f"rule_{r}")
    if a:
        print(f"rule {r}: lr {a['lr']}, params {a['params']}, beta {np.round(a['beta'], 2).tolist()}, train {pc(a['train'])}, loss {np.round(a['final_loss'], 3).tolist()}, {a['seconds']:.0f} s")
        for T in (8, 16, 20):
            for side in ("in", "out"):
                k = f"{side}{T}"
                print(f"  {k}: A {pc(a[k + '_A'])} | B {pc(a[k + '_B'])} | A64 {pc(a[k + '_A64'])} | B64 {pc(a[k + '_B64'])} | other rate A {pc(a[k + '_A_other_rate'])}")
        for cl in a["clocks"]:
            print("  clocks:", cl)
    c = out.get(f"curve_{r}")
    if c:
        for n, v in sorted(c["cells"].items()):
            print(f"curve {r} N={n}: lr {v['lr']} test {pc(v['test'])} (mean {100 * v['mean']:.1f}) test64 {pc(v['test64'])} train {pc(v['train'])} beta {np.round(v['beta'], 2).tolist()}")
        print(f"curve {r}: first mean >= 90% at {c['first90']}")
print(json.dumps(g))
