"""Export every Newton number on SET (results_nset/) and the comparison arms (results2/, results3/, results4/w2,
results8/q-m27a and q-m27b) to newton_data.json, which Fig. 3 reads. Standard library only.
Run from research/set:  python3 -I newton_paper/export.py
Accuracies are percentages; seeds are 0, 1, 2 in order; "picked" is the rate with the highest validation accuracy
summed over the three seeds (first rate wins a tie), the rule every arm in the paper uses.
"""
import json
from pathlib import Path
CON = Path(__file__).resolve().parent.parent
R = CON / "results_nset"
OUT = CON / "newton_paper" / "newton_data.json"
SIZES = [2048, 8192, 32768, 131072, 524288]
USED = {"newton_bare": 55714, "newton_oscale": 55714, "newton_r256": 203554}   # parameters the layer and head use


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pick(r):
    return max(range(len(r["lrs"])), key=lambda n: sum(r["val"][n]))


def pc(v):
    return [round(100 * x, 2) for x in v]


def chars(r, i, s):
    c = r["chars"]["trained"][i][s]
    T, H, I = c["table"], c["hand"], r["chars"]["init"][s]["table"]
    return dict(found=T["found"], share=round(T["share"], 4), baseline_mean=round(T["base_mean"], 4), baseline_max=round(T["base_max"], 4),
                hand_found=H["found"], hand_share=round(H["share"], 4), untrained_share=round(I["share"], 4), untrained_found=I["found"])


JOBS = {  # job id: (arm, rerun, role)
    "nset-smoke": ("newton_bare", None, "smoke, 1,000 updates at 2k and 500 in the rule path; training loss only"),
    "nset-p1": ("newton_bare", "2", "P1, rank 64 unfixed; run 1 of this cell is results4/w2"),
    "nset-r256-smoke": ("newton_r256", None, "smoke, 1,000 updates at 2k; training loss only"),
    "nset-r256-p1-a": ("newton_r256", "a", "R1, rank 256 unfixed"), "nset-r256-p1-b": ("newton_r256", "b", "R1, rank 256 unfixed"),
    "nset-oscale-p1-a": ("newton_oscale", "a", "F1, rank 64 fixed"), "nset-oscale-p1-b": ("newton_oscale", "b", "F1, rank 64 fixed"),
    "nset-oscale-curve-a": ("newton_oscale", "a", "P2', rank 64 fixed"), "nset-oscale-curve-b": ("newton_oscale", "b", "P2', rank 64 fixed"),
    "nset-oscale-rule-a": ("newton_oscale", "a", "P3', rank 64 fixed"), "nset-oscale-rule-b": ("newton_oscale", "b", "P3', rank 64 fixed"),
}


def runs():
    out = []
    for job, (arm, rerun, role) in JOBS.items():
        for r in jl(R / job / "cells.jsonl"):
            i0 = pick(r)
            for i, lr in enumerate(r["lrs"]):
                for s in range(3):
                    out.append(dict(job=job, arm=arm, params=USED[arm], rerun=rerun, role=role, kind="curve", N=r["N"], updates=r["updates"], lr=lr, seed=s,
                                    picked=i == i0, test=round(100 * r["test"][i][s], 2), train=round(100 * r["train"][i][s], 2), val=round(100 * r["val"][i][s], 2),
                                    final_loss=round(r["final_loss"][i][s], 5), **chars(r, i, s)))
        for r in jl(R / job / "fits.jsonl"):
            i0 = pick(r)
            m, T, N, U = r["stages"][-1]
            for i, lr in enumerate(r["lrs"]):
                for s in range(3):
                    row = dict(job=job, arm=arm, params=USED[arm], rerun=rerun, role=role, kind="rule", m=m, N=N, updates=U, lr=lr, seed=s, picked=i == i0,
                               train=round(100 * r["train"][-1][i][s], 2), val=round(100 * r["val"][i][s], 2), final_loss=round(r["final_loss"][-1][i][s], 5))
                    for k in ("in8", "out8", "in16", "out16", "full8"):
                        if k in r:
                            row[k] = round(100 * r[k][i][s], 2)
                    out.append(row | chars(r, i, s))
    return out


def picked_curve(arm, jobs):
    cells = []
    for rerun, ids in jobs.items():
        for job in ids:
            for r in jl(R / job / "cells.jsonl"):
                i = pick(r)
                cells.append(dict(N=r["N"], rerun=rerun, lr=r["lrs"][i], lrs=r["lrs"], test=pc(r["test"][i]), mean=round(100 * sum(r["test"][i]) / 3, 2),
                                  train=pc(r["train"][i]), test_by_rate={str(lr): pc(r["test"][k]) for k, lr in enumerate(r["lrs"])}, source=f"results_nset/{job}"))
    return dict(arm=arm, params=USED[arm], cells=sorted(cells, key=lambda c: (c["rerun"], c["N"])))


def bench(dirn):
    rows = {}
    for f in sorted((CON / dirn).glob("*/*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r.get("budget"), r["arm"], r["task"], r.get("N"))] = (r, str(f.relative_to(CON)))
    return rows


def comparison():
    B2, B3 = bench("results2"), bench("results3")
    out = {}
    for arm, label in (("c1", "join (one layer of 2-simplicial attention + MLP)"), ("sa4", "Transformer, 4 layers, width 44"),
                       ("sa_big", "Transformer, 4 layers, width 128"), ("newton_mlp", "Newton + MLP head (unfixed, rounds 2-3)")):
        cells = []
        for N in SIZES:
            hit = B3.get((None, arm, "set", N)) or (B2.get((None, arm, "set", None)) if N == 32768 else None)
            if hit:
                r, src = hit
                i = pick(r)
                cells.append(dict(N=N, lr=r["lrs"][i], test=pc(r["test"][i]), mean=round(100 * sum(r["test"][i]) / 3, 2), train=pc(r["train"][i]), source=src))
        out[arm] = dict(label=label, params=hit[0]["params"], cells=cells)
    m27 = []
    for d in ("q-m27a", "q-m27b"):
        for r in jl(CON / "results8" / d / "fits.jsonl"):
            i = pick(r)
            m27.append(dict(arm=r["arm"], N=r["stages"][-1][2], lr=r["lrs"][i], out8=pc(r["out8"][i]), out8_mean=round(100 * sum(r["out8"][i]) / 3, 2),
                            in8=pc(r["in8"][i]), in16=pc(r["in16"][i]), out16=pc(r["out16"][i]), params=r["params"], source=f"results8/{d}"))
    r4 = [json.loads(l) for l in (CON / "results4/w2/newton_bare.jsonl").read_text().splitlines() if json.loads(l)["task"] == "set"][0]
    return dict(fig5a=out, m27=m27, newton_bare_32k_run1=dict(note="run 1 of the unfixed rank-64 32k cell (rates 1e-4..3e-3); no character data",
                                                                    lrs=r4["lrs"], test=[pc(t) for t in r4["test"]], train=[pc(t) for t in r4["train"]],
                                                                    val=[pc(t) for t in r4["val"]], params=USED["newton_bare"], source="results4/w2/newton_bare.jsonl"))


def main():
    rs = runs()
    rule = [dict(rerun=r["rerun"], lr=r["lr"], seed=r["seed"], params=r["params"], **{k: r[k] for k in ("in8", "out8", "full8", "in16", "out16", "train", "found", "share")})
            for r in rs if r["kind"] == "rule" and r["arm"] == "newton_oscale" and r["picked"]]
    real = [r for r in rs if "smoke" not in r["role"]]
    data = dict(
        meta=dict(pick_rule="rate with the highest validation accuracy summed over seeds 0-2", test_set="1,024 new deals (tasks.generate seed 20000)",
                  rule_sets="bench8.deals: in8 own cards seed 20000, out8 the 54 held-out cards seed 20001, in16/out16 16-card tables, full8 all 81 cards",
                  params=USED, comparison_params=dict(c1=62018, sa4=65606, sa_big=534914, newton_mlp=74530),
                  hardware="Modal A10G, jax 0.8.0"),
        fixed_curve=picked_curve("newton_oscale", {"a": ["nset-oscale-curve-a", "nset-oscale-p1-a"], "b": ["nset-oscale-curve-b", "nset-oscale-p1-b"]}),
        m27_new_cards=dict(arm="newton_oscale", m=27, N=55296, rows=rule),
        runs=rs,
        random_phase_baseline=dict(definition="mean channel share with phases replaced by independent uniform phases, gates kept, 200 draws per run",
                                   mean_range=[min(r["baseline_mean"] for r in real), max(r["baseline_mean"] for r in real)],
                                   max_draw_range=[min(r["baseline_max"] for r in real), max(r["baseline_max"] for r in real)],
                                   untrained_share_range=[min(r["untrained_share"] for r in real), max(r["untrained_share"] for r in real)]),
        comparison=comparison())
    cost = CON / "results_nset" / "cost-1" / "cost.jsonl"
    if cost.exists():
        data["cost"] = dict(rows=jl(cost), source="results_nset/cost-1/cost.jsonl", setup="newton_paper/cost_bench.py: one layer, forward and backward, A10G, 16,384 tokens per batch")
    OUT.write_text(json.dumps(data, indent=1))
    print("wrote", OUT, len(rs), "run rows")


if __name__ == "__main__":
    main()
