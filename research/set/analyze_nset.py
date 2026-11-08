"""Score the Newton runs in results_nset/ against the registered predictions (Supplementary
Table stbl-register2): P1-P4 (rank 64, no step correction), R1-R2 (rank 256), F1 and P2'-P4' (read-out step correction),
the count form (C1) and training on 8- and 12-card tables (X1-X3). Each job ran twice (reruns a and b); predictions
are graded on rerun a. python3 -I analyze_nset.py   (standard library only)"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = HERE / "results_nset"
SIZES = [2048, 8192, 32768, 131072, 524288]


def jl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def pick(r):
    return max(range(len(r["lrs"])), key=lambda n: sum(r["val"][n]))


def pct(v):
    return [round(100 * x, 1) for x in v]


def smoke():
    for r in jl(R / "nset-smoke" / "cells.jsonl") + jl(R / "nset-smoke" / "fits.jsonl"):
        print("smoke", r.get("N", r.get("stages")), "train loss", r["final_loss"])


def main():
    if (R / "nset-smoke").exists():
        smoke()
    cells = {r["N"]: r for d in ("nset-p1", "nset-curve") for r in jl(R / d / "cells.jsonl")}
    exact, curve = [], {}
    for N in SIZES:
        if N not in cells:
            continue
        r = cells[N]
        i = pick(r)
        t = r["test"][i]
        curve[N] = sum(t) / 3
        print(f"N={N:>6} lr={r['lrs'][i]:g} test {pct(t)} mean {100 * curve[N]:.1f}  train {pct(r['train'][i])}  loss {[round(x, 4) for x in r['final_loss'][i]]}")
        print("        all rates: val", [round(100 * sum(v) / 3, 1) for v in r["val"]], "test", [round(100 * sum(v) / 3, 1) for v in r["test"]])
        for s in range(3):
            c = r["chars"]["trained"][i][s]
            if t[s] >= .99:
                exact.append((f"N={N} s{s}", c, r["chars"]["init"][s]))
    if 32768 in cells:
        r = cells[32768]
        t = r["test"][pick(r)]
        print("P1", "PASS" if sum(x >= .97 for x in t) == 3 else "FAIL", pct(t))
    if len(curve) < len(SIZES):
        print("P2 NOT RUN (stop rule after P1)" if 32768 in curve else "P2 NOT RUN")
    else:
        first = next((N for N in SIZES if N in curve and curve[N] >= .9), None)
        print("P2", "PASS" if first and first <= 32768 else "FAIL", "first 90% at", first, "| at 8,192:", round(100 * curve.get(8192, 0), 1))
    if not (R / "nset-rule").exists():
        print("P3 NOT RUN (stop rule after P1)")
    for r in jl(R / "nset-rule" / "fits.jsonl"):
        i = pick(r)
        o, n = r["out8"][i], r["in8"][i]
        print("rule", r["stages"], "lr", r["lrs"][i], "out8", pct(o), "mean", round(100 * sum(o) / 3, 1), "in8", pct(n), "full8", pct(r["full8"][i]),
              "loss", [round(x, 5) for x in r["final_loss"][-1][i]])
        print("        all rates: val", [round(100 * sum(v) / 3, 1) for v in r["val"]], "out8", [round(100 * sum(v) / 3, 1) for v in r["out8"]])
        print("P3", "PASS" if sum(o) / 3 >= .9 else "FAIL")
        for s in range(3):
            if o[s] >= .99 and n[s] >= .99:
                exact.append((f"rule s{s}", r["chars"]["trained"][i][s], r["chars"]["init"][s]))
    if exact:
        for name, c, c0 in exact:
            T, H = c["table"], c["hand"]
            print(f"  {name}: table share {T['share']:.3f} (baseline {T['base_mean']:.3f}, max {T['base_max']:.3f}), found {T['found']}/40;"
                  f" hand share {H['share']:.3f}, found {H['found']}; untrained table {c0['table']['share']:.3f}")
        m = sum(c["table"]["share"] for _, c, _ in exact) / len(exact)
        b = sum(c["table"]["base_mean"] for _, c, _ in exact) / len(exact)
        ok = m >= 3 * b and all(c["table"]["share"] > c["table"]["base_max"] for _, c, _ in exact)
        print("P4", "PASS" if ok else "FAIL", f"mean share {m:.3f} over {len(exact)} exact runs, baseline {b:.3f}")
    elif len(cells) < len(SIZES):
        print("P4 NOT RUN (stop rule after P1; no exact run among the cells that ran)")
    elif cells:
        print("P4 VOID (no exact run)")


def rank256():
    """R1 and R2: rank 256 at 32k, graded on rerun a, rerun b beside it."""
    for run in "ab":
        for r in jl(R / f"nset-r256-p1-{run}" / "cells.jsonl"):
            i = pick(r)
            t, found = r["test"][i], [r["chars"]["trained"][i][s]["table"]["found"] for s in range(3)]
            print(f"rank 256 rerun {run} ({r['params']:,} params): lr {r['lrs'][i]:g} test {pct(t)} found {found}")
            if run == "a":
                print("R1", "PASS" if sum(x >= .97 for x in t) == 3 else "FAIL", "| R2", "PASS" if min(found) >= 38 else "FAIL")


def step_correction():
    """F1, P2', P3', P4' for newton_oscale: rerun a graded, rerun b beside it; the 32k cell is nset-oscale-p1."""
    P4, verdict = [], {}
    for run in "ab":
        cells = {r["N"]: r for d in (f"nset-oscale-p1-{run}", f"nset-oscale-curve-{run}") for r in jl(R / d / "cells.jsonl")}
        means = {}
        for N in sorted(cells):
            r = cells[N]
            i = pick(r)
            t = r["test"][i]
            means[N] = sum(t) / 3
            print(f"oscale {run} N={N:>6} ({r['params']:,} params) lr {r['lrs'][i]:g} test {pct(t)} mean {100 * means[N]:.1f} train {pct(r['train'][i])}")
            for s_ in range(3):
                if t[s_] >= .97:
                    P4.append((f"{run} N={N} s{s_}", r["chars"]["trained"][i][s_]["table"]))
        first = next((N for N in sorted(means) if means[N] >= .9), None)
        verdict[f"P2' {run}"] = ("PASS" if first and first <= 32768 else "FAIL", "first 90% at", first, "| at 8k:", "PASS" if means.get(8192, 0) >= .9 else "FAIL")
        for r in jl(R / f"nset-oscale-rule-{run}" / "fits.jsonl"):
            i = pick(r)
            o, n = r["out8"][i], r["in8"][i]
            print(f"oscale {run} rule m=27 ({r['params']:,} params) lr {r['lrs'][i]:g} out8 {pct(o)} mean {100 * sum(o) / 3:.1f} in8 {pct(n)} full8 {pct(r['full8'][i])}")
            verdict[f"P3' {run}"] = "PASS" if sum(o) / 3 >= .9 else "FAIL"
            for s_ in range(3):
                if o[s_] >= .97 and n[s_] >= .97:
                    P4.append((f"{run} rule s{s_}", r["chars"]["trained"][i][s_]["table"]))
    f1 = [x for run in "ab" for r in jl(R / f"nset-oscale-p1-{run}" / "cells.jsonl") for x in r["test"][pick(r)]]
    if f1:
        print("F1", "PASS" if len(f1) == 6 and min(f1) >= .97 else "FAIL", f"({sum(x >= .97 for x in f1)} of {len(f1)} seed-runs at least 97%)")
    for k, v in verdict.items():
        print(k, v)
    for name, T in P4:
        print(f"  P4' {name}: share {T['share']:.3f} baseline max {T['base_max']:.3f} mean {T['base_mean']:.3f} found {T['found']}")
    print("P4'", ("PASS" if all(T["share"] > T["base_max"] for _, T in P4) else "FAIL") if P4 else "VOID", f"({len(P4)} runs)")


def count_form():
    """C1: the count form (newton_count), 27 cards in hand, 8- and 16-card tables, own (in) and new (out) cards."""
    for run in "ab":
        for r in jl(R / f"nset-count-rule-{run}" / "fits.jsonl"):
            i = pick(r)
            print(f"count form {run} ({r['params']:,} params) lr {r['lrs'][i]:g} " + " ".join(f"{k} {pct(r[k][i])}" for k in ("in8", "out8", "in16", "out16"))
                  + f" train {pct(r['train'][-1][i])} found {[r['chars']['trained'][i][s]['table']['found'] for s in range(3)]}")
            if run == "a":
                ok16 = all(min(r[k][i]) >= .95 for k in ("in16", "out16"))
                print("C1", "PASS" if ok16 and sum(r["in8"][i]) / 3 >= .9953 - .01 and sum(r["out8"][i]) / 3 >= .9980 - .01 else "FAIL")


def two_sizes():
    """X1-X3: newton_oscale trained on 8- and 12-card tables; tests at 8, 12, 16 and 20 cards, own and new cards."""
    for run in "ab":
        for r in jl(R / f"nset-mix-{run}" / "fits.jsonl"):
            i = pick(r)
            print(f"two sizes {run} ({r['params']:,} params) lr {r['lrs'][i]:g}")
            for T in r["sizes"] + [n for n in (16, 20) if n not in r["sizes"]]:
                print(f"  {T} cards: own {pct(r[f'in{T}'][i])} new {pct(r[f'out{T}'][i])}")
            print("  found", [r["chars"]["trained"][i][s]["table"]["found"] for s in range(3)])
            if run == "a":
                low = lambda T: min(min(r[f"in{T}"][i]), min(r[f"out{T}"][i]))
                m8 = (sum(r["in8"][i]) / 3, sum(r["out8"][i]) / 3)
                print("X1", "PASS" if low(16) >= .95 else "FAIL", f"(lowest {100 * low(16):.1f})", "| X2", "PASS" if low(20) >= .9 else "FAIL",
                      f"(lowest {100 * low(20):.1f})", "| X3", "PASS" if m8[0] >= .9953 - .01 and m8[1] >= .9980 - .01 else "FAIL")


if __name__ == "__main__":
    main()
    rank256()
    step_correction()
    count_form()
    two_sizes()
