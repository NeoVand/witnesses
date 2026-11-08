"""Launch the exponential Newton runs on SET (bench_exp.py) on the witnesses-set app (modal_app.py). Usage:
spawn_exp.py main|diag|main6|split. Results go to results/<job>/; jobs ending in -a and -b are the two runs of one job
(separate containers, identical code and inputs). Job ids are never reused.
main:  four clocks, the rule fit (27 cards in hand, 55,296 deals) and the curve cells at 2,048, 8,192 and 32,768 deals.
diag:  where the four-clock models' errors fall (diag_exp.py, next to the saved weights; no training).
main6: six clocks, the rule fit and the 8,192-deal cell, trained in the pair form.
split: six clocks, the rule fit on seeds 3-5 with the rate picked on held-out cards in hand."""
import hashlib, json, sys
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
files = {n: HERE.parent / n for n in ("bench2.py", "bench8.py", "operators.py", "tasks.py", "hoa_polybench.py", "groups.py")} | {n: HERE / n for n in ("bench_exp.py", "diag_exp.py")}
sources = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}
G3 = [3e-4, 1e-3, 3e-3]
RULE = dict(lrs=[1e-3, 3e-3], m=27, T=8, N=55296, updates=24000, test_sizes=[8, 16, 20], tag="q")
stages = {
    "main": {},
    "diag": {"exp-diag": dict(script="diag_exp.py")},
    "split": {f"exp8-rule-{r}": dict(clocks=6, train_form="pairs", rule=[dict(RULE, seeds=[3, 4, 5], split=True, tag="split")]) for r in "ab"},
    "main6": {f"exp6-{k}-{r}": dict(clocks=6, train_form="pairs", **v) for r in "ab" for k, v in (("rule", dict(rule=[RULE])), ("curve8k", dict(cells=[[8192, 12000, G3]])))},
}
for r in "ab":
    stages["main"][f"exp-rule-{r}"] = dict(rule=[RULE])
    stages["main"][f"exp-curve-{r}"] = dict(cells=[[n, 12000, G3] for n in (2048, 8192, 32768)])
if __name__ == "__main__":
    run = modal.Function.from_name("witnesses-set", "run")
    jobs = stages[sys.argv[1]]
    calls = {k: run.spawn(dict(dict(script="bench_exp.py", root="connected-expnewton", id=k, sources=sources), **v)).object_id for k, v in jobs.items()}
    print("spawned", json.dumps(calls, indent=1))
