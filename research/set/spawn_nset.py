"""Launch the Newton runs on SET (bench_nset.py) by stage. Usage: spawn_nset.py smoke|p1|smoke256|r1|fix|fixrest|count|mix.
Results go to results_nset/<job>/. Job ids are never reused; jobs ending in -a and -b are the two runs of one job
(separate containers, identical code and inputs)."""
import hashlib, json, sys
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
names = ("bench_nset.py", "newton_chars.py", "bench8.py", "bench4.py", "bench2.py", "operators.py", "tasks.py", "hoa_polybench.py", "groups.py")
sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}
G3, G2 = [3e-4, 1e-3, 3e-3], [3e-4, 1e-3, 3e-3, 1e-2]   # the join's rate grids: curve cells (bench3) and the 32k cell (bench2)
U = {2048: 12000, 8192: 12000, 32768: 12000, 131072: 24000, 524288: 48000}
A = "newton_bare"
stages = {
    # rank 64, no step correction: a smoke job (training loss only), then the 32k cell
    "smoke": {"nset-smoke": dict(cells=[[A, 2048, 1000, [3e-3]]], rule=[dict(arm=A, lrs=[3e-3], stages=[[27, 8, 2048, 500]], tag="smoke")])},
    "p1": {"nset-p1": dict(cells=[[A, 32768, U[32768], G2]])},
}
# rank 256, no step correction: smoke, then the 32k cell twice
B = "newton_r256"
stages["smoke256"] = {"nset-r256-smoke": dict(cells=[[B, 2048, 1000, [3e-3]]])}
stages["r1"] = {f"nset-r256-p1-{r_}": dict(cells=[[B, 32768, U[32768], G2]]) for r_ in "ab"}
# rank 64 with the read-out step correction (newton_oscale): the 32k cell
stages["fix"] = {f"nset-oscale-p1-{r_}": dict(cells=[["newton_oscale", 32768, U[32768], G2]]) for r_ in "ab"}
# newton_oscale: the other curve cells (2k, 8k, 128k) and the rule fit with 27 cards in hand
stages["fixrest"] = {}
for r_ in "ab":
    stages["fixrest"][f"nset-oscale-curve-{r_}"] = dict(cells=[["newton_oscale", n, U[n], G3] for n in (2048, 8192, 131072)])
    stages["fixrest"][f"nset-oscale-rule-{r_}"] = dict(rule=[dict(arm="newton_oscale", lrs=[1e-3, 3e-3], stages=[[27, 8, 55296, 24000]], tag="q")])
# count form (newton_count), rule fit only
stages["count"] = {f"nset-count-rule-{r_}": dict(rule=[dict(arm="newton_count", lrs=[1e-3, 3e-3], stages=[[27, 8, 55296, 24000]], tag="q")]) for r_ in "ab"}
# newton_oscale trained on a mix of 8- and 12-card tables
stages["mix"] = {f"nset-mix-{r_}": dict(mixed=[dict(arm="newton_oscale", lrs=[1e-3, 3e-3], m=27, sizes=[8, 12], n_each=27648, updates=24000,
                                                  test_sizes=[8, 12, 16, 20], tag="mix8-12")]) for r_ in "ab"}
if __name__ == "__main__":
    run = modal.Function.from_name("witnesses-set", "run")
    jobs = stages[sys.argv[1]]
    calls = {k: run.spawn(dict(script="bench_nset.py", root="connected-nset", id=k, sources=sources, **v)).object_id for k, v in jobs.items()}
    print("spawned", json.dumps(calls, indent=1))
