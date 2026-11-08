"""Launch the cards-in-hand fits (bench8.py): m cards in hand (the first m of a fixed shuffle) at several data sizes,
for the join c1 and the Transformers sa4 and sa_big; results8/<job>/fits.jsonl. Job ids are never reused."""
import hashlib, json
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
names = ("bench8.py", "bench2.py", "operators.py", "tasks.py", "hoa_polybench.py", "groups.py")
sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}
SA, CU = [3e-4, 1e-3], [1e-3, 3e-3]
U = lambda N: 12000 if N <= 32768 else 24000 if N <= 131072 else 48000
LR = dict(c1=CU, sa4=SA, sa_big=SA)


def cell(m, N, arms=("c1", "sa4", "sa_big")):
    return [dict(arm=a, lrs=LR[a], stages=[[m, 8, N, U(N)]], save=a != "sa_big", tag="q") for a in arms]


per = lambda m, *cards: [f for c in cards for f in cell(m, m * c)]
jobs = {
    "q-m1": per(1, 512, 2048, 8192),
    "q-m3": per(3, 512, 2048, 8192),
    "q-m9": per(9, 512, 2048, 8192),
    "q-m27a": per(27, 512, 2048),
    "q-m27b": per(27, 8192),
    "q-m81a": per(81, 512),
    "q-m81b": per(81, 2048),
    "q-gap": [f for m in (2, 4, 6) for f in cell(m, 32768)],
}

if __name__ == "__main__":
    run = modal.Function.from_name("witnesses-set", "run")
    calls = {k: run.spawn(dict(script="bench8.py", root="connected8", id=k, fits=v, sources=sources)).object_id for k, v in jobs.items()}
    print("spawned", json.dumps(calls, indent=1))
