"""Launch the 32,768-deal cell of the SET curves (bench2.py, every task of the bench): the join c1, the Transformers
sa4 (66k) and sa_big (535k), and Newton with the join's MLP head. Worker ids match results2/. Rates 3e-4 to 1e-2."""
import hashlib, json
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
groups = {"w0": ["c1"], "w4": ["sa4"], "w5": ["sa_big"], "w7": ["newton_mlp"]}
names = ("bench2.py", "operators.py", "tasks.py", "hoa_polybench.py", "groups.py")
sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}
run = modal.Function.from_name("witnesses-set", "run")
tasks = ["typed", "chronological", "composition", "who", "match", "set"]
calls = {w: run.spawn(dict(script="bench2.py", root="connected2", id=w, arms=g, tasks=tasks, lrs=[3e-4, 1e-3, 3e-3, 1e-2], sources=sources)).object_id for w, g in groups.items()}
print("spawned", json.dumps(calls))
