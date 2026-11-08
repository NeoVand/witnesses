"""Launch the other cells of the SET curves (bench3.py): 2,048 to 524,288 deals, 12,000 updates up to 32,768 deals,
24,000 at 131,072 and 48,000 at 524,288. Worker ids match results3/. Joins and Newton: rates 3e-4, 1e-3, 3e-3;
Transformers: 3e-4, 1e-3."""
import hashlib, json
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
K2, K8, K32, K128, K512 = 2048, 8192, 32768, 131072, 524288
U = {K2: 12000, K8: 12000, K32: 12000, K128: 24000, K512: 48000}
cell = lambda a, *ns: [[a, n, U[n]] for n in ns]
groups = {"w0": cell("c1", K2, K8, K128, K512), "w3": cell("newton_mlp", K2, K8, K128, K512), "w4": cell("sa4", K2, K8, K128), "w5": cell("sa4", K512),
          "w6": cell("sa_big", K2, K8), "w7": cell("sa_big", K128), "w8": cell("sa_big", K512)}
names = ("bench3.py", "bench2.py", "operators.py", "tasks.py", "hoa_polybench.py", "groups.py")
sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}
run = modal.Function.from_name("witnesses-set", "run")
calls = {w: run.spawn(dict(script="bench3.py", root="connected3", id=w, cells=g, tasks=["typed", "chronological", "match", "set"], lrs=[3e-4, 1e-3, 3e-3], sa_lrs=[3e-4, 1e-3], sources=sources)).object_id for w, g in groups.items()}
print("spawned", json.dumps(calls))
