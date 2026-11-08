"""Launch the layer-cost benchmark (newton_paper/cost_bench.py) on the witnesses-set app (A10G); results_nset/cost-1."""
import hashlib, json
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent.parent
names = ("newton_paper/cost_bench.py", "bench2.py", "operators.py", "tasks.py", "hoa_polybench.py", "groups.py")
sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names}
if __name__ == "__main__":
    run = modal.Function.from_name("witnesses-set", "run")
    job = dict(script="newton_paper/cost_bench.py", root="connected-nset", id="cost-1", sources=sources,
               methods=["softmax", "newton", "join"], ns=[9, 64, 256, 1024, 4096])
    print("spawned", json.dumps({"cost-1": run.spawn(job).object_id}))
