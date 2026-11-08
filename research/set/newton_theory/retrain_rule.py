"""Retrain, on CPU, the rule fit whose 16-card score the paper reports (Newton, read-out step correction).

Same code, data, seeds, rates and updates as the A10G job nset-oscale-rule-{a,b} (bench_nset.rule with
spec {"arm": "newton_oscale", "lrs": [1e-3, 3e-3], "stages": [[27, 8, 55296, 24000]]}); only the device
differs. The A10G weights were not downloaded from the Modal volume, so table16.py and table16_model.py analyse
these CPU retrains (params/rule_newton_oscale_cpu_24000.npz) and report their own measured 16-card accuracy.

python -I retrain_rule.py [updates]   (from newton_theory/; jax, optax, numpy; about 26 min on a laptop CPU)
"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np, jax
import bench_nset, bench8  # bench_nset patches bench4 / bench8 with the newton_oscale arm

updates = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
spec = {"arm": "newton_oscale", "tag": "q", "lrs": [0.001, 0.003], "stages": [[27, 8, 55296, updates]]}
t0 = time.time()
row, params = bench8.fit(spec)
row["device"] = str(jax.devices()[0])
row["wall_seconds"] = time.time() - t0
tag = f"rule_newton_oscale_cpu_{updates}"
np.savez_compressed(HERE / "params" / f"{tag}.npz", **bench_nset.flat(params))
(HERE / "params" / f"{tag}.json").write_text(json.dumps(row, indent=1))
print(json.dumps({k: row[k] for k in ("in8", "in16", "out8", "out16", "wall_seconds")}))
