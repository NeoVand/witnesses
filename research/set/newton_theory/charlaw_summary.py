"""Summary of the characters law (charlaw.json): the oracle curve against 1/2 + k/80, the two-moment Gaussian law, and the
trained runs of the paper (newton_data.json, nine uncorrected rank-64 runs at 32,768 deals) against the oracle.

python -I charlaw_summary.py charlaw.json charlaw_summary.json   (numpy, scipy)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm

d = json.load(open(sys.argv[1]))
rows = [r for r in d["sampled"] if r["family"] == "random"]
ks = sorted({r["k"] for r in rows})
oracle = {k: float(np.mean([r["linear"] if "linear" in r else r["sum"] for r in rows if r["k"] == k])) for k in ks}
spread = {k: float(np.ptp([r["sum"] for r in rows if r["k"] == k])) for k in ks}
dev = {k: oracle[k] - (0.5 + k / 80) for k in ks}
n0, n1 = d["notes"]["label0"], d["notes"]["label1"]
mu0, mu1, V0, V1, C1 = n0["mean_per_channel"], n1["mean_per_channel"], n0["var"], n1["var"], n1["cov"]


def gauss(k):
    if k == 0:
        return 0.5
    s0 = np.sqrt(max(V0 * k * (40 - k) / 39, 1e-12)); s1 = np.sqrt(k * V1 + k * (k - 1) * C1)
    t = np.linspace(k * mu0 - 5 * s1, k * mu1 + 5 * s1, 4001)
    return float((0.5 * norm.cdf((t - k * mu0) / s0) + 0.5 * norm.cdf((k * mu1 - t) / s1)).max())


g = {k: gauss(k) for k in ks}
here = Path(sys.argv[1]).resolve().parent
data = json.load(open(here.parent / "newton_paper" / "newton_data.json"))
tr = [(r["found"], r["test"] / 100) for r in data["runs"] if r.get("arm") == "newton_bare" and r.get("N") == 32768
      and r.get("lr") in (3e-4, 1e-3, 3e-3) and r.get("job") == "nset-p1"]
F = np.array([f for f, _ in tr]); A = np.array([a for _, a in tr])
line = 0.5 + F / 80; orc = np.array([oracle[int(f)] for f in F])
out = dict(
    max_dev_above_line=max(dev.values()), k_at_max=max(dev, key=dev.get),
    oracle={k: round(oracle[k], 4) for k in ks}, gauss={k: round(g[k], 4) for k in ks},
    max_gauss_error=max(abs(g[k] - oracle[k]) for k in ks), max_spread_random=max(spread.values()),
    trained=[dict(found=int(f), acc=float(a), line=float(l), oracle=float(o)) for f, a, l, o in zip(F, A, line, orc)],
    trained_minus_line=dict(mean=float((A - line).mean()), sd=float((A - line).std(ddof=1)), max_abs=float(abs(A - line).max())),
    trained_minus_oracle=dict(mean=float((A - orc).mean()), sd=float((A - orc).std(ddof=1)), min=float((A - orc).min()), max=float((A - orc).max())),
    r_line=float(np.corrcoef(A, line)[0, 1]), r_oracle=float(np.corrcoef(A, orc)[0, 1]))
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("oracle", "gauss")}, indent=1))
print("k: oracle / gauss / line:", {k: (round(oracle[k], 3), round(g[k], 3), 0.5 + k / 80) for k in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 38, 39)})
