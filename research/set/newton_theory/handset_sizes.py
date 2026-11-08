"""Table-size check on the hand-set Newton layer (newton_paper/construction.py): with mean features (the 1/C(n,2) factor) and the construction's fixed offset
eps = 1/240, the threshold sits at N = C(n,2) eps, so the layer is exact for C(n,2) < 240 (n <= 22 table cards) and calls
every one-set deal negative from n = 23 on. The same layer with the 1/C(n,2) factor replaced by the constant 1/C(8,2)
(count form) and the offset moved to the trivial channel's count is exact at every table size tested.

python -I handset_sizes.py handset_sizes.json   (jax, numpy)
"""
import json, math, sys, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "newton_paper")]
import numpy as np, jax, jax.numpy as j
import bench4, bench8
import hoa_polybench as legacy
from bench2 import D, R
import construction as C
from table16 import set_counts


def main(dest):
    S = bench8.query_set(27)
    out = {}
    p = C.hand_params("trivial")
    for T in (8, 16, 20, 22, 23, 24, 26):   # balanced deals need negatives; beyond ~26 cards they are rare
        x, y = bench8.deals(50000 + T, 2048, S, T)
        N = set_counts(x, T)
        pred = np.concatenate([np.asarray(C.apply(p, j.asarray(x[i:i + 1024])).argmax(-1)) for i in range(0, len(x), 1024)])
        acc = float((pred == y).mean())
        one = N == 1
        # count form, exact: y* = kappa/81 * (sum_s 2 Re chi_s(q) e_2 + C(n,2) - 81/2) / C(8,2)
        h = j.tanh(j.asarray(x) @ p["input"]); L = p["legacy"]
        th = 2. * (h @ L["th"]["w"] + L["th"]["b"]); g = jax.nn.sigmoid(h @ L["g"]["w"] + L["g"]["b"])
        phi = g * j.exp(1j * th); P = phi.at[:, 0].set(0)
        p1, p2 = P.sum(1, keepdims=True) - P, (P * P).sum(1, keepdims=True) - P * P
        e2q = np.asarray((phi * (p1 * p1 - p2) / 2)[:, -1])           # phi_q e_2, a count, per channel
        val = (e2q[:, 0].real + 2 * e2q[:, 1:41].real.sum(-1)) / 81    # trivial channel + 40 characters = N exactly
        count_form_acc = float(((val > 0.5) == (y == 1)).mean())
        out[T] = dict(pairs=math.comb(T, 2), threshold_in_sets=math.comb(T, 2) * C.EPS, handset_mean_form_acc=acc,
                      one_set_deals=int(one.sum()), one_set_called_positive=float((pred[one] == 1).mean()) if one.any() else None,
                      count_form_acc=count_form_acc, count_form_max_error=float(np.abs(val - N).max()))
        print(T, out[T], flush=True)
    json.dump(out, open(dest, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
