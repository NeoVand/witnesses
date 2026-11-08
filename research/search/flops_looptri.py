"""Floating-point operations of the looped triangle-attention pair layer (search_train.pair_stack, tri='attn'), per
question, counted from its einsums (multiply-add = 2 FLOPs; softmax counted as 5 FLOPs per score). python flops_looptri.py"""
import json
from pathlib import Path
D, H = 16, 4
def per_run(N):
    lin = 5 * D * D * N * N            # a, b, g, v1, v2
    scores = D * N ** 3                 # q . k over channels, for every (u, k, v)
    tri = 2 * D * N ** 3                # v1(z_uk) * v2(z_kv), weighted, summed over k
    mlp = 2 * D * 2 * D * N * N         # f1, f2
    macs = lin + scores + tri + mlp
    return 2 * macs + 5 * H * N ** 3
out = {}
for V in (49, 125, 250):
    N = V + 1                           # name slots 0..V
    out[str(V)] = {str(K): per_run(N) * K for K in (8, 12, 16)}
(Path(__file__).resolve().parent / 'results' / 'flops_looptri.json').write_text(json.dumps(out, indent=1))
print(json.dumps({V: {K: f'{f / 1e9:.2f} GFLOP' for K, f in d.items()} for V, d in out.items()}))
