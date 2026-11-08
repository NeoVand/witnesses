"""Random-graph accuracy of a saved pairn model, split by the step of the path that is asked .
The benchmark's random-graph test set with distance_from_start=-1 asks every step; with 1 it asks only the first step,
which is what the first-step models in Table 1 were scored on.
python step_eval.py <params.npz> <their train.py> <vertices> <n> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as j
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st, l2s_eval


def main(params, train_py, V, num, out_path):
    cfg = dict(kind='pairn', emb=32, D=16, layers=8, r=8, bf16=False, remat=False)
    n, pad, vocab = 3 * V + 5, V + 3, V + 5
    p = st.init(jax.random.PRNGKey(0), cfg, vocab, n)
    P = dict(np.load(params))
    flat, tree = jax.tree_util.tree_flatten_with_path(p)
    p = jax.tree_util.tree_unflatten(tree, [j.array(P[jax.tree_util.keystr(k)]) for k, _ in flat])
    ns = l2s_eval.load(train_py)
    fwd = jax.jit(lambda p, x: st.forward(p, cfg, x, pad)[1])
    import random
    res = {}
    for dfs in (1, -1):
        random.seed(777); np.random.seed(777)
        x, y = ns['generate_eval_data'](n, min_path_length=2, distance_from_start=dfs, distance_from_end=-1, lookahead_steps=None,
                                        num_paths_at_fork=None, num_samples=num, max_prefix_vertices=None)
        pred = np.concatenate([np.asarray(fwd(p, j.array(x[i:i + 64]))).argmax(-1) for i in range(0, len(x), 64)])
        ok = pred == y
        step = np.array([int((row != pad).sum()) - (row.tolist().index(V + 1) if V + 1 in row.tolist() else 0) - 1 for row in x])
        step = np.array([len(row) - row.tolist().index(V + 1) - 1 for row in x])  # path vertices after the P marker
        by = {int(s): [float(ok[step == s].mean()), int((step == s).sum())] for s in np.unique(step)}
        res[str(dfs)] = dict(acc=float(ok.mean()), n=int(len(ok)), by_step=by)
        print(dfs, res[str(dfs)], flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
