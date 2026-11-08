"""Saparov et al.'s own test-set code, loaded from their train.py at run time (not copied into this repo).

Their train.py imports torch and a model file at the top, so it cannot be imported directly. This parses it and
executes only the graph and test-set functions, so our test sets are theirs: 1,000s of held-out examples per lookahead
from `generate_eval_data`. python l2s_eval.py <their train.py> runs a self-check.
"""
import ast, sys

NEEDED = ('Node', 'lookahead_depth', 'generate_graph', 'get_descendants', 'generate_graph_with_lookahead', 'compute_paths',
          'generate_example', 'generate_eval_data')


def load(train_py):
    tree = ast.parse(open(train_py).read())
    keep = [n for n in tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NEEDED)
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'RESERVED_INDICES' for t in n.targets))]
    missing = set(NEEDED) - {getattr(n, 'name', None) for n in keep}
    assert not missing, missing
    ns = {}
    exec('from random import sample, randrange, choice, shuffle, seed, getstate, setstate, Random\nimport numpy as np\n', ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), train_py, 'exec'), ns)
    return ns


def test_sets(ns, max_input_size, per_lookahead, rng_seed=0, distance_from_start=-1):
    """{lookahead: (inputs, outputs)} for every lookahead their training run reserves, plus None (no fixed lookahead)."""
    import random, numpy as np
    out = {}
    top = ((max_input_size - 5) // 3 - 1) // 2
    for L in list(range(1, top + 1)) + [None]:
        random.seed(rng_seed); np.random.seed(rng_seed)
        out[L] = ns['generate_eval_data'](max_input_size, min_path_length=2, distance_from_start=distance_from_start, distance_from_end=-1,
                                          lookahead_steps=L, num_paths_at_fork=None, num_samples=per_lookahead, max_prefix_vertices=None)
    return out


if __name__ == '__main__':
    ns = load(sys.argv[1])
    sets = test_sets(ns, 3 * 9 + 5, 50)
    for L, (x, y) in sets.items():
        print(L, x.shape, y[:5])
