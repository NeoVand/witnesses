"""The benchmark's own 125-vertex test questions for the browser demo, in one Modal CPU job.

Saparov et al.'s train.py is cloned inside the image at the commit modal_search.py pins (github.com/asaparov/
learning_to_search, 2c3f32b); only its pure-Python test-set functions run (l2s_eval.load), as in every test.

Fixed rule: for each lookahead in export_orjoin.LOOKS_125 (1, 2, 4, 8, 16, 24, 32, 48, 62) and for the random graphs
(lookahead None), the first 4 questions of the benchmark's test set at max input 3 * 125 + 5 = 380 tokens, generated
exactly as l2s_eval.test_sets does (random and numpy seeded 777 before each lookahead, the seed export_demo.py used for
the 49-vertex demo set). Nothing is filtered by the model's answers.

    modal run modal_export125.py
writes results/export125/graphs_v125_raw.json; `export_orjoin.py questions-from results/export125/graphs_v125_raw.json`
turns it into static/model/graphs_v125.json.
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('witnesses-search-export')
image = (modal.Image.debian_slim(python_version='3.12').apt_install('git').pip_install('numpy==2.3.4')
         .run_commands('git clone https://github.com/asaparov/learning_to_search.git /root/l2s_repo && cd /root/l2s_repo && '
                       'git checkout 2c3f32bd8073bba2144e112748bd6e8216bd7399')
         .add_local_file(HERE / 'l2s_eval.py', '/root/search/l2s_eval.py'))
V, PER, SEED = 125, 4, 777
LOOKS = [1, 2, 4, 8, 16, 24, 32, 48, 62]


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0)
def export():
    import random, sys, time
    import numpy as np
    sys.path.insert(0, '/root/search')
    import l2s_eval
    t0 = time.time()
    ns = l2s_eval.load('/root/l2s_repo/train.py')
    n = 3 * V + 5
    top = ((n - 5) // 3 - 1) // 2
    assert top == 62
    sets = []
    for L in LOOKS + [None]:
        random.seed(SEED)
        np.random.seed(SEED)  # as l2s_eval.test_sets, which reseeds before every lookahead
        xs, ys = ns['generate_eval_data'](n, min_path_length=2, distance_from_start=-1, distance_from_end=-1, lookahead_steps=L,
                                          num_paths_at_fork=None, num_samples=PER, max_prefix_vertices=None)
        sets.append(dict(lookahead=L, inputs=np.asarray(xs).astype(int).tolist(), outputs=np.asarray(ys).astype(int).tolist()))
    return dict(vertices=V, tokens=n, per=PER, seed=SEED, commit='2c3f32bd8073bba2144e112748bd6e8216bd7399', sets=sets,
                seconds=round(time.time() - t0, 1))


@app.local_entrypoint()
def main():
    out = export.remote()
    dest = HERE / 'results' / 'export125'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'graphs_v125_raw.json').write_text(json.dumps(out))
    print('wrote', dest / 'graphs_v125_raw.json', sum(len(s['inputs']) for s in out['sets']), 'questions in', out['seconds'], 's')
