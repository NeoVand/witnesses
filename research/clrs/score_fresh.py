"""Score predictions on a fresh draw of 64-node graphs with clrs.evaluate (dm-clrs 2.0.3).

The fresh split is regenerated here exactly as gen_data.py made it on Modal ('fresh:<seed>:128') and its sha256 is
checked against the manifest (results/data/<task>/fresh_L<seed>_L128.json). Predictions after each run K come from
results/<run>/preds_s<seed>.npz ('fresh<seed>/<output>'). Each score must match the job's own
scorer to 1e-6. For checkpoint-scoring runs (params_from) the job's standard-test scores must also match the stored
scores of the run it loaded (rows.json of that run) to 1e-6 at every K.
FRESH_SEED=<seed> python -I score_fresh.py <results dir> <run prefix> [<run prefix> ...]
(seed 20261107, the default: runs c10, c10tri, c10c9; seed 20261109: runs c13 to c17)
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_data
from score import evaluate

import os
SEED, NUM = int(os.environ.get('FRESH_SEED', 20261107)), 128
SPLIT = f'fresh{SEED}'


def main(results, *prefixes):
    results = Path(results)
    cache, out = {}, {}
    for run in sorted(p for p in results.iterdir() if any(p.name.startswith(x) for x in prefixes) and (p / 'rows.json').exists()):
        for r in json.loads((run / 'rows.json').read_text()):
            task, seed = r['task'], r['seed']
            if task not in cache:
                fb = gen_data.sample(task, NUM, 64, SEED)
                sha = gen_data.digest(gen_data.pack(fb, 64))
                man = json.loads((results / 'data' / task / f'fresh_L{SEED}_L128.json').read_text())
                assert man['sha256'] == sha, (task, 'fresh graphs on Modal differ from the library here')
                cache[task] = fb
            fb = cache[task]
            pz = np.load(run / f'preds_s{seed}.npz')
            names = [dp.name for dp in fb.outputs]
            kmax = pz[f'{SPLIT}/{names[0]}'].shape[0]
            sc = [float(evaluate(fb, {nm: pz[f'{SPLIT}/{nm}'][K] for nm in names})) for K in range(kmax)]
            assert max(abs(a - b) for a, b in zip(sc, r[SPLIT])) < 1e-6, (run.name, 'our scorer disagrees with clrs.evaluate')
            res = dict(task=task, seed=seed, run=run.name, fresh=sc, test_job=r['test'], params=r['params'])
            if r.get('params_from'):
                src = r['params_from'].split('/runs/')[1].split('/')[0]
                stored = [x for x in json.loads((results / src / 'rows.json').read_text()) if x['seed'] == seed][0]
                res['source'] = src
                res['test_reproduces_source'] = max(abs(a - b) for a, b in zip(r['test'], stored['test']))
            out[f'{run.name}/s{seed}'] = res
            print(run.name, seed, 'fresh K=8', round(sc[7], 4), 'repro', res.get('test_reproduces_source'), flush=True)
    tag = '' if SEED == 20261107 else f'{SEED}_'
    (HERE / f'scores_fresh_{tag}{"_".join(x.rstrip("-") for x in prefixes)}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
