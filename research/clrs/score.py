"""Score saved predictions with DeepMind's own scorer, clrs.evaluate (dm-clrs 2.0.3), in an environment with the library.

For each task the val and test splits are regenerated here with the library exactly as gen_data.py made them on Modal,
and their sha256 is checked against the manifest of the data the runs trained on (results/data/<task>/<split>.json), so the
predictions are scored against the very questions the model saw. Predictions (indices after each run K) come
from results/<run id>/preds_s<seed>.npz (with the run's rows.json, as the training job writes them). Writes
scores_<prefix>.json and prints the table.
python -I score.py <results dir> [run id prefix, e.g. c13-]
"""
import json, math, sys
from pathlib import Path
import numpy as np
import clrs
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # -I drops the script directory; gen_data is ours
import gen_data

K_RULE = lambda n: math.ceil(math.log2(n)) + 2  # 6 at 16 nodes, 8 at 64


def library_split(task, split):
    c = clrs.CLRS30[split]
    num = c['num_samples'] * clrs.CLRS_30_ALGS_SETTINGS[task]['num_samples_multiplier']
    fb = gen_data.sample(task, num, c['length'], c['seed'])
    return fb, gen_data.digest(gen_data.pack(fb, c['length']))


def evaluate(fb, pred_by_name):
    preds = {}
    for dp in fb.outputs:
        a = pred_by_name[dp.name]
        if dp.type_ == clrs.Type.MASK_ONE:
            a = np.eye(dp.data.shape[-1])[a]
        elif dp.type_ == clrs.Type.MASK:  # our 0/1 decisions (logit > 0); _mask_fn thresholds at 0.5
            a = a.astype(np.float32)
        preds[dp.name] = clrs.DataPoint(name=dp.name, location=dp.location, type_=dp.type_, data=a)
    return clrs.evaluate(fb.outputs, preds)['score']


def main(results, prefix='c1-'):
    results = Path(results)
    cache, out = {}, {}
    for run in sorted(p for p in results.iterdir() if p.name.startswith(prefix) and (p / 'rows.json').exists()):
        rows = json.loads((run / 'rows.json').read_text())
        for r in rows:
            task, seed = r['task'], r['seed']
            pz = np.load(run / f'preds_s{seed}.npz')
            res = dict(task=task, seed=seed, run=run.name, steps_done=r.get('steps_done'), seconds=r['seconds'], params=r['params'])
            for split in ('val', 'test'):
                if (task, split) not in cache:
                    fb, sha = library_split(task, split)
                    manifest = json.loads((results / 'data' / task / f'{split}.json').read_text())
                    assert manifest['sha256'] == sha, (task, split, 'data on Modal differ from the library here')
                    cache[task, split] = fb
                fb = cache[task, split]
                names = [dp.name for dp in fb.outputs]
                kmax = pz[f'{split}/{names[0]}'].shape[0]
                res[split] = [float(evaluate(fb, {nm: pz[f'{split}/{nm}'][K] for nm in names})) for K in range(kmax)]
                ours = r[split]  # the training job's own copy of the scorer
                assert max(abs(a - b) for a, b in zip(res[split], ours)) < 1e-6, (run.name, split, 'our scorer disagrees with clrs.evaluate')
            out[f'{run.name}/s{seed}'] = res
    (HERE / f'scores_{prefix.rstrip("-")}.json').write_text(json.dumps(out, indent=1))
    by_task = {}
    for res in out.values():
        by_task.setdefault(res['task'], []).append(res)
    print('task | OOD test at K=%d (mean, min-max) | best K (mean) | val at K=%d | seeds' % (K_RULE(64), K_RULE(16)))
    for task, rs in by_task.items():
        t = np.array([r['test'] for r in rs]); v = np.array([r['val'] for r in rs])
        kr, kv = K_RULE(64) - 1, K_RULE(16) - 1
        best = int(t.mean(0).argmax())
        print(f'{task} | {t[:, kr].mean():.4f} ({t[:, kr].min():.4f}-{t[:, kr].max():.4f}) | K={best + 1}: {t[:, best].mean():.4f} | '
              f'{v[:, kv].mean():.4f} ({v[:, kv].min():.4f}-{v[:, kv].max():.4f}) | {len(rs)}')
    return out


if __name__ == '__main__':
    main(*sys.argv[1:])
