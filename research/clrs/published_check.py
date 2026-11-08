"""Compare DeepMind's published CLRS-30 test splits (CLRS30_v1.0.0.tar.gz from storage.googleapis.com/dm-clrs, 85.6 MB,
unpacked) with the test splits gen_data.py regenerated (data_local/<task>/test.npz).
Both are packed by gen_data.pack, so the comparison is array by array. The file stores the graphs in another order
(tfds shuffles records when it writes them), so graphs are matched by their adjacency matrix first; then every saved
run's test predictions (every K) are reordered and scored on the published graphs with clrs.evaluate, and compared with
the scores on our own split (rows.json). Writes published_check.json.
Needs dm-clrs 2.0.3 and TensorFlow, and every run folder's rows.json and preds (from the Modal volume).
python -I published_check.py <folder holding clrs_dataset/> <data_local>"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_data
import clrs


class Feedback:  # what gen_data.pack and evaluate read: .features.inputs and .outputs, lists of clrs.DataPoint
    def __init__(self, inputs, outputs):
        self.features = type('F', (), dict(inputs=inputs))(); self.outputs = outputs


def published(folder, task):
    """Read <folder>/clrs_dataset/<task>_test/1.0.0 with TensorFlow's record reader and tf.train.Example (the tfds
    loader fails under protobuf 7). Shapes come from the file's features.json; DataPoints as dataset._preprocess makes
    them (stage, location and type from specs.SPECS). Hints are not read."""
    import tensorflow as tf
    d = Path(folder) / 'clrs_dataset' / f'{task}_test' / '1.0.0'
    feats = json.loads((d / 'features.json').read_text())['featuresDict']['features']
    rows = {}
    for raw in tf.data.TFRecordDataset(sorted(str(f) for f in d.glob('*.tfrecord-*'))):
        ex = tf.train.Example.FromString(raw.numpy())
        for k, spec in feats.items():
            if k.startswith('hint_') or k == 'lengths':
                continue
            f = ex.features.feature[k]
            shape = [int(x) for x in spec['tensor']['shape'].get('dimensions', [])]
            rows.setdefault(k, []).append(np.asarray(getattr(f, f.WhichOneof('kind')).value).reshape(shape))
    spec = clrs.SPECS[task]
    dp = lambda k, name: clrs.DataPoint(name=name, location=spec[name][1], type_=spec[name][2], data=np.stack(rows[k]))
    inputs = [dp(k, k[6:]) for k in sorted(rows) if k.startswith('input_')]
    outputs = [dp(k, k[7:]) for k in sorted(rows) if k.startswith('output_')]
    return Feedback(inputs, outputs), len(rows[next(iter(rows))])


def main(folder, data_local):
    import hashlib
    import score
    out, rescored = {}, {}
    key = lambda d, i: hashlib.sha256(np.ascontiguousarray(d['adj'][i]).tobytes()).hexdigest()
    for task in gen_data.TASKS:
        fb, num = published(folder, task)
        theirs = gen_data.pack(fb, 64)
        ours = dict(np.load(Path(data_local) / task / 'test.npz'))
        manifest = json.loads((HERE / 'results' / 'data' / task / 'test.json').read_text())
        assert gen_data.digest(ours) == manifest['sha256'], (task, 'data_local differs from the split the runs saw')
        where = {key(ours, i): i for i in range(ours['n'].shape[0])}
        perm = [where.get(key(theirs, i), -1) for i in range(num)]  # published graph i is our graph perm[i]
        r = dict(num=int(num), ours_num=int(ours['n'].shape[0]), keys_equal=sorted(theirs) == sorted(ours),
                 in_order=perm == list(range(num)), found=sum(q >= 0 for q in perm), perm=perm)
        r['permutation'] = r['found'] == num == r['ours_num'] and len(set(perm)) == num
        diffs = {}
        if r['permutation']:
            for k in sorted(set(theirs) & set(ours)):
                a, b = theirs[k], ours[k][perm]
                if a.shape != b.shape:
                    diffs[k] = dict(shape_theirs=list(a.shape), shape_ours=list(b.shape))
                elif a.dtype.kind == 'f':
                    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
                    if d > 0:
                        diffs[k] = dict(max_abs=d)
                elif not np.array_equal(a, b):
                    diffs[k] = dict(graphs_differing=int((a != b).reshape(num, -1).any(1).sum()))
        r['diffs'] = diffs
        r['identical_after_reorder'] = r['permutation'] and r['keys_equal'] and not diffs
        out[task] = r
        print(task, 'same graphs, reordered' if r['identical_after_reorder'] else 'DIFFERS', json.dumps(diffs)[:300], flush=True)
        if not r['permutation']:
            continue
        names = [dp.name for dp in fb.outputs]
        for run in sorted((HERE / 'results').iterdir()):
            if not (run / 'rows.json').exists():
                continue
            for row in json.loads((run / 'rows.json').read_text()):
                if row['task'] != task:
                    continue
                pz = np.load(run / f'preds_s{row["seed"]}.npz')
                kmax = pz[f'test/{names[0]}'].shape[0]
                pub = [float(score.evaluate(fb, {nm: pz[f'test/{nm}'][K][perm] for nm in names})) for K in range(kmax)]
                rescored[f'{run.name}/s{row["seed"]}'] = dict(task=task, published=pub, ours=row['test'],
                                                               max_abs_diff=max(abs(a - b) for a, b in zip(pub, row['test'])))
    worst = max(v['max_abs_diff'] for v in rescored.values())
    (HERE / 'published_check.json').write_text(json.dumps(dict(tasks=out, rescored=rescored, runs=len(rescored), worst_abs_diff=worst), indent=1))
    print('same graphs on every task:', all(r['identical_after_reorder'] for r in out.values()))
    print(f'runs rescored on the published file: {len(rescored)}; largest score change at any K: {worst:.2e}')


if __name__ == '__main__':
    main(*sys.argv[1:])
