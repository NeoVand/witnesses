"""CLRS-30 data from DeepMind's own library (dm-clrs 2.0.3), saved as padded numpy arrays without hints.

Splits, as the library builds them:
- test: build_sampler(alg, seed=CLRS30['test']['seed'] = 3, num_samples=32 * multiplier, length=64), default sampler
  arguments (p = 0.5), in generation order; this is clrs/_src/dataset.py CLRSDataset._create_data. multiplier is 1
  for every task here (specs.CLRS_30_ALGS_SETTINGS).
- val (in distribution): the same with the 'val' row of CLRS30 (seed 2, 32 samples, 16 nodes).
- train: the library's training protocol in clrs/examples/run.py (create_samplers): sizes 4, 7, 11, 13 and 16, edge
  probability p drawn per graph from 0.1, 0.2, ..., 0.9 (halved, 0.05 to 0.45, for articulation_points, bridges and
  mst_kruskal, as run.py does), positions randomised (done at training time, see clrs_train.to_x). run.py samples on the fly; we pre-generate a pool of `per_size` graphs per size in chunks of
  8,000 with seeds 10^7 * (task index + 1) + 1000 * size + chunk.
Arrays per split file: n [N], pos [N, M], A [N, M, M] float32, adj [N, M, M] uint8, s [N] (source index, if the task
has one), outputs as integer indices padded with -1 (pi, topo, scc_id [N, M]; Pi [N, M, M]; topo_head [N]); masks as
0/1 padded with -1 (is_cut [N, M]; is_bridge, in_mst [N, M, M]; the library itself marks is_bridge on non-edges -1). M is the largest
size in the file. sha256 of the arrays goes to the manifest so other machines can check they generate the same data.
fresh:SEED:NUM draws NUM graphs as the test split is drawn (64 nodes, default sampler arguments) at another seed, saved
as fresh<SEED>.npz (the reported fresh draws: seeds 20,261,107 and 20,261,109, 128 graphs each).
Run with dm-clrs 2.0.3 installed: python3 -I gen_data.py <out_dir> <task> <split|train:SIZE|fresh:SEED:NUM> [per_size]
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

TASKS = ['bfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall', 'mst_prim', 'topological_sort',
         'dfs', 'strongly_connected_components', 'articulation_points', 'bridges', 'mst_kruskal']  # order fixes the training-pool seeds
SIZES = [4, 7, 11, 13, 16]
P_TRAIN = tuple(0.1 + 0.1 * i for i in range(9))
HALF_P = ('articulation_points', 'bridges', 'mst_kruskal')  # run.py: "a lower connection probability ... otherwise trajectories are very long"
CHUNK = 8000


def pack(fb, M):
    ins = {d.name: np.asarray(d.data) for d in fb.features.inputs}
    outs = {d.name: np.asarray(d.data) for d in fb.outputs}
    N, n = ins['pos'].shape
    padn = lambda x, v: np.pad(x, [(0, 0)] + [(0, M - n)] * (x.ndim - 1), constant_values=v)
    out = dict(n=np.full(N, n, np.int16), pos=padn(ins['pos'].astype(np.float32), 0), A=padn(ins['A'].astype(np.float32), 0),
               adj=padn(ins['adj'].astype(np.uint8), 0))
    if 's' in ins:
        out['s'] = ins['s'].argmax(-1).astype(np.int16)
    for name, d in outs.items():
        if name == 'topo_head':
            out[name] = d.argmax(-1).astype(np.int16)
        else:
            out[name] = padn(np.rint(d).astype(np.int16), -1)
    return out


def sample(alg, num, length, seed, **kw):
    import clrs
    sampler, _ = clrs.build_sampler(alg, num_samples=num, length=length, seed=seed, **kw)
    return sampler.next(None)  # the whole dataset in generation order, as dataset.py does


def digest(d):
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode()); h.update(np.ascontiguousarray(d[k]).tobytes())
    return h.hexdigest()


def make(task, what, per_size=80000):
    import clrs
    if what.startswith('fresh:'):  # the test split's sampler and settings (64 nodes, default arguments), another seed
        _, seed, num = what.split(':')
        c = clrs.CLRS30['test']
        return pack(sample(task, int(num), c['length'], int(seed)), c['length']), dict(seed=int(seed), num=int(num), length=c['length'])
    if what in ('test', 'val'):
        c = clrs.CLRS30[what]
        num = c['num_samples'] * clrs.CLRS_30_ALGS_SETTINGS[task]['num_samples_multiplier']
        return pack(sample(task, num, c['length'], c['seed']), c['length']), dict(seed=c['seed'], num=num, length=c['length'])
    L = int(what.split(':')[1])
    parts, seeds = [], []
    p = tuple(np.array(P_TRAIN) / 2) if task in HALF_P else P_TRAIN
    for ci in range((per_size + CHUNK - 1) // CHUNK):
        seed = 10 ** 7 * (TASKS.index(task) + 1) + 1000 * L + ci
        parts.append(pack(sample(task, min(CHUNK, per_size - ci * CHUNK), L, seed, p=p), 16))
        seeds.append(seed)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, dict(seeds=seeds, num=per_size, length=L, p=[float(q) for q in p])


def main(out_dir, task, what, per_size=80000):
    t0 = time.time()
    d, meta = make(task, what, int(per_size))
    name = f"fresh{what.split(':')[1]}" if what.startswith('fresh:') else what.replace(':', '_L')  # fresh<seed>
    dest = Path(out_dir) / task
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f'{name}.npz', **d)
    import clrs
    meta |= dict(task=task, split=what, sha256=digest(d), shapes={k: list(v.shape) for k, v in d.items()}, seconds=round(time.time() - t0, 1),
                 clrs_version=getattr(clrs, '__version__', '?'), numpy=np.__version__)
    (dest / f'{name}.json').write_text(json.dumps(meta))
    return meta


if __name__ == '__main__':
    print(json.dumps(main(*sys.argv[1:])))
