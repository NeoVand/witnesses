"""Spawn the CLRS-30 jobs on the deployed witnesses-clrs app (modal_clrs.py). python spawn_clrs.py <group>; call ids go
to calls_<group>.json. Groups:
- data: the library's val and test splits and the training pools of all twelve graph tasks (CPU);
  fresh10, fresh13: the fresh 64-node test draws at seeds 20,261,107 and 20,261,109 (128 graphs per task).
- c1, c3: the looped triangle-attention layer at width 32 (13,312 parameters), the width-32 baseline of the OR-join
  designs (c1 on seven tasks, c3 on the other five).
- c7, c8, c9: the three width-32 designs: the OR join (agg = 'newton'), half OR join and half power mean (agg = 'mix'),
  and the same with the learned log n shift (logn = True, 13,342 parameters). c10: c9 on new seeds 3 to 5, with the
  fresh draw 20,261,107 predicted; c10eval: the saved c1/c3 (c10tri) and c9 (c10c9) parameters scored on that draw.
- c13: c9's block at width 128 (186,232 parameters), 80,000 steps, all twelve tasks, seeds 0 to 2, with the fresh draw
  20,261,109 predicted. c14 (dfs) and c17 (strongly_connected_components): c13 with K drawn from 4 to 24 in training
  and predictions up to K = 34. c15: c13's mst_prim job plus the Kirchhoff read-out; c16: c15 plus the learned log n
  shift on the conductances."""
import json, sys
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
APP = 'witnesses-clrs'
TASKS = ['bfs', 'bellman_ford', 'dijkstra', 'dag_shortest_paths', 'floyd_warshall', 'mst_prim', 'topological_sort']
C3_TASKS = ['dfs', 'strongly_connected_components', 'articulation_points', 'bridges', 'mst_kruskal']
MODEL = dict(D=32, heads=4, mlp=64, wmlp=32)
MAIN = dict(data_dir='/vol/clrs/data', model=MODEL, batch=32, steps=40000, lr=1e-3, warmup=1000, final_lr_frac=.01, wd=.01,
            train_iters=[4, 5, 6, 7, 8], eval_kmax=16, check_every=250, test_every=5000, eval_batch=32,
            max_train_seconds=2700, timeout=6600)


def data(tasks=TASKS + C3_TASKS):
    """The library's splits and training pools (gen_data.py; the pools' seeds depend only on the task, not on the group)."""
    return [dict(fn='gen', args=(t, w, 80000)) for t in tasks for w in ['test', 'val'] + [f'train:{L}' for L in (4, 7, 11, 13, 16)]]


def c1():
    """The triangle-attention layer, width 32, 40,000 steps, K from 4 to 8, on seven tasks, seeds 0 to 2."""
    return [dict(fn='train', job=dict(MAIN, id=f'c1-{t}-s{sd}', task=t, seeds=[sd])) for t in TASKS for sd in (0, 1, 2)]


def c3():
    """c1's model, protocol, steps, seeds and K rule on the five remaining CLRS-30 graph tasks."""
    return [dict(fn='train', job=dict(MAIN, id=f'c3-{t}-s{sd}', task=t, seeds=[sd])) for t in C3_TASKS for sd in (0, 1, 2)]


def c7():
    """c1/c3 unchanged except the OR-join block (agg = 'newton', MLP width 76 for equal parameters)."""
    return [dict(fn='train', job=dict(MAIN, id=f'c7-{t}-s{sd}', task=t, seeds=[sd], model=dict(MODEL, agg='newton', mlp=76)))
            for t in TASKS + C3_TASKS for sd in (0, 1, 2)]


def c8():
    """c7 unchanged except agg = 'mix' (half OR join, half power mean), MLP width 78."""
    return [dict(fn='train', job=dict(MAIN, id=f'c8-{t}-s{sd}', task=t, seeds=[sd], model=dict(MODEL, agg='mix', mlp=78)))
            for t in TASKS + C3_TASKS for sd in (0, 1, 2)]


def c9():
    """c8 unchanged except logn = True (input sigmoids shifted by kappa log n, kappa learned per channel)."""
    return [dict(fn='train', job=dict(MAIN, id=f'c9-{t}-s{sd}', task=t, seeds=[sd], model=dict(MODEL, agg='mix', mlp=78, logn=True)))
            for t in TASKS + C3_TASKS for sd in (0, 1, 2)]


FRESH = 'fresh20261107'  # 128 graphs per task drawn as the test split, seed 20,261,107


def fresh10():
    """The fresh 64-node test graphs at seed 20,261,107 (CPU)."""
    return [dict(fn='gen', args=(t, 'fresh:20261107:128', 80000)) for t in TASKS + C3_TASKS]


def c10():
    """c9's exact job on seeds 3, 4, 5, with the fresh graphs predicted at the end."""
    return [dict(fn='train', job=dict(jb['job'], id=jb['job']['id'].replace('c9-', 'c10-').rsplit('-s', 1)[0] + f'-s{sd}', seeds=[sd],
                                      extra_splits=[FRESH])) for jb in c9() if jb['job']['seeds'] == [0] for sd in (3, 4, 5)]


def c10eval():
    """Saved parameters scored on the fresh graphs, no training: the triangle layer (c1/c3) and c9."""
    jobs = []
    for t in TASKS + C3_TASKS:
        run = 'c1' if t in TASKS else 'c3'
        jobs.append(dict(fn='train', job=dict(MAIN, id=f'c10tri-{t}', task=t, seeds=[0, 1, 2], extra_splits=[FRESH],
                                              params_from=f'/vol/clrs/runs/{run}-{t}-s{{seed}}/params_s{{seed}}.npz')))
        jobs.append(dict(fn='train', job=dict(MAIN, id=f'c10c9-{t}', task=t, seeds=[0, 1, 2], extra_splits=[FRESH],
                                              model=dict(MODEL, agg='mix', mlp=78, logn=True),
                                              params_from=f'/vol/clrs/runs/c9-{t}-s{{seed}}/params_s{{seed}}.npz')))
    return jobs


W128 = dict(MODEL, D=128, mlp=312, agg='mix', logn=True)  # c9's block at width 128, MLP scaled by 4 (78 -> 312)
TEST3 = 'fresh20261109'  # 128 graphs per task drawn as the test split, seed 20,261,109


def fresh13():
    """The fresh 64-node test graphs at seed 20,261,109 for all twelve tasks (CPU)."""
    return [dict(fn='gen', args=(t, 'fresh:20261109:128', 80000)) for t in TASKS + C3_TASKS]


def c13():
    """The width-128 block, 80,000 steps (warm-up 2,000, cosine to 1%), all twelve tasks, seeds 0 to 2."""
    return [dict(fn='train', job=dict(MAIN, id=f'c13-{t}-s{sd}', task=t, seeds=[sd], model=W128, steps=80000, warmup=2000,
                                      extra_splits=[TEST3], eval_batch=16, max_train_seconds=6600, timeout=7100))
            for t in TASKS + C3_TASKS for sd in (0, 1, 2)]


def c14():
    """c13's dfs job with K from 4 to 24 in training and predictions up to K = 34."""
    return [dict(fn='train', job=dict(jb['job'], id=jb['job']['id'].replace('c13-', 'c14-'), train_iters=list(range(4, 25)), eval_kmax=34,
                                      test_every=80000, max_train_seconds=12000, timeout=13000))
            for jb in c13() if jb['job']['task'] == 'dfs']


K15 = dict(W128, kirch=4, keps=1e-4)  # c13's block plus the Kirchhoff read-out (4 heads, eps 1e-4)


def c15():
    """c13's mst_prim job unchanged plus the Kirchhoff read-out (model keys kirch, keps), seeds 0 to 2."""
    return [dict(fn='train', job=dict(jb['job'], id=jb['job']['id'].replace('c13-', 'c15-'), model=K15))
            for jb in c13() if jb['job']['task'] == 'mst_prim']


K16 = dict(K15, klogn=True)  # c15 plus the learned log n shift on the conductance logits


def c16():
    """c15's job exactly, plus model key klogn (kappa log n on the conductance logits), seeds 0 to 2."""
    return [dict(fn='train', job=dict(jb['job'], id=jb['job']['id'].replace('c15-', 'c16-'), model=K16)) for jb in c15()]


def c17():
    """c14 on strongly_connected_components: c13's job with K from 4 to 24 in training, predictions up to K = 34."""
    return [dict(fn='train', job=dict(jb['job'], id=jb['job']['id'].replace('c13-', 'c17-'), train_iters=list(range(4, 25)), eval_kmax=34,
                                      test_every=80000, max_train_seconds=12000, timeout=13000))
            for jb in c13() if jb['job']['task'] == 'strongly_connected_components']


GROUPS = {'data': data, 'fresh10': fresh10, 'fresh13': fresh13, 'c1': c1, 'c3': c3, 'c7': c7, 'c8': c8, 'c9': c9, 'c10': c10,
          'c10eval': c10eval, 'c13': c13, 'c14': c14, 'c15': c15, 'c16': c16, 'c17': c17}


def main():
    group = sys.argv[1]
    jobs = GROUPS[group]()
    fns = {n: modal.Function.from_name(APP, n) for n in ('gen', 'train')}
    calls = {}
    for jb in jobs:
        if jb['fn'] == 'gen':
            calls['gen-' + '-'.join(map(str, jb['args'])).replace(':', '')] = fns['gen'].spawn(*jb['args']).object_id
        else:
            calls[jb['job']['id']] = fns['train'].spawn(jb['job']).object_id
    (HERE / f'calls_{group}.json').write_text(json.dumps(calls, indent=1))
    print('spawned', len(calls))


if __name__ == '__main__':
    main()
