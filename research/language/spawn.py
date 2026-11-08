"""python spawn.py <job list>. Spawns the jobs of one list on the deployed witnesses-language app (modal_app.py).

Lists, in the order they were run:
    language_data   FineWeb-Edu tokens and the held-out sets (CPU; run once before language)
    language        the 51M-parameter language models: baseline, join every second block, two more layers; seeds 0, 1
    chess_data      the Lichess puzzle database as move-prediction examples (CPU; run once before the chess lists)
    chess24         8-layer attention and joins replacing attention in blocks 3 and 6, all data, seeds 0 and 1
    chess25         the join beside attention (connected output) at all data, 10% and 1%; attention and joins
                    replacing attention at 10% and 1%
    chess26         the join beside attention with plain output, and with centred keys, at all data and 1%
Each job writes /vol/results/<id>/; download with `modal volume get witnesses-language results/<id> results/`.
Job ids are never reused.
"""
import hashlib, json, sys
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
NAMES = ['model.py', 'cubics.py', 'worker.py', 'hoa_lm.py', 'chess_train.py', 'chess_train2.py']
APP = 'witnesses-language'


def language_data():
    return [dict(script='prepare.py', cpu=True, id='lm-data', timeout=7000)]


def language():
    """Rotary positions, 12,208 steps of 64 x 256 tokens, peak rate 1e-3 (worker default), seeds 0 and 1."""
    R = dict(pos='rope')
    arms = dict(r=('gpt', {}), every2=('cubic', dict(branch_every=2)), r10=('gpt', dict(layers=10)))
    return [dict(id=f'p27-{name}-s{seed}', arm=arm, config={**R, **cfg}, seed=seed, steps=12208, diag_every=2000, timeout=4200) for name, (arm, cfg) in arms.items() for seed in (0, 1)]


def chess_data():
    return [dict(script='chess_data.py', cpu=True, id='c24-data', procs=16, timeout=7000)]


C24 = dict(sa8=['sa'] * 8, cub8=['sa', 'sa', 'cubic', 'sa', 'sa', 'cubic', 'sa', 'sa'])


def chess24():
    """Two seeds per arm, 20,000 updates (chess_train.py)."""
    return [dict(script='chess_train.py', id=f'c24-{a}-s{sd}', data='c24-data', model=dict(d=256, H=8, layers=L, remat=True), steps=20000, batch=512, lr=5e-4, seeds=[sd], log=2000, save=True, timeout=13000)
            for a, L in C24.items() for sd in (0, 1)]


HYB8C = dict(layers=['sa', 'sa', 'hyb', 'sa', 'sa', 'hyb', 'sa', 'sa'], connected=True, diag=True, impl='cudnn')


def chess25():
    """The join beside attention against 8-layer attention at 100%, 10% and 1% of the data (chess_train2.py)."""
    arms = dict(sa8=dict(layers=C24['sa8']), cub8=dict(layers=C24['cub8']), hyb8c=HYB8C)
    plan = [('full', None, 20000, ('hyb8c',)), ('p10', 1418000, 10000, ('sa8', 'hyb8c')), ('p1', 141781, 4000, ('sa8', 'cub8', 'hyb8c'))]
    return [dict(script='chess_train2.py', id=f'c25-{size}-{a}', data='c24-data', model=dict(d=256, H=8, remat=True, **arms[a]), steps=st, batch=512, lr=5e-4, seeds=[0],
                 log=1000, save=True, timeout=13000, **({'train_examples': n} if n else {})) for size, n, st, names in plan for a in names]


def chess26():
    """Centred keys in the join beside attention, and the uncentred plain-output join as control (chess_train2.py)."""
    base = dict(layers=['sa', 'sa', 'hyb', 'sa', 'sa', 'hyb', 'sa', 'sa'], diag=True, impl='cudnn')
    arms = dict(hyb8z=dict(base, center=True), hyb8r=dict(base))
    plan = [('full', None, 20000), ('p1', 141781, 4000)]
    return [dict(script='chess_train2.py', id=f'c26-{size}-{a}', data='c24-data', model=dict(d=256, H=8, remat=True, **m), steps=st, batch=512, lr=5e-4, seeds=[0],
                 log=1000, save=True, timeout=13000, **({'train_examples': n} if n else {})) for size, n, st in plan for a, m in arms.items()]


def main():
    jobs = globals()[sys.argv[1]]()
    sources = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in NAMES}
    run, run_cpu = modal.Function.from_name(APP, 'run'), modal.Function.from_name(APP, 'run_cpu')
    calls = {}
    for job in jobs:
        calls[job['id']] = (run_cpu.spawn(job) if job.get('cpu') else run.spawn(dict(job, sources=sources))).object_id
    print(json.dumps(calls, indent=1))
    print('spawned', len(calls))


if __name__ == '__main__':
    main()
