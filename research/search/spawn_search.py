"""Spawn search-benchmark jobs on the deployed witnesses-search Modal app. python spawn_search.py <group>, where <group>
is one of the functions below; each returns the job specs (run id, model config, protocol) of one group of runs.
Results land on the witnesses-search volume under search/<run id>; spawned call ids are printed."""
import json, sys
from pathlib import Path
import modal
HERE = Path(__file__).resolve().parent
PAIR = dict(kind='pair', emb=32, D=32, layers=8, r=8)
SA16 = dict(kind='sa', emb=32, d=16, mlp=16, H=1, layers=8, readout='vocab')  # their size
SA256 = dict(kind='sa', emb=64, d=256, mlp=1024, H=4, layers=8, readout='pointer')


def pilot():
    """Engineering check: speed and memory at 31 and 49 vertices, 300 updates."""
    base = dict(batch=256, warmup=100, seeds=[0], max_examples=256 * 300, check_every=50, test_every=10 ** 9, per_lookahead=200, stop_on_converge=False, timeout=3400)
    return [dict(base, id=f's0-pilot-{name}-v{V}', vertices=V, model=m, lr=1e-3) for V in (31, 49) for name, m in (('pair', PAIR), ('sa16', SA16), ('sa256', SA256))]


def pilot2():
    """Speed after the channel-first bf16 rewrite: pair only, 31 and 49 vertices, 300 updates."""
    base = dict(batch=256, warmup=100, seeds=[0], max_examples=256 * 300, check_every=50, test_every=10 ** 9, per_lookahead=100, stop_on_converge=False, timeout=3400)
    return [dict(base, id=f's0-pilot2-pair-v{V}', vertices=V, model=PAIR, lr=1e-3) for V in (31, 49)]


def pilot3():
    """Speed of narrower pair states at 49 vertices, 300 updates."""
    base = dict(batch=256, warmup=100, seeds=[0], max_examples=256 * 300, check_every=50, test_every=10 ** 9, per_lookahead=100, stop_on_converge=False, timeout=3400)
    return [dict(base, id=f's0-pilot3-pair-D{D}-v49', vertices=49, model=dict(PAIR, D=D), lr=1e-3) for D in (16, 24)]


MAIN = dict(batch=256, warmup=2000, max_examples=5_000_000, check_every=500, test_every=5000, per_lookahead=1000, timeout=13000)
PAIR16 = dict(PAIR, D=16)


def stageA():
    """S1 stage A: pair at 31 and 41 vertices (3 seeds), Transformer controls at 41 and 49."""
    jobs = [dict(MAIN, id=f's1-pair16-v{V}-s{sd}', vertices=V, model=PAIR16, lr=1e-3, seeds=[sd]) for V in (31, 41) for sd in (0, 1, 2)]
    for V in (41, 49):
        jobs += [dict(MAIN, id=f's1-sa16-lr{tag}-v{V}', vertices=V, model=SA16, lr=lr, seeds=[0]) for tag, lr in (('1e-3', 1e-3), ('3e-4', 3e-4))]
        jobs.append(dict(MAIN, id=f's1-sa256-lr3e-4-v{V}', vertices=V, model=SA256, lr=3e-4, seeds=[0]))
    return jobs


def stageB():
    """S1 stage B: pair at 49 vertices, 3 seeds (only if S0 passed)."""
    return [dict(MAIN, id=f's1-pair16-v49-s{sd}', vertices=49, model=PAIR16, lr=1e-3, seeds=[sd], max_examples=10_000_000) for sd in (0, 1, 2)]


def control():
    """S1 control check: can our Transformer recipe learn small graphs at all?"""
    jobs = []
    for V in (15, 23):
        jobs.append(dict(MAIN, id=f's1c-sa16-lr1e-3-v{V}', vertices=V, model=SA16, lr=1e-3, seeds=[0], max_examples=10_000_000))
        jobs.append(dict(MAIN, id=f's1c-sa256-lr3e-4-v{V}', vertices=V, model=SA256, lr=3e-4, seeds=[0], max_examples=10_000_000))
    return jobs


def theirs_pilot():
    return [dict(id='s1t-pilot-v31', vertices=31, seed=1, max_examples=1_000_000, timeout=3400, fn='run_theirs')]


def theirs():
    """S1: their own Transformer training, unmodified, capped at 5M examples."""
    return [dict(id=f's1t-v{V}-s{sd}', vertices=V, seed=sd, max_examples=5_000_000, timeout=13000, fn='run_theirs') for V, seeds in ((31, (1,)), (41, (1, 2, 3)), (49, (1, 2, 3))) for sd in seeds]


def gap():
    """S1b: retrain two 31-vertex pair runs with parameters saved."""
    return [dict(MAIN, id=f's1b-pair16-v31-s{sd}', vertices=31, model=PAIR16, lr=1e-3, seeds=[sd], save=True) for sd in (1, 2)]


PAIRN = dict(kind='pairn', emb=32, D=16, layers=8, r=8)


def pilot4():
    """Speed of the name-level pair model at 49, 100 and 125 vertices, 300 updates."""
    base = dict(batch=256, warmup=100, seeds=[0], max_examples=256 * 300, check_every=50, test_every=10 ** 9, per_lookahead=50, stop_on_converge=False, timeout=3400)
    return [dict(base, id=f's2-pilot-pairn-v{V}', vertices=V, model=PAIRN, lr=1e-3) for V in (49, 100, 125)]


def stage2():
    """S2: name-level pair state at 49 to 125 vertices; their Transformer at 100 and 125."""
    jobs = [dict(MAIN, id=f's2-pairn-v{V}-s{sd}', vertices=V, model=PAIRN, lr=1e-3, seeds=[sd], max_examples=10_000_000, save=True,
                 per_lookahead=1000 if V <= 75 else 500) for V in (49, 75, 100, 125) for sd in (0, 1, 2)]
    jobs += [dict(id=f's2t-v{V}-s1', vertices=V, seed=1, max_examples=5_000_000, timeout=13000, fn='run_theirs') for V in (100, 125)]
    return jobs


SANN = dict(kind='sann', emb=32, D=16, r=8, d=64, mlp=128, H=4, layers=8)


def stage3():
    """S3: Transformer over pooled names, with and without the pair features as attention biases."""
    jobs = [dict(MAIN, id=f's3-sann-v{V}-s{sd}', vertices=V, model=SANN, lr=1e-3, seeds=[sd], max_examples=10_000_000, save=True,
                 per_lookahead=1000 if V <= 75 else 500) for V in (49, 125) for sd in (0, 1, 2)]
    jobs += [dict(MAIN, id=f's3-sannnb-v49-s{sd}', vertices=49, model=dict(SANN, nobias=True), lr=1e-3, seeds=[sd], max_examples=10_000_000) for sd in (0, 1, 2)]
    return jobs


def stage4():
    """S4: first-step variant, a longer and re-rated benchmark Transformer, triangle attention."""
    jobs = [dict(MAIN, id=f's4-pairn-first-v{V}-s{sd}', vertices=V, model=PAIRN, lr=1e-3, seeds=[sd], max_examples=10_000_000, first_step=True)
            for V in (31, 41, 49) for sd in (0, 1, 2)]
    jobs += [dict(id=f's4t-v{V}-lr{lr}', vertices=V, seed=1, lr=lr, max_examples=15_000_000, timeout=13500, fn='run_theirs')
             for V, lr in ((41, 1e-5), (49, 1e-5), (49, 1e-4))]
    jobs += [dict(MAIN, id=f's4-pairt-v{V}-s{sd}', vertices=V, model=dict(PAIRN, tri='attn'), lr=1e-3, seeds=[sd], max_examples=10_000_000, save=True,
                  per_lookahead=1000 if V <= 75 else 500) for V, sd in ((49, 0), (49, 1), (49, 2), (100, 0))]
    return jobs


def stage5():
    """S5: depth sweep and mixed random-graph training at 49 vertices."""
    jobs = [dict(MAIN, id=f's5-pairn-L{L}-v49-s{sd}', vertices=49, model=dict(PAIRN, layers=L), lr=1e-3, seeds=[sd], max_examples=5_000_000, save=True)
            for L in (3, 4, 5, 6, 7) for sd in (0, 1)]
    jobs += [dict(MAIN, id=f's5-pairn-mix-v49-s{sd}', vertices=49, model=PAIRN, lr=1e-3, seeds=[sd], max_examples=10_000_000, save=True, mix_random=0.5)
             for sd in (0, 1, 2)]
    return jobs


def stage6():
    """S6: controls (compute-matched benchmark Transformer, no offsets, first step)."""
    jobs = [dict(id=f's6t-v31-s{sd}', vertices=31, seed=sd, max_examples=15_728_640, max_seconds=1900, stop_on_converge=False, timeout=2300, fn='run_theirs')
            for sd in (1, 2, 3)]
    jobs += [dict(MAIN, id=f's6-pair16-norel-v31-s{sd}', vertices=31, model=dict(PAIR16, norel=True), lr=1e-3, seeds=[sd]) for sd in (0, 1, 2)]
    jobs += [dict(MAIN, id=f's6-pair16-first-v31-s{sd}', vertices=31, model=PAIR16, lr=1e-3, seeds=[sd], max_examples=2_500_000, first_step=True)
             for sd in (0, 1, 2)]
    return jobs


def stage7():
    """S7: depth L against lookahead 2^L at new sizes."""
    runs = [(49, 2), (100, 4), (100, 5), (100, 6)]
    return [dict(MAIN, id=f's7-pairn-L{L}-v{V}-s{sd}', vertices=V, model=dict(PAIRN, layers=L), lr=1e-3, seeds=[sd], max_examples=3_000_000,
                 per_lookahead=1000 if V <= 75 else 500) for V, L in runs for sd in (0, 1)]


def stage8():
    """S8: one pair layer, looped, trained at 49 vertices and run more times on graphs of up to 125."""
    base = dict(MAIN, script='loop_train.py', vertices=49, name_pool=128, lr=1e-3, max_examples=3_000_000, test_every=5000,
                eval_sizes=[49, 75, 100, 125], timeout=7000)
    jobs = [dict(base, id=f's8-loop-v49-s{sd}', seeds=[sd], model=dict(PAIRN, layers=1, loop=True, nopos=True),
                 train_iters=[5, 6, 7, 8], eval_iters=list(range(1, 17))) for sd in (0, 1, 2)]
    jobs += [dict(base, id=f's8-fixed8-v49-s{sd}', seeds=[sd], model=dict(PAIRN, nopos=True)) for sd in (0, 1, 2)]
    return jobs


def stage9():
    """S9: S8's looped layer with triangle attention over the middle vertex."""
    return [dict(j, id=j['id'].replace('s8-loop', 's9-looptri'), model=dict(j['model'], tri='attn'), eval_batch=32) for j in stage8() if 'loop' in j['id']]


def stage10():
    """S10: the benchmark's Transformer with its authors' relative-position option (Toeplitz attention)."""
    return [dict(id=f's10t-toep-v{V}-s{sd}', vertices=V, seed=sd, toeplitz=True, max_examples=5_242_880, timeout=4000, fn='run_theirs')
            for V in (31, 49) for sd in (1, 2, 3)]


def stage11():
    """S11: the looped triangle-attention layer with 256 names, run on graphs of up to 250 vertices."""
    return [dict(j, id=j['id'].replace('s9-looptri', 's11-looptri256'), name_pool=256, eval_sizes=[49, 125, 150, 200, 250],
                 per_lookahead_big=200, eval_batch=8, timeout=10000) for j in stage9()]


def stage12():
    """S12: the looped triangle-attention layer trained with half of its batches from random graphs."""
    return [dict(j, id=j['id'].replace('s9-looptri', 's12-looptrimix'), mix_random=0.5) for j in stage9()]


def stage13():
    """S13: the Transformer over names at learning rate 3e-4."""
    jobs = [dict(MAIN, id=f's13-sann-lr3e-4-v49-s{sd}', vertices=49, model=SANN, lr=3e-4, seeds=[sd], max_examples=10_000_000) for sd in (0, 1, 2)]
    jobs += [dict(MAIN, id=f's13-sann-lr3e-4-v125-s{sd}', vertices=125, model=SANN, lr=3e-4, seeds=[sd], max_examples=10_000_000,
                  per_lookahead=500) for sd in (0, 1)]
    return jobs


def stage14():
    """S14: the S2 model over names trained past convergence, to 3M examples."""
    return [dict(MAIN, id=f's14-pairn-long-v49-s{sd}', vertices=49, model=PAIRN, lr=1e-3, seeds=[sd], max_examples=3_000_000,
                 stop_on_converge=False, save=True) for sd in (0, 1, 2)]


def stage15():
    """S15: the 256-name looped triangle layer trained so that K runs answer lookaheads up to 2^K."""
    jobs = []
    for arm, mix in (('tight', 0.0), ('tightmix', 0.5)):
        for j in stage11():
            jobs.append(dict(j, id=j['id'].replace('s11-looptri256', f's15-{arm}256'), tight_k=True, mix_random=mix,
                             train_iters=[2, 3, 4, 5, 6, 7, 8]))
    return jobs


def stage16():
    """S16: degree-invariant pooling, and scalable softmax over the middle vertex."""
    jobs = []
    for arm, extra in (('pool', dict(pool_norm=True)), ('poolss', dict(pool_norm=True, ssmax=True))):
        for j in stage15():
            if 'tightmix' in j['id']:
                jobs.append(dict(j, id=j['id'].replace('s15-tightmix256', f's16-{arm}256'), model=dict(j['model'], **extra)))
    return jobs



def stage17():
    """S17: the Transformer over names (S3 sann), looped, against S9's looped pair layer."""
    jobs = []
    for sd in (0, 1, 2):
        lp = [j for j in stage8() if j['id'] == f's8-loop-v49-s{sd}'][0]
        fx = [j for j in stage8() if j['id'] == f's8-fixed8-v49-s{sd}'][0]
        jobs.append(dict(lp, id=f's17-sannloop-v49-s{sd}', model=dict(SANN, layers=1, loop=True, nopos=True), max_examples=6_000_000))
        jobs.append(dict(fx, id=f's17-sannfixed8-v49-s{sd}', model=dict(SANN, nopos=True), max_examples=6_000_000))
    return jobs

def stage20():
    """S20: S9's looped layer with the Newton-OR or the power mean over the middle vertex in place of
    triangle attention; everything else as S9."""
    jobs = []
    for agg in ('newton', 'pmean'):
        for j in stage9():
            jobs.append(dict(j, id=j['id'].replace('s9-looptri', f's20-loop{agg}'), model=dict(j['model'], tri=agg), eval_batch=64,
                             timeout=12000))
    return jobs


def smoke20():
    """S20 smoke: 300 updates of each S20 arm, scored at 49 vertices only."""
    return [dict(j, id=j['id'].replace('-v49-s0', '-smoke'), max_examples=256 * 300, check_every=50, test_every=10 ** 9,
                 per_lookahead=50, eval_sizes=[49], eval_iters=list(range(1, 9)), timeout=1800) for j in stage20() if j['id'].endswith('-s0')]


def bench20():
    """S20 N3: speed and peak memory of one layer per aggregator, H100."""
    return [dict(id='s20-bench-h100', script='bench_layers.py', sizes=[49, 100, 125, 250], aggs=['sum', 'attn', 'newton', 'pmean'],
                 batch=32, reps=20, timeout=5000)]


def theory20():
    """S20 theory check on the benchmark's test graphs at 125 and 250 vertices (CPU)."""
    return [dict(id='s20-theory-bench', script='bench_layers.py', fn='run_cpu', modes=[], sizes=[], aggs=[], batch=1,
                 theory_sizes=[[125, 8], [250, 4]], timeout=8000)]


def stage21():
    """S21: S20's Newton-OR with J = 1, O = 1 - exp(-omega_1 sum_k a c) (the rate form); all else as S20."""
    return [dict(j, id=j['id'].replace('s20-loopnewton', 's21-looprate'), model=dict(j['model'], J=1))
            for j in stage20() if 'newton' in j['id']]


PROP = dict(PAIRN, layers=1, nopos=True, tri='prop', pheads=4, t0s=[.5, .9, .99, .999])
S22 = dict(MAIN, script='loop_train.py', vertices=49, name_pool=256, lr=1e-3, max_examples=3_000_000, test_every=5000,
           eval_sizes=[49, 75, 100, 125, 250], per_lookahead_big=200, eval_batch=32, timeout=9000)


def pilot22():
    """S22 pilot: one and two propagator layers, seed 100, 512k examples, scored at 49 vertices only."""
    return [dict(S22, id=f's22-pilot-L{L}', seeds=[100], model=dict(PROP, layers=L), max_examples=512_000, check_every=100,
                 test_every=10 ** 9, per_lookahead=100, eval_sizes=[49], timeout=2400) for L in (1, 2)]


def stage22():
    """S22: the propagator join, seeds 0 to 2; layers as the pilot chose (S22_LAYERS)."""
    import os
    L = int(os.environ.get('S22_LAYERS', 1))  # the paper's runs: one layer
    return [dict(S22, id=f's22-prop-L{L}-v49-s{sd}', seeds=[sd], model=dict(PROP, layers=L)) for sd in (0, 1, 2)]


def stage22f():
    """S22 follow-up: every channel starts at t0 = 0.999 (seed 1 used the t0 = 0.5 channel and decayed)."""
    return [dict(S22, id=f's22f-t999-v49-s{sd}', seeds=[sd], model=dict(PROP, t0s=[.999] * 4)) for sd in (0, 1, 2)]


def stage22c():
    """S22 row-capped design: rows of every channel capped at t0 = 0.99 (fixed), so rho(G) <= 0.99 on any graph."""
    return [dict(S22, id=f's22c-cap99-v49-s{sd}', seeds=[sd], eval_batch=64, abort_nonfinite=True,
                 model=dict(PROP, t0s=None, t0=.99, learn_t0=False, rowcap=True)) for sd in (0, 1, 2)]


def smoke22c():
    """S22c smoke test: 600 updates of the capped layer on real 49-vertex questions; the job fails unless every
    check's loss and gradient norm are finite and the loss falls."""
    j = stage22c()[0]
    return [dict(j, id='s22c-smoke', max_examples=256 * 600, check_every=50, test_every=10 ** 9, per_lookahead=20, eval_sizes=[49],
                 assert_smoke=True, abort_nonfinite=True, timeout=1500)]


def bench22():
    """S22: speed and memory against the rate OR join (1 and K blocks), and the hand-set check on the GPU."""
    import os
    return [dict(id='s22-bench-h100', script='bench_layers.py', sizes=[49, 100, 125, 250], aggs=['rate', 'rateK', 'propL'],
                 prop_layers=int(os.environ.get('S22_LAYERS', 1)), batch=32, reps=20, gpu_exact=True, timeout=5000)]


def stage23():
    """S23: S3's Transformer over names with the pair state's read-out input (w_r . zn[cur, v], zeros at
    start); S3's protocol, 49 and 125 vertices, seeds 0 to 2, abort on a non-finite loss."""
    return [dict(MAIN, id=f's23-sannpr-v{V}-s{sd}', vertices=V, model=dict(SANN, pair_read=True), lr=1e-3, seeds=[sd], max_examples=10_000_000,
                 save=True, per_lookahead=1000 if V <= 75 else 500, abort_nonfinite=True) for V in (49, 125) for sd in (0, 1, 2)]


def smoke23():
    """S23 smoke test: 600 updates of the S23 arm on real 49-vertex questions; fails unless every check's loss and
    gradient norm are finite and the loss falls."""
    j = stage23()[0]
    return [dict(j, id='s23-smoke', max_examples=256 * 600, check_every=50, test_every=10 ** 9, per_lookahead=20, save=False,
                 assert_smoke=True, timeout=1500)]


def main():
    jobs = globals()[sys.argv[1]]()
    fns = {name: modal.Function.from_name('witnesses-search', name) for name in ('run', 'run_theirs', 'run_cpu')}
    calls = {job['id']: fns[job.get('fn', 'run')].spawn(job).object_id for job in jobs}
    print(json.dumps(calls, indent=1))
    print('spawned', len(calls))


if __name__ == '__main__':
    main()
