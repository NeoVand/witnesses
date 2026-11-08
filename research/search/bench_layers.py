"""S20: speed and peak memory of one looped pair layer, per aggregator over the middle vertex.

For each aggregator (sum = S8, attn = S9's triangle attention, newton = Newton-OR, pmean = power mean) and each graph
size n (names 0..n, so N = n + 1 slots), one training step of one block as in training (bf16 matmuls, the block
rematerialised): forward and backward of a loss on the block's output with respect to the parameters and the input
pair state. Timed over 'reps' steps after compiling; peak memory from the device's peak_bytes_in_use, measured in a
fresh process per configuration, and XLA's own temp + argument + output sizes. Inference (one forward in float32, as at
test) is measured the same way. A configuration that runs out of memory is retried at a quarter of the batch.
python bench_layers.py '<json>' prints one JSON line; writes bench.json to LAB_DEST.
"""
import json, math, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PAIRN = dict(kind='pairn', emb=32, D=16, layers=1, r=8, loop=True, nopos=True)
# S22: the rate-form OR join (S21) one block and K = ceil(log2 n) + 2 blocks (its scored test depth), the propagator join
PROP = dict(tri='prop', pheads=4, t0s=[.5, .9, .99, .999])
S22_AGGS = {'rate': dict(tri='newton', J=1), 'rateK': dict(tri='newton', J=1), 'prop': PROP, 'propL': PROP}


def one(spec):
    import numpy as np, jax, jax.numpy as j
    sys.path.insert(0, str(HERE))
    import search_train as st
    agg, n, B, mode, reps = spec['agg'], spec['n'], spec['batch'], spec['mode'], spec['reps']
    cfg = dict(PAIRN) | (S22_AGGS[agg] if agg in S22_AGGS else {} if agg == 'sum' else dict(tri=agg))
    train = mode == 'train'
    iters = math.ceil(math.log2(n)) + 2 if agg == 'rateK' else spec.get('prop_layers', 1) if agg == 'propL' else 1  # S22: whole joins
    c1 = dict(cfg, iters=iters, bf16=train, remat=train)
    N = n + 1
    p = st.init(jax.random.PRNGKey(0), cfg, 200, 3 * n + 5)
    kz = jax.random.PRNGKey(1)
    z = jax.random.normal(kz, (B, 16, N, N))
    m = j.ones((B, N)).at[:, 0].set(0.)  # slot 0 is never a name
    nv = m.sum(1)[:, None, None, None]
    if train:
        def loss(p, z):
            out = st.pair_stack(p, c1, z, m, nv)
            return (out * out).mean()
        f = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
        args = (p, z)
    else:
        f = jax.jit(lambda p, z: st.pair_stack(p, c1, z, m, nv))
        args = (p, z)
    t0 = time.time()
    comp = f.lower(*args).compile()
    ma = comp.memory_analysis()
    xla = dict(temp=int(ma.temp_size_in_bytes), args=int(ma.argument_size_in_bytes), out=int(ma.output_size_in_bytes))
    compile_s = time.time() - t0
    jax.block_until_ready(comp(*args))
    jax.block_until_ready(comp(*args))
    t0 = time.time()
    for _ in range(reps):
        r = comp(*args)
    jax.block_until_ready(r)
    dt = time.time() - t0
    dev = jax.devices()[0]
    ms = dev.memory_stats() or {}
    return dict(spec, examples_per_s=B * reps / dt, sec_per_step=dt / reps, peak_bytes=int(ms.get('peak_bytes_in_use', -1)),
                xla_bytes=xla, compile_s=round(compile_s, 1), device=str(dev.device_kind))


def main(job):
    dest = Path(os.environ.get('LAB_DEST', '.'))
    rows = []
    for mode in job.get('modes', ['train', 'infer']):
        for n in job['sizes']:
            for agg in job['aggs']:
                B = job['batch']
                while True:
                    spec = dict(agg=agg, n=n, batch=B, mode=mode, reps=job.get('reps', 20), prop_layers=job.get('prop_layers', 1))
                    r = subprocess.run([sys.executable, __file__, '--one', json.dumps(spec)], capture_output=True, text=True,
                                       env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false'), timeout=job.get('one_timeout', 900))
                    if r.returncode == 0:
                        row = json.loads(r.stdout.strip().splitlines()[-1])
                        break
                    oom = 'RESOURCE_EXHAUSTED' in r.stderr or 'out of memory' in r.stderr.lower()
                    if oom and B > 1:
                        rows.append(dict(spec, error='out of memory'))
                        B = max(1, B // 4)
                        continue
                    row = dict(spec, error=r.stderr[-1500:])
                    break
                rows.append(row)
                (dest / 'bench.json').write_text(json.dumps(rows, indent=1))
                print(json.dumps({k: row.get(k) for k in ('mode', 'agg', 'n', 'batch', 'examples_per_s', 'peak_bytes', 'error')}), file=sys.stderr, flush=True)
    if job.get('gpu_exact'):  # S22: the hand-set exactness check (propagator_checks.py exact) on this GPU
        r = subprocess.run([sys.executable, str(HERE / 'propagator_checks.py'), 'none', job.get('their_train_py', '/root/l2s/train.py'), 'exact'],
                           capture_output=True, text=True, env=dict(os.environ, S22_OUT=str(dest / 's22_checks_gpu.json'), XLA_PYTHON_CLIENT_PREALLOCATE='false'),
                           timeout=job.get('one_timeout', 900) * 2)
        (dest / 'gpu_exact.log').write_text(r.stdout[-20000:] + r.stderr[-5000:])
    if job.get('theory_sizes'):
        sys.path.insert(0, str(HERE))
        rows_t = theory_bench(job)
        (dest / 'theory_bench.json').write_text(json.dumps(rows_t, indent=1))
    return dict(rows=len(rows))


def theory_bench(job):
    """The hand-set OR layer (and Theorem 1's summed construction) on the benchmark's own test graphs at larger sizes."""
    import math
    import numpy as np
    sys.path.insert(0, job.get('generator_dir', '/root/l2s'))
    import l2s_eval, newton_or_checks as chk
    ns = l2s_eval.load(job.get('their_train_py', '/root/l2s/train.py'))
    out = {}
    for V, per in job['theory_sizes']:
        sets = l2s_eval.test_sets(ns, 3 * V + 5, per, rng_seed=job.get('test_seed', 12345))
        graphs = []
        for L, (xs, ys) in sets.items():
            for x, y in zip(xs, ys):
                toks = [int(t) for t in x if t != V + 3]
                q = toks.index(V + 4)
                A = np.zeros((V + 1, V + 1))
                for i in range(0, q, 3):
                    if toks[i] == V + 2:
                        A[toks[i + 1], toks[i + 2]] = 1
                graphs.append((A, toks[q + 2], toks[q + 4:][-1], int(y), L))
        Ls = [1, 2, 3, math.ceil(math.log2(V)) + 1]
        t0 = time.time()
        out[str(V)] = dict(questions=len(graphs), per_lookahead=per, or_layer=chk.score(graphs, chk.or_layers, Ls),
                           or_layer_J1=chk.score(graphs, chk.or_layers, Ls, J=1), sum_layer=chk.score(graphs, chk.sum_layers, Ls),
                           seconds=round(time.time() - t0))
        print('theory', V, json.dumps(out[str(V)])[:600], file=sys.stderr, flush=True)
    return out


if __name__ == '__main__':
    if sys.argv[1] == '--one':
        print(json.dumps(one(json.loads(sys.argv[2]))))
    else:
        print(json.dumps(main(json.loads(sys.argv[1]))))
