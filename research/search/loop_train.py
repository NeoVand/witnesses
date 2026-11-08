"""S8: one pair layer, looped, trained on graphs of 49 vertices and run on larger graphs.

Names. The pair state is over the names of the question (slots 0..V, by the generator's own names), but each name's
embedding is looked up from a pool of P: in training the names 1..V are mapped at random, per question, into 1..P, so
every embedding a graph of up to P vertices uses is trained. The markers P (path), E (edge), padding and Q (query) are
looked up at P+1..P+4 at every size. A test question of V' <= P vertices keeps its names. The pair state does not
depend on the order of its slots, so this is the same function as a pair state over all P names, at the cost of V.
Positions. No absolute position embedding ('nopos'), so the model reads token sequences of any length.
Loop. With 'loop', the model has one block, run K times; K is drawn per training batch from 'train_iters'. At test
the read-out after iteration K of one run of max(eval_iters) iterations equals the read-out of a run of K, so one
pass scores every K.
python loop_train.py '<json>' prints one JSON line; writes rows.json, progress.jsonl and params to LAB_DEST.
"""
import json, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as j, optax
sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_train as st
DEST = st.DEST


def pool_ids(x, V, P, rng):
    """Embedding ids for tokens x in names 1..V: names drawn at random into 1..P per question, markers at P+1..P+4."""
    B = x.shape[0]
    table = np.zeros((B, V + 5), np.int64)
    table[:, 1:V + 1] = np.argsort(rng.random((B, P)), 1)[:, :V] + 1
    table[:, V + 1:] = P + np.arange(1, 5)
    return np.take_along_axis(table, x.astype(np.int64), 1)


def marker_ids(x, V, P):
    """Embedding ids for a test question: its names as they are, markers at P+1..P+4."""
    t = np.arange(V + 5)
    t[V + 1:] = P + np.arange(1, 5)
    return t[x]


def logits(p, c, x, e, pad):
    """Vocabulary logits over the question's own token space (names 0..V, pad = V + 3); e are embedding ids."""
    if c['kind'] == 'sann':  # S17: the Transformer over names, looped
        h, zn, present, Nn = st.sann_start(p, x, pad, p['tok'][e])
        return st.sann_read(p, st.sann_stack(p, c, h, zn, present), x, present, Nn, pad, zn)
    z, present, nv, Nn = st.pairn_start(p, x, pad, p['tok'][e], c.get('pool_norm', False))
    return st.pairn_read(p, st.pair_stack(p, c, z, present, nv), x, present, Nn, pad)


def main(job):
    sys.path.insert(0, job.get('generator_dir', '/root/l2s'))
    import generator, l2s_eval
    V, P = job['vertices'], job['name_pool']
    n = 3 * V + 5
    pad = V + 3  # the question's own padding token; the embedding table has P + 5 rows
    max_look = (V - 1) // 2
    ns = l2s_eval.load(job.get('their_train_py', '/root/l2s/train.py'))
    tests = l2s_eval.test_sets(ns, n, job.get('per_lookahead', 1000), rng_seed=job.get('test_seed', 12345))
    reserved = {tuple(int(t) for t in row if t != V + 3) for L, (xs, _) in tests.items() for row in xs}
    cfg = job['model']
    loop = bool(cfg.get('loop'))
    train_iters = job.get('train_iters', [None]) if loop else [None]
    eval_iters = max(job.get('eval_iters', [1])) if loop else None
    ck = lambda K: dict(cfg, iters=K) if loop else cfg
    B = job['batch']
    sched = optax.join_schedules([optax.linear_schedule(0., job['lr'], job.get('warmup', 2000)), optax.constant_schedule(job['lr'])], [job.get('warmup', 2000)])
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(sched, weight_decay=job.get('wd', .01)))

    def make_step(c):
        @jax.jit
        def step(p, s, x, e, lab):
            def loss(p):
                lp = jax.nn.log_softmax(logits(p, c, x, e, pad), -1)
                return -j.take_along_axis(lp, lab[:, None], 1).mean(), lp
            (v, lp), g = jax.value_and_grad(loss, has_aux=True)(p)
            u, s = tx.update(g, s, p)
            return optax.apply_updates(p, u), s, v, lp.argmax(-1), optax.global_norm(g)
        return step
    steps = {K: make_step(ck(K)) for K in train_iters}
    ecfg = dict(cfg, bf16=False, remat=False)

    @jax.jit
    def said_all(p, x, e):
        """[iterations, B] answers: after each iteration of a looped model, or the one answer of a fixed stack."""
        pad_t = x.shape[1] // 3 + 2  # (n - 5) / 3 + 3 for the question's own size
        if cfg['kind'] == 'sann':  # S17
            h, zn, present, Nn = st.sann_start(p, x, pad_t, p['tok'][e])
            if not loop:
                return st.sann_read(p, st.sann_stack(p, ecfg, h, zn, present), x, present, Nn, pad_t, zn).argmax(-1)[None]
            out = []
            for _ in range(eval_iters):
                h = st.sann_stack(p, dict(ecfg, iters=1), h, zn, present)
                out.append(st.sann_read(p, h, x, present, Nn, pad_t, zn).argmax(-1))
            return j.stack(out)
        z, present, nv, Nn = st.pairn_start(p, x, pad_t, p['tok'][e], cfg.get('pool_norm', False))
        if not loop:
            return st.pairn_read(p, st.pair_stack(p, ecfg, z, present, nv), x, present, Nn, pad_t).argmax(-1)[None]
        one = dict(ecfg, iters=1)
        out = []
        for _ in range(eval_iters):
            z = st.pair_stack(p, one, z, present, nv)
            out.append(st.pairn_read(p, z, x, present, Nn, pad_t).argmax(-1))
        return j.stack(out)

    def score(p, sets, Vt):
        """{iterations: {lookahead: accuracy}} on test sets of Vt vertices."""
        assert sets[1][0].shape[1] == 3 * Vt + 5
        out = {}
        for L, (xs, ys) in sets.items():
            e = marker_ids(xs, Vt, P)
            EB = job.get('eval_batch', B)
            said = np.concatenate([np.asarray(said_all(p, j.array(xs[i:i + EB]), j.array(e[i:i + EB]))) for i in range(0, len(xs), EB)], 1)
            for k, row in enumerate(said):
                out.setdefault(str(k + 1) if loop else str(cfg['layers']), {})[str(L)] = float((row == ys).mean())
        return out

    rows = []
    for seed in job['seeds']:
        generator.set_seed(seed + 1000)
        rng = np.random.default_rng(seed + 7000)
        p = st.init(jax.random.PRNGKey(seed), cfg, P + 5, n)
        s = tx.init(p)
        params = int(sum(w.size for w in jax.tree.leaves(p)))
        t0, seen, curve, hits, done_at = time.time(), 0, [], 0, None
        while seen < job['max_examples']:
            K = train_iters[rng.integers(len(train_iters))]
            look = -1 if rng.random() < job.get('mix_random', 0.) else max_look  # -1: the benchmark's random graphs (S12)
            if job.get('tight_k'):  # S15: K runs see only lookaheads up to 2^K, so every run must double the reach
                if look == -1:
                    K = max(train_iters)
                else:
                    look = min(2 ** K, max_look)
            x, y, lab, _ = generator.generate_training_set(n, B, look, V, reserved, -1, n, True)
            e = pool_ids(x, V, P, rng)
            p, s, v, said, gn = steps[K](p, s, j.array(x), j.array(e), j.array(lab.astype(np.int32)))
            seen += B
            if (seen // B) % job.get('check_every', 500) == 0:
                acc = float(np.mean(y[np.arange(B), np.asarray(said)] > 0))
                row = dict(seed=seed, examples=seen, loss=float(v), train_acc=acc, iters=K, sec=round(time.time() - t0), gnorm=float(gn))
                if job.get('abort_nonfinite') and not (np.isfinite(row['loss']) and np.isfinite(row['gnorm'])):  # S22c
                    with (DEST / 'progress.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
                    raise RuntimeError(f'non-finite loss or gradient norm at {seen} examples: {row}')
                if (seen // B) % job.get('test_every', 5000) == 0:
                    row['test'] = {k: v for k, v in score(p, tests, V).items() if not loop or int(k) == max(train_iters)}
                curve.append(row)
                with (DEST / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                hits = hits + 1 if acc >= .995 else 0
                if hits >= 2 and done_at is None:
                    done_at = seen
                    if job.get('stop_on_converge', False):
                        break
        train_sec = time.time() - t0
        if job.get('assert_smoke'):  # S22c: finite losses and gradient norms at every check, and the loss falls
            ls, gs = [r['loss'] for r in curve], [r['gnorm'] for r in curve]
            assert len(ls) >= 6 and all(np.isfinite(ls)) and all(np.isfinite(gs)), (ls, gs)
            assert np.mean(ls[-3:]) < np.mean(ls[:3]) - .1, ('loss did not fall', ls)
        flat, _ = jax.tree_util.tree_flatten_with_path(p)
        np.savez_compressed(DEST / f'params_s{seed}.npz', **{jax.tree_util.keystr(k): np.asarray(w) for k, w in flat})
        sizes = {}
        for Vt in job['eval_sizes']:
            per = 1000 if Vt <= 75 else 500 if Vt <= 125 else job.get('per_lookahead_big', 200)
            sets = tests if Vt == V else l2s_eval.test_sets(ns, 3 * Vt + 5, per, rng_seed=job.get('test_seed', 12345))
            sizes[str(Vt)] = score(p, sets, Vt)
            (DEST / f'sizes_s{seed}.json').write_text(json.dumps(sizes))
        rows.append(dict(cfg=cfg, seed=seed, vertices=V, name_pool=P, params=params, lr=job['lr'], batch=B, train_iters=train_iters,
                         examples=seen, converged_at=done_at, seconds=train_sec, sizes=sizes, curve=curve))
        (DEST / 'rows.json').write_text(json.dumps(rows))
    return dict(rows=[{k: v for k, v in r.items() if k not in ('curve', 'sizes')} for r in rows])


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
