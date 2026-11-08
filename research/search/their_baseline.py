"""Run Saparov et al.'s own Transformer training (their train.py, unmodified) as the control, and stop it at a cap.

Their loop never ends on its own; this reads its log, records training accuracy, test accuracy and test loss per
"epoch" (2^18 examples by default), and kills it once the example cap is reached. Their training accuracy is printed
to two decimals, so "converged" here means it prints 1.00 on two evaluations in a row (that is, at least 0.995).
python their_baseline.py '<json>' prints one JSON line; writes rows.json and progress.jsonl to LAB_DEST.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
DEST = Path(os.environ.get('LAB_DEST', '.'))


def main(job):
    V = job['vertices']
    n = 3 * V + 5
    args = ['--max-input-size', n, '--dataset-size', -1, '--max-lookahead', (V - 1) // 2, '--nlayers', job.get('nlayers', 8),
            '--hidden-dim', job.get('hidden', 16), '--seed', job['seed'], '--bidirectional', 'y', '--pos-emb', 'absolute', '--learn-tok-emb', 'n',
            '--toeplitz-attn', 'y' if job.get('toeplitz') else 'n', '--toeplitz-reg', 0.0, '--toeplitz-pos-only', 'y' if job.get('toeplitz') else 'n', '--add-padding', job.get('padding', 'n'), '--ablate', 'none',
            '--preLN', 'y', '--curriculum', 'n', '--looped', 'n', '--task', 'search', '--distribution', 'crafted',
            '--batch-size', job.get('batch', 256), '--learning-rate', job.get('lr', 1e-5)]
    cmd = [sys.executable, 'train.py'] + [str(a) for a in args]
    block = 2 ** 18
    t0, rows, hits, conv = time.time(), [], 0, None
    proc = subprocess.Popen(cmd, cwd=job.get('repo', '/root/l2s_repo'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    epoch, cur = None, {}
    log = (DEST / 'their_stdout.log').open('w')
    for line in proc.stdout:
        log.write(line)
        m = re.match(r'epoch = (\d+), training loss = ([\d.eE+-]+)', line)
        if m:
            epoch, cur = int(m.group(1)), dict(epoch=int(m.group(1)), loss=float(m.group(2)))
        m = re.match(r'training accuracy: ([\d.]+)', line)
        if m and epoch is not None:
            cur['train_acc'] = float(m.group(1))
        m = re.match(r'test accuracy = ([\d.]+).*test loss = ([\d.eE+-]+)', line)
        if m and epoch is not None:
            cur |= dict(test_acc=float(m.group(1)), test_loss=float(m.group(2)), examples=(epoch + 1) * block, sec=round(time.time() - t0))
            rows.append(cur)
            with (DEST / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(cur) + '\n')
            hits = hits + 1 if cur.get('train_acc', 0) >= .995 else 0
            if hits >= 2 and conv is None:
                conv = cur['examples']
            if cur['examples'] >= job['max_examples'] or (conv is not None and job.get('stop_on_converge', True)) or time.time() - t0 > job.get('max_seconds', 1e9):
                break
    proc.kill()
    log.close()
    out = dict(vertices=V, seed=job['seed'], converged_at=conv, examples=rows[-1]['examples'] if rows else 0, seconds=time.time() - t0,
               last=rows[-1] if rows else None, best_test_acc=max((r['test_acc'] for r in rows), default=None))
    (DEST / 'rows.json').write_text(json.dumps([out | dict(curve=rows)]))
    return out


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
