"""S23 scoring: per run, convergence, training accuracy at the end, test mean and worst over
lookaheads, random graphs, |w_r| and wall time, beside S3 (the same control without the read-out input).
python3 analyze_s23.py; reads results/<id>/rows.json, result.json and params; writes results/s23_summary.json."""
import json
from statistics import mean
from pathlib import Path
R = Path(__file__).resolve().parent / 'results'


def one(run):
    d = R / run
    if not (d / 'rows.json').exists():
        return None
    r = json.loads((d / 'rows.json').read_text())[0]
    prog = [json.loads(l) for l in (d / 'progress.jsonl').read_text().splitlines()]
    looks = [k for k in r['test'] if k != 'None']
    out = dict(converged_at=r['converged_at'], examples=r['examples'], final_train_acc=prog[-1]['train_acc'], final_loss=prog[-1]['loss'],
               test_mean=mean(r['test'][k] for k in looks), worst=min(r['test'][k] for k in looks),
               worst_lookahead=int(min(looks, key=lambda k: r['test'][k])), random=r['test'].get('None'), train_seconds=round(r['seconds']))
    res = d / 'result.json'
    if res.exists():
        out['wall_seconds'] = round(json.loads(res.read_text()).get('wall_seconds', 0))
    pz = d / f"params_s{r['seed']}.npz"
    if pz.exists():
        import numpy as np
        w = np.load(pz)
        out['wr_norm'] = float(np.linalg.norm(w["['wr']"])) if "['wr']" in w else None
    return out


summary = {}
for V in (49, 125):
    for sd in (0, 1, 2):
        for tag in (f's23-sannpr-v{V}-s{sd}', f's3-sann-v{V}-s{sd}'):
            s = one(tag)
            if s is None:
                continue
            summary[tag] = s
            print(f"{tag}: converged_at {s['converged_at']} examples {s['examples']} train_acc {s['final_train_acc']:.3f} "
                  f"test mean {s['test_mean']:.4f} worst {s['worst']:.3f} (L={s['worst_lookahead']}) random {s['random']:.3f} "
                  f"|w_r| {s.get('wr_norm')} train {s['train_seconds']} s wall {s.get('wall_seconds')} s")
(R / 's23_summary.json').write_text(json.dumps(summary, indent=1))
