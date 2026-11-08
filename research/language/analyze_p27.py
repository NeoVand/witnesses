"""Language-model numbers from results/p27-{r,every2,r10}-s{0,1}/result.json; prints one JSON object.

Gain = baseline test loss - arm test loss (token-weighted cross-entropy on the 1,024 held-out documents, nats per
token), paired by seed; the two arms of a seed see the same data order (data_order_sha256).
    python3 analyze_p27.py
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ARMS = dict(r='baseline, 8 layers', every2='join over 16 tokens in blocks 2, 4, 6, 8', r10='two more layers')
SEEDS = (0, 1)


def load(arm, seed):
    return json.loads((HERE / 'results' / f'p27-{arm}-s{seed}' / 'result.json').read_text())


def main():
    runs = {a: {s: load(a, s) for s in SEEDS} for a in ARMS}
    test = {a: {s: runs[a][s]['result']['test_loss'] for s in SEEDS} for a in ARMS}
    gain = {a: {s: test['r'][s] - test[a][s] for s in SEEDS} for a in ARMS if a != 'r'}
    one = lambda a, k: runs[a][0]['result'][k]  # noqa: E731
    out = dict(
        arms=ARMS,
        test_loss=test,
        gain={a: dict(seeds=g, mean=sum(g.values()) / len(g)) for a, g in gain.items()},
        params={a: one(a, 'counts')['total'] for a in ARMS},
        branch_params={a: one(a, 'counts')['branch'] for a in ARMS},
        tokens={a: one(a, 'tokens') for a in ARMS}, steps={a: one(a, 'steps') for a in ARMS}, lr={a: one(a, 'lr') for a in ARMS},
        config={a: runs[a][0]['job']['config'] | dict(arm=runs[a][0]['job']['arm']) for a in ARMS},
        same_data_order={s: len({runs[a][s]['result']['data_order_sha256'] for a in ARMS}) == 1 for s in SEEDS},
        finite=all(runs[a][s]['result']['finite'] for a in ARMS for s in SEEDS))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
