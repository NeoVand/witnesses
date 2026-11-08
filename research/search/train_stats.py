"""Statistics of the benchmark's balanced training distribution (every step asked), from its own generator.
python train_stats.py <generator_dir> [vertices] [batches]; writes results/train_stats_v<V>.json. Backs the paper's
'96% of training questions ... are at the first step' and '0.4% of questions with a choice are after the first step'."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, sys.argv[1])
import generator
V = int(sys.argv[2]) if len(sys.argv) > 2 else 49
nb = int(sys.argv[3]) if len(sys.argv) > 3 else 200
n, B = 3 * V + 5, 256
PATH, EDGE, PAD, QUERY = V + 1, V + 2, V + 3, V + 4
generator.set_seed(4242)
first = choice = choice_first = 0
total = 0
for _ in range(nb):
    x, y, lab, _ = generator.generate_training_set(n, B, (V - 1) // 2, V, set(), -1, n, True)
    for row in x:
        row = [int(t) for t in row if t != PAD]
        p = len(row) - 1 - row[::-1].index(PATH)
        path = row[p + 1:]
        cur = path[-1]
        e = [i for i, t in enumerate(row[:p]) if t == EDGE]
        kids = {row[i + 2] for i in e if row[i + 1] == cur}
        f = len(path) == 1
        c = len(kids) > 1
        total += 1; first += f; choice += c; choice_first += c and f
out = dict(vertices=V, questions=total, first_step=first / total, with_choice=choice / total,
           choice_at_first_step=choice_first / max(choice, 1), choice_after_first_step=1 - choice_first / max(choice, 1))
(HERE / 'results' / f'train_stats_v{V}.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out))
