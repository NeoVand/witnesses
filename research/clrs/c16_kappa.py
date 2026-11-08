"""The learned log n shift of the Kirchhoff conductances (c16): the final kappa of each of the 4 heads, per seed, from the
runs' params_s<seed>.npz. The prediction asked for a mean of at least 1.7.
CPU, numpy. Writes results/c16_kappa.json.
python3 -I c16_kappa.py [results dir holding c16-mst_prim-s<seed>/, default results]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


def main(results=HERE / 'results'):
    out, allk = {}, []
    for sd in (0, 1, 2):
        W = np.load(Path(results) / f'c16-mst_prim-s{sd}' / f'params_s{sd}.npz')
        k = [float(x) for x in W["['kir']['kap']"]]
        out[f's{sd}'] = [round(x, 4) for x in k]
        allk += k
    out |= dict(min=round(min(allk), 4), max=round(max(allk), 4), mean=round(float(np.mean(allk)), 4), init=0.5, required_mean=1.7)
    print(json.dumps(out))
    (HERE / 'results' / 'c16_kappa.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
