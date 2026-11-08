# SET

Joins, Transformers, Newton attention and exponential Newton on SET (Section "Learning SET", Fig. 3, Supplementary
Table "SET", SET rows of the registered-outcomes table). All SET jobs ran on one NVIDIA A10G each.

## Install and rerun

- Modal image (`modal_app.py`): Python 3.12, `jax[cuda12]==0.8.0`, `optax==0.2.6`, `numpy==2.3.4`; app and volume
  `witnesses-set`.
- `modal deploy modal_app.py`, then `python spawn2.py`, `python spawn3.py`, `python spawn8.py`,
  `python spawn_nset.py {smoke,p1,smoke256,r1,fix,fixrest,count,mix}`, `python newton_paper/spawn_cost.py`,
  `python expnewton/spawn_exp.py {main,diag,main6,split}` (`diag` after `main`).
- Volume folders map to local ones: `connected2` -> `results2/`, `connected3` -> `results3/`, `connected8` ->
  `results8/`, `connected-nset` -> `results_nset/`, `connected-expnewton` -> `expnewton/results/`.
- Local analyses need numpy and scipy; `newton_theory/*`, `newton_paper/count_check.py`, `construction.py` and
  `expnewton/check_layer.py` also need jax 0.8.0 and optax 0.2.6 on CPU.

## Files

| Script | Produces | Paper item |
|---|---|---|
| `tasks.py`, `operators.py`, `groups.py`, `hoa_polybench.py` | deals, masks, layers | shared |
| `bench2.py`, `bench3.py` (+ `spawn2.py`, `spawn3.py`) | `results2/`, `results3/` | Fig. 3a join and Transformers; Supp. Table SET |
| `bench8.py` (+ `spawn8.py`) | `results8/q-*/fits.jsonl` | 27 cards in hand: Transformer 93%, join 99.6%; chance with 9 or fewer cards in hand |
| `bench8.py` | `results9/cov-{a,b}/fits.jsonl` | the join with 3 to 21 cards in hand (55,296 deals): chance up to 12, 59 to 87% with 18 or 21 at the picked rate |
| `mechanism.py` | `results8/q-*/mech_*_c1.json` (from `params_*_c1.npz`) | clock share 82-94% (81 cards in hand) |
| `clocks_data.py` | Fig. 3b clocks (`paper/figures/data/clocks_m81.json`, from `results8/q-m81b`) | Fig. 3b |
| `bench_nset.py`, `newton_chars.py`, `analyze_nset.py` (+ `spawn_nset.py`) | `results_nset/` | Newton: 55,714 parameters, 90% at 32,768, 38-40 of 40 characters, 16-card tables, count form, 8- and 12-card training |
| `newton_paper/export.py` | `newton_paper/newton_data.json` | Fig. 3a/3d data, Supp. Table SET |
| `newton_paper/cost_bench.py` (+ `spawn_cost.py`) | `results_nset/cost-1/cost.jsonl` | Fig. 3d time and memory |
| `newton_paper/count_check.py`, `construction.py`, `newton_theory/check_count.py`, `handset_sizes.py` | `.json` beside each | Proposition 2; count exact at every size |
| `newton_theory/table16.py`, `table16_model.py`, `retrain_rule.py` | `table16.json`, `table16_model.json`, `params/` | Methods, Newton's 16-card threshold (0.54 -> 1.76-1.79, measured 1.76-1.77) |
| `newton_theory/charlaw*.py` | `charlaw*.json` | channels hold characters one by one |
| `expnewton/bench_exp.py`, `analyze_exp.py`, `diag_exp.py` (+ `spawn_exp.py`) | `expnewton/summary.json`, `expnewton/results/exp-*` | four clocks: 69% at 2,048, 90% at 8,192, rank 3 in 10 of 12, pseudo-set errors |
| `expnewton/analyze6.py` | `expnewton/summary6.json`, `expnewton/results/exp8-rule-{a,b}/fits.jsonl` | six clocks: 99.9% at 8,192; at least 97% at 16 and 20 cards; beta = 4.86 |
| `expnewton/check_exact.py`, `handset.py`, `check_layer.py`, `ref.py` | `check_exact.json`, `handset.json`, `check_layer.json` | Proposition 3 against brute force; hand-set layer; 8-card beta 66.7% at 32 cards |

`results*/` hold per-fit summaries (`fits.jsonl`); parameters ship only where a shipped analysis reads them
(`results8/q-m*/params_*_c1.npz`, read by `mechanism.py` and by `../language/shares.py` for Fig. 5d).
