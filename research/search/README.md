# Search on the benchmark of Saparov et al. (2025)

Code and per-run results behind Section 3, Fig. 2, Fig. 4b and Supplementary Tables 1, 2 and the search register
table. Models are in JAX; the benchmark's own Transformer runs from its authors' PyTorch code.

## Setup

- Benchmark: `./setup_benchmark.sh [dest]` clones github.com/asaparov/learning_to_search at 2c3f32b and its header
  library github.com/asaparov/core at 1f3b022, and builds the C++ generator (pybind11) into `dest/build` with their
  `train.py`. Their code is not copied here. Test sets are theirs: `l2s_eval.py` executes only the graph and
  test-set functions of their `train.py` (`generate_eval_data`) at run time.
- Python 3.12, `jax[cuda12]==0.8.0`, `optax==0.2.6`, `numpy==2.3.4`, `pybind11==2.13.6`; the benchmark's Transformer
  uses `torch==2.5.1`; `modal` for the GPU runs.

## Running on Modal

    modal deploy modal_search.py          # app and volume witnesses-search; builds the generator in the image
    python spawn_search.py stage22c       # one group of runs (see the functions in spawn_search.py)
    python analyze_search.py --fetch s22c-cap99-v49-s0 s22c-cap99-v49-s1 s22c-cap99-v49-s2

`run` uses one H100 per job; `run_cpu` is CPU-only; `run_theirs` runs their `train.py` unmodified under
`their_baseline.py`. Each job writes `result.json`, `rows.json` (final scores per size and lookahead),
`progress.jsonl` (training curve) and, if saved, `params_s<seed>.npz` to `search/<run id>` on the volume.
A job runs locally too: `LAB_DEST=out python search_train.py '<job json>'` with `generator_dir` and
`their_train_py` in the job pointing at the setup output.

## Scripts

| Script | What it does | Paper item |
|---|---|---|
| `search_train.py` | Models and training: pair state over positions (`pair`) and over names (`pairn`, summed, triangle attention, OR join / Newton-OR `newton`, power mean, propagator `prop`), Transformer over names (`sann`), Transformer over tokens (`sa`); training with convergence check and scoring on the benchmark's test sets | Sec. 3, Supp. Table 1 |
| `loop_train.py` | Looped one-block layers and the propagator, trained at 49 vertices and scored at 49 to 250 | Sec. 3, 5, Fig. 2d-e, Supp. Table 2 |
| `l2s_eval.py` | Loader for the benchmark's test sets (their `generate_eval_data`) | Methods, every test score |
| `their_baseline.py` | Runs the benchmark's Transformer (their `train.py`) and stops it at an example cap | Supp. Table 1, benchmark rows |
| `modal_search.py`, `spawn_search.py` | Modal app and the job spec of every run group (`s1`...`s23`; run ids carry the group) | all trained runs |
| `setup_benchmark.sh` | Clones and builds the benchmark at the pinned commits | Methods |
| `bench_layers.py` | Speed and peak memory per training step (bf16, batch 32, H100): propagator against the looped OR join | Sec. 3 (6.0 and 6.9 times; 0.39 and 0.52 of memory), `results/s22-bench-h100/bench.json` |
| `propagator_checks.py` | Hand-set propagator: exact reach and distance, background-gate margin against Sherman-Morrison, row cap | Methods, propagator; `results/s22_checks.json` |
| `newton_or_checks.py` | Newton-OR series error, Boolean limit, hand-set OR layer on the benchmark's graphs | Methods, OR join; `results/s20_checks.json` |
| `analyze_search.py` | One line per run (convergence, examples, accuracy) | Supp. Table 1 |
| `analyze_s22.py`, `analyze_s23.py`, `analyze_s20.py`, `analyze_s8.py`, `table_rows.py`, `s15_score.py` | Summaries of the propagator, the Transformer over names with pair read-out, the OR join, the looped layers and their rows | Supp. Tables 1, 2, register table; `results/s2x_summary.json` |
| `diag_s22.py` | Learned gates, sources and spectral radius of trained propagators | Methods, earlier propagator designs; `results/s22*/diag.json` |
| `search250.py`, `fig2_data.py` | The 250-vertex test question, its map and the trained propagators' gates on it | Fig. 2a-c, Sec. 3 gate medians; write `paper/figures/data/{q250,map250,prop250,fig2_gates}.json` |
| `bg250.py` | Largest background gate the hand-built capped layer tolerates on that question | Sec. 3 (4 x 10^-3), Methods (pole at 10^-3.6) |
| `probe_search.py`, `mech_loop.py` | Linear probes on the pair state; the read-out applied after each layer | Supplement, "What the trained pair state holds" |
| `step_eval.py`, `merge_eval.py`, `gap_analysis.py`, `train_stats.py` | Random-graph accuracy by step and by merging paths; statistics of the training distribution | Supplement, "Random graphs and token positions" |
| `flops_looptri.py` | Operation count of looped triangle attention | `results/flops_looptri.json` |
| `export_demo.py`, `export_prop.py`, `export_orjoin.py`, `modal_export125.py`, `maps_v125.py` | Weights, questions and parity references for the browser demo (`static/model`, `test-data`) | demo |

The map layouts (`search250.py map`, `fig2_data.py map`, `maps_v125.py`) use `map_engine.py` and `map_draw.py` in this folder
(the road-map layout of every question; `python -I map_engine.py all` lays out the demo's 300 questions; they need
`numpy`, `scipy`, `matplotlib==3.10.7` and `networkx`).
Analyses run on a CPU from saved parameters, for example
`JAX_PLATFORMS=cpu python -I search250.py prop`.

## Results

`results/<run id>/` holds `started.json` (the job spec), `result.json`, `rows.json`, `progress.jsonl` and, where
scored, `sizes_s<seed>.json`, `diag.json`, `probe.json`, `step_eval.json`. Run ids give the group, model and size, e.g.
`s2-pairn-v125-s0` (pair state over names, 125 vertices, seed 0), `s22c-cap99-v49-s0` (propagator with the row cap),
`s23-sannpr-v49-s0` (Transformer over names with the pair read-out), `s1t-v41-s1` (the benchmark's Transformer).
Group summaries: `s20_summary.json`, `s21_summary.json`, `s22_summary.json`, `s23_summary.json`, `s22_checks.json`.

Parameters are included for the three propagator runs behind Fig. 2 (`s22c-cap99-v49-s*`) and the OR join in the
demo (`s21-looprate-v49-s1`). Other checkpoints (17 MB in all) and raw logs are not included; every run can be
retrained from its job spec. `export/graphs_v49.json` holds tokenised 49-vertex test questions from the benchmark's
own test-set code, read by `newton_or_checks.py`.
