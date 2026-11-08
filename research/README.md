# Research code and results

The code, results and proofs behind every figure, table and theorem of "Attention Is Only as Good as Its Witnesses:
When Higher-Order Layers Help, and When They Do Not". Each folder has its own README with its scripts, what each
produces and the paper item it backs.

| Folder | What it holds | Paper |
|---|---|---|
| [`search/`](search/) | Pair states, the propagator, the OR join, Newton-OR and the Transformer over names on the search benchmark of Saparov et al. (2025): training, the loader for the benchmark's test sets, the analyses, the demo export and per-run results | Sec. 3, Sec. 5, Fig. 2, Fig. 4b, Supplementary Tables 1 and 2 |
| [`set/`](set/) | Newton attention, exponential Newton and the joins on the card game SET, the controls and the theory checks | Sec. 4, Sec. 5, Fig. 3, Supplementary Tables on SET |
| [`clrs/`](clrs/) | The OR-join and power-mean block on CLRS-30 without hints: training, scoring on the official and fresh test graphs, the loop-budget runs, the error analyses and the Kirchhoff read-out tests | Sec. 6, Fig. 5, Supplementary Table on CLRS-30 |
| [`language/`](language/) | The language model with the join and the interaction-share analyses (SET, chess, centred keys) | Sec. 6, Supplement |
| [`lean/`](lean/) | The Lean 4 project that checks every declaration named in the theorems and propositions | Theorems 1 and 2, Propositions, Proofs |
| [`checks/`](checks/) | Numeric checks cited in the Proofs and the Supplement, runnable on a CPU | Proofs, Supplement |

The scripts that draw the paper's five figures, and the data they read, are in [`../paper/figures/`](../paper/figures/).

## Running

Training runs used [Modal](https://modal.com): one NVIDIA H100 per search run, one L4 per CLRS-30 run and one A10G per
SET run. Each folder's `modal_*.py` defines its app and volume and its launcher holds the job specification of every
run, so any run can be repeated from its specification. Analyses of trained models run on a CPU from saved
parameters. Python and package versions are pinned in each folder's README.

Not included: model checkpoints other than those the figures and the demo read, raw training logs, and the code of
other authors (the search benchmark is cloned and built at its pinned commits by `search/setup_benchmark.sh`; CLRS-30
is the `dm-clrs` package).

## License

Apache License 2.0, as the rest of this repository.
