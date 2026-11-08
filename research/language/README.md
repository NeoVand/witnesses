# Language models, chess and interaction share

The 51-million-parameter language model with a join over 16 tokens, the chess models, and the interaction share of
trained joins (Section "What the principles predict elsewhere", last paragraph; Fig. 5d; Supplement "Language models
and chess"). Training ran on one NVIDIA H100 per job.

## Install and rerun

- Modal (`modal_app.py`), one app and volume `witnesses-language` for language and chess. Training image: Python 3.12,
  `jax[cuda12]==0.8.0`, `optax==0.2.6`, `numpy==2.3.4`, `tiktoken==0.12.0`. Chess data image: `chess==1.11.2`,
  `zstandard==0.23.0`, `numpy==2.3.4`.
- `modal deploy modal_app.py`; `python spawn.py language_data`, then `language`, `chess_data`, `chess24`, `chess25`,
  `chess26`; fetch with `modal volume get witnesses-language results/<id> results/`.
- FineWeb-Edu and the Lichess puzzle database are downloaded live; their revisions are not pinned. Counts and hashes
  of the data used are in `results/data-v1/manifest.json` and `results/c24-data/result.json`.

## Files

| Script | Produces | Paper item |
|---|---|---|
| `prepare.py` | tokenised FineWeb-Edu, `results/data-v1/manifest.json` | held-out documents (every 100th) |
| `model.py`, `cubics.py`, `hoa_lm.py`, `worker.py` | `results/p27-{r,every2,r10}-s{0,1}/result.json` | 8-layer GPT (50,975,744 parameters), join in blocks 2, 4, 6, 8 (54,715,392), two more layers (57,280,512); 200,015,872 tokens |
| `analyze_p27.py` | prints gains paired by seed | +0.011 nats (join), +0.026 (two more layers); Fig. 5d bottom |
| `chess_data.py` | `results/c24-data/result.json` | 6,193,530 puzzles; 50,000 test, 20,000 validation; 14,178,137 positions |
| `chess_train.py`, `chess_train2.py` | `results/c2{4,5,6}-*/{result,rows}.json` | chess models (6,450,176 parameters), joins in place of or beside attention |
| `analyze_chess.py` | prints chess comparisons | replacing attention -0.6 points; beside attention never more than +0.1; line themes 84.6-85.3% against 85.0% (6,681 puzzles) |
| `chess_diag.py`, `chess_diag2.py` | `results/c26-full-hyb8z/diag.json` | centred keys: interaction share 1.03 |
| `shares.py` | `results/shares.json` | interaction share: 42 trained SET joins 0.98-1.02, 6 trained chess joins 0.11-0.29, initialisation values; Fig. 5d top |

`shares.py` reads the saved SET joins from `../set/results8/q-m*/params_*_c1.npz` and the chess parameters from the
Modal volume (`results/c24-cub8-s0`, `c25-full-hyb8c`, `c26-full-hyb8r`; 25 to 27 MB each, not shipped). The language
runs saved no parameters. `hoa_lm.py` is reduced to the functions the backbone uses; `worker.py` and the chess scripts
differ from the run versions in paths and docstrings only, so the source hashes recorded in the results no longer
match the shipped files.
