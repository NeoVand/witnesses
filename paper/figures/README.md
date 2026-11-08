# Figures

Scripts that draw the paper's five figures. Each reads only the files under `data/` (snapshots of the results the
figures plot) and writes `figN.pdf`, `figN.svg` (text as outlines) and `figN.png` (300 dpi).

- Fig. 1, what a layer sums over (lookup, join, closed form on the 49-town map and a SET deal): `fig1.py`; reads `data/map_layout.json`, `data/map_question.json`.
- Fig. 2, the trained propagator on a 250-vertex search question (map, learned gates, convergence by size, accuracy by lookahead): `fig2.py`; reads `data/q250.json`, `data/prop250.json`, `data/fig2_gates.json`, `data/fig2_map_even.json`, `data/search/results/`.
- Fig. 3, SET (learning curves, one trained join's clocks, characters of (Z_3)^4, cost, larger tables): `fig3.py`; reads `data/connected/`, `data/clocks_m81.json`, `data/fig3_tables.json`.
- Fig. 4, the size law and the depth law: `fig4.py`; reads `data/fig4_extra.json`, `data/q250.json`, `data/map250.json`, `data/search/results/`.
- Fig. 5, what the principles predict (CLRS, loops, near-ties, interaction shares and language): `fig5.py`; reads `data/clrs/`, `data/fig5_margins.json`, `data/shares.json`, `data/lab/results/`.

Shared modules: `style.py`, `house.py` (house style, legends, SET cards, run readers), `layout.py` (millimetre layout,
fonts, checks), `palette.py` (`data/style/`), `mapdraw.py` (the road map) and `common_fig1.py` ... `common_fig5.py`.

Run (macOS, for Helvetica Neue; the bold, italic and medium faces are extracted from the system font into a
git-ignored `.fonts/` cache on first run):

    cd paper/figures && FIG_OUT=/tmp/figs uv run --no-project --with numpy --with scipy --with matplotlib==3.10.7 python -I fig1.py

`FIG_OUT` defaults to `out/`. The files in `out/` are the ones in the paper; with matplotlib 3.10.7 the scripts
reproduce them byte for byte.
