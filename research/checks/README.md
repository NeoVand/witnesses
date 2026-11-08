# Numeric checks

CPU checks of the statements the paper proves on paper and checks numerically (Proofs). Each runs in seconds to
minutes on a laptop. Run them from this folder.

| Paper | Script | Checks |
|---|---|---|
| Theorem 1(ii), the length condition | `fwl_sym.py` (with `fwl.py`) | 2-FWL colour of `(c, v)` on two paths, goal `D` steps beyond `v` or `E` steps behind it, for `L ≤ 6`. |
| Theorem 1(ii), the window invariant | `fwl_invariant.py` (with `fwl.py`) | Equal invariants with equal censuses give equal colours, at every pair and round, for `L ≤ 5`. |
| Proposition 3 | `set_characters.py` (with `set_ref.py`) | The 81-channel form of exponential Newton against brute force over all pairs on SET deals of 8 to 64 cards. |
| Proposition 4, hardness | `scc_gadget.py` | The DFS-order gadget against the CLRS-30 library's own `strongly_connected_components`, 400 random digraphs. |

## Install

Python 3.12 or later with `numpy` (2.5.4) and `scipy` (1.18.1). `scc_gadget.py` also needs the CLRS-30 library,
`pip install dm-clrs==2.0.3`.

## Commands and expected output

### `python -I fwl_sym.py 1 6`

Same colour at `(D, E) = (2^L, 2^L + 1)`; different colours at `(2^L - 1, 2^L - 1)`, `(2^L - 1, 2^L)` and
`(2^L, 2^L)`, for every `L ≤ 6`. About five minutes of CPU time, most of it at `L = 6`.

```
L=1: D=1,E=1:diff  D=1,E=2:diff  D=2,E=2:diff  D=2,E=3:same  D=3,E=3:same  D=3,E=4:same
L=2: D=3,E=3:diff  D=3,E=4:diff  D=4,E=4:diff  D=4,E=5:same  D=5,E=5:same  D=5,E=6:same
L=3: D=7,E=7:diff  D=7,E=8:diff  D=8,E=8:diff  D=8,E=9:same  D=9,E=9:same  D=9,E=10:same
L=4: D=15,E=15:diff  D=15,E=16:diff  D=16,E=16:diff  D=16,E=17:same  D=17,E=17:same  D=17,E=18:same
L=5: D=31,E=31:diff  D=31,E=32:diff  D=32,E=32:diff  D=32,E=33:same  D=33,E=33:same  D=33,E=34:same
L=6: D=63,E=63:diff  D=63,E=64:diff  D=64,E=64:diff  D=64,E=65:same  D=65,E=65:same  D=65,E=66:same
```

### `python -I fwl_invariant.py 1 5`

For each `L`: the six instances of `fwl_sym.py` checked at every pair of both graphs and every round (soundness and
exactness hold in all of them, and the observed colour of `(c, v)` equals the predicted one); a grid of goal
distances; random marker placements (0 failures); and the path lengths at which `(c, v)` first gets the same colour,
which match the proof's range. About ten minutes of CPU time. A non-empty list after "rounds with census differing and colour not a function of F"
marks a round whose censuses differ, where the proof claims nothing. Full output: `fwl_invariant.txt`.

### `python -I set_characters.py`

Largest relative error of the partition function and of the output, over 40 deals per row:

```
SET deals, Z_3 phases, 81 channels against brute force (largest relative error of Z and of the output):
  n= 8 beta=1: Z 1.4e-15  output 1.7e-15
  n= 8 beta=2: Z 9.6e-15  output 2.1e-14
  n= 8 beta=4: Z 2.7e-12  output 5.6e-12
  n= 8 beta=8: Z 1.8e-07  output 2.5e-06
  n=16 beta=1: Z 8.5e-16  output 5.2e-15
  n=16 beta=2: Z 4.5e-15  output 1.2e-14
  n=16 beta=4: Z 9.4e-14  output 6.4e-13
  n=16 beta=8: Z 2.9e-11  output 3.1e-10
  n=32 beta=1: Z 5.8e-16  output 2.7e-15
  n=32 beta=2: Z 6.6e-16  output 3.1e-15
  n=32 beta=4: Z 1.9e-15  output 5.4e-15
  n=32 beta=8: Z 1.1e-15  output 6.1e-15
  n=64 beta=1: Z 6.4e-16  output 3.3e-15
  n=64 beta=2: Z 5.4e-16  output 5.9e-15
  n=64 beta=4: Z 1.0e-15  output 3.6e-15
  n=64 beta=8: Z 7.1e-16  output 8.6e-15
  largest over n = 8..64, beta <= 4: 5.6e-12
Categorical clocks against explicit convolution:
  n= 8 beta=1: 1.5e-15
  n= 8 beta=4: 2.3e-14
  n= 8 beta=8: 3.0e-15
  n=16 beta=1: 2.5e-15
  n=16 beta=4: 2.7e-15
  n=16 beta=8: 1.4e-15
  n=32 beta=1: 9.5e-16
  n=32 beta=4: 1.8e-15
  n=32 beta=8: 3.1e-15
  n=64 beta=1: 5.8e-15
  n=64 beta=4: 4.4e-15
  n=64 beta=8: 2.2e-15
```

Under a second. An optional argument writes the rows as JSON.

### `python -I scc_gadget.py`

```
gadget agrees on 400 of 400
```
