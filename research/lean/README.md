# Lean proofs

One Lake project that builds every Lean declaration the paper names: Theorem 1(i) (depth law), Theorem 2 (size law)
and Proposition 2 (characters count set pairs at linear cost). No `sorry`, no axioms beyond Lean's three standard
ones.

## Toolchain

- Lean `leanprover/lean4:v4.35.0` (`lean-toolchain`), installed with [elan](https://github.com/leanprover/elan).
- Mathlib `v4.35.0` (`lakefile.toml`; commit pinned in `lake-manifest.json`).

## Build

```sh
cd research/lean
lake exe cache get   # download Mathlib's prebuilt files (optional, saves building Mathlib)
lake build
```

`lake build` compiles the three libraries and `Check.lean`, which prints `#check` and `#print axioms` for each
declaration below. The build ends with `Build completed successfully`. The only warnings are Lake's notes that the
imported Mathlib files are designed for the module system.

## Files

| File | Contents |
|---|---|
| `PairSearch/Reach.lean` | Walks, reach within `m` steps, and the splitting lemma behind equation (square). |
| `PairSearch/Layer.lean` | The integer pair-state layers of a first construction with `L + 1` layers. |
| `PairSearch/Network.lean` | The real-valued layer of equation (pair) (sigmoid gate, scale `1/√n`, ReLU MLP with biases) with explicit weights. |
| `PairSearch/Sharp.lean` | Theorem 1(i): `L` layers decide whether the goal is within `2^L - 1` steps of `v`. |
| `Framework/Gating.lean` | Theorem 2: the size law, both halves and the equality case, and the softmax read-out. |
| `Newton/Characters.lean` | Proposition 2: the general count over a finite abelian group, Newton's identities, and SET. |
| `Newton/Precision.lean` | How far power-sum errors move `e_k` and the count (not named in the paper). |
| `Check.lean` | `#check` and `#print axioms` for every named declaration, and a worked instance of Theorem 1(i). |

## Declarations named in the paper

Every declaration below depends on `propext`, `Classical.choice` and `Quot.sound` only (output of `#print axioms` in
`Check.lean`).

| Paper | Declaration | Statement |
|---|---|---|
| Theorem 1(i) | `PairSearch.pair_states_search_sharp` | For every `n` and `L ≥ 1`, weights for `L` layers of equation (pair) and a read-out, fixed before the graph, give a positive score at `(c, v)` exactly when `v` is a child of `c` from which the goal is reached within `2^L - 1` steps. |
| Theorem 1(i) | `PairSearch.pair_states_search_looped` | The same with the weights chosen before `L`: one layer repeated `L - 1` times and a last layer. |
| Theorem 2 | `Framework.size_law_iff` | For `|F| ≥ 2`, `Δ > 0`, `β ≥ 0` and scores free within the margin, some threshold on `Z` decides existence on every input iff `log |F| ≤ β Δ`. |
| Theorem 2 | `Framework.size_law_equality` | At `β Δ = log |F|`, the strict threshold `Z > 1` decides existence, is the only strict threshold that does, and no non-strict threshold does. |
| Theorem 2 | `Framework.gated_existence` | If `|F| e^{-βΔ} < 1`, some witness is true iff `Z ≥ 1`. |
| Theorem 2 | `Framework.gated_existence_log` | The same under `log |F| < β Δ`. |
| Theorem 2 | `Framework.background_le` | With no true witness, `Z ≤ |F| e^{-βΔ}`. |
| Theorem 2 | `Framework.tight` | With every score at `a - Δ`, `Z = |F| e^{-βΔ}`. |
| Theorem 2 | `Framework.softmax_existence` | With a true witness and `|N| e^{-βΔ} ≤ 1`, the softmax weight on true witnesses is at least `1/2`. |
| Proposition 2 | `Newton.set_count` | 40 characters of `(ℤ/3)⁴` give `81 N = C(n, 2) + 2 Re Σ_c χ_c(q) e_{2,c}` for every card in hand and table. |
| Proposition 2 | `Newton.newton_identities` | Newton's identities, `k e_k = (-1)^(k+1) Σ_{i<k} (-1)^i e_i p_{k-i}`. |
| Proposition 2 | `Newton.count_eq` | For any finite abelian group `G` and subset size `k`, `|G| · #{S : |S| = k, q + Σ_S a = 0} = Σ_ψ ψ(q) e_k(ψ ∘ a)`. |

Not formalised, and proved in the paper: Theorem 1(ii), the corollaries after Theorem 2, and Propositions 1, 3 and 4.
Theorem 1(ii) and Propositions 3 and 4 are also checked numerically in `research/checks/`.
