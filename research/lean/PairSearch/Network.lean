import PairSearch.Layer
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Data.Matrix.Mul
import Mathlib.Algebra.BigOperators.Fin

/-!
# The `L + 1`-layer construction for the real-valued layer of equation (pair)

Source: equation (pair) of the paper, and the proof of Theorem 1(i) in its Proofs for the weights.

One layer of equation (pair) on pair states `z_{uv} ∈ ℝ^D`, without the normalisation (`t = z`):
```
z_{uv} ← z_{uv} + σ(G z_{uv}) ⊙ (1/√n) ∑_k (A z_{uk}) ⊙ (B z_{kv}),
z_{uv} ← z_{uv} + MLP(z_{uv}),     MLP(x) = W₂ ReLU(W₁ x + b₁) + b₂,
```
with the logistic sigmoid `σ` (`Real.sigmoid`). The MLP reads the state after the triangle update, as in
a pre-norm block; the construction needs this.

The weights are the paper's. Channels are numbered `R, T, E, Γ, O, F = 0, …, 5`.
- `squareWeights s`: `G = 0`, so the gate is `σ(0) = 1/2`; `A` maps `R → R` and `T → T` with weight
  `2s`; `B` maps `R → R` and `T → T` with weight `1`; two ReLU units with bias `-1`,
  `h₁ = ReLU(R - 1)`, `h₂ = ReLU(T - 1)`, subtracted from `R` and `T`.
- `finalWeights s`: `G = 0`; `A` maps `Γ → O` with weight `2s`; `B` maps `T → O` with weight `1`; one
  ReLU unit `h = ReLU(O + E - 1)` added to `F` (the second unit of the width-2 MLP is zero).
- The read-out is `w · z_{cv} - 1/2` with `w` selecting channel `F`.

With `s = √n`, the gate, the scale and the weight multiply to `(1/2) · (1/√n) · 2√n = 1` on every
triangle term, so each layer is exactly the integer layer of `PairSearch/Layer.lean` on every integer
state (`squareWeights_layer`, `finalWeights_layer`). The weights depend on `n` only through `s = √n`
and not on the graph.

- `network_score_pos_iff`: the read-out after `L` squaring layers and the final layer is positive
  exactly when `v` is a child of `c` from which `g` can be reached in at most `2^L` steps.
- `pair_states_search`: the same, with the weights quantified before the graph: for every `n` and `L` there are weights
  (one width-2 ReLU MLP per layer, chosen before the graph) and a read-out vector that work for every
  graph on `n` vertices, every goal and every current vertex.
-/

namespace PairSearch

open Finset Matrix

variable {n : ℕ}

/-! ### The layer of equation (pair) -/

/-- The weights of one layer of equation (pair) with `D` channels and a ReLU MLP of width `H`
with biases. -/
structure Weights (D H : ℕ) where
  /-- The gate matrix `G`. -/
  G : Matrix (Fin D) (Fin D) ℝ
  /-- The left triangle map `A`, applied to `z_{uk}`. -/
  A : Matrix (Fin D) (Fin D) ℝ
  /-- The right triangle map `B`, applied to `z_{kv}`. -/
  B : Matrix (Fin D) (Fin D) ℝ
  /-- The MLP's first linear map. -/
  W₁ : Matrix (Fin H) (Fin D) ℝ
  /-- The MLP's hidden bias. -/
  b₁ : Fin H → ℝ
  /-- The MLP's second linear map. -/
  W₂ : Matrix (Fin D) (Fin H) ℝ
  /-- The MLP's output bias. -/
  b₂ : Fin D → ℝ

namespace Weights

variable {D H : ℕ}

/-- The ReLU MLP `x ↦ W₂ ReLU(W₁ x + b₁) + b₂`. -/
noncomputable def mlp (w : Weights D H) (x : Fin D → ℝ) : Fin D → ℝ :=
  w.W₂ *ᵥ (fun h => max ((w.W₁ *ᵥ x) h + w.b₁ h) 0) + w.b₂

/-- The triangle update of equation (pair) without normalisation:
`z_{uv} + σ(G z_{uv}) ⊙ (1/√n) ∑_k (A z_{uk}) ⊙ (B z_{kv})`. -/
noncomputable def triangle (w : Weights D H) (z : Fin n → Fin n → Fin D → ℝ) :
    Fin n → Fin n → Fin D → ℝ :=
  fun u v c => z u v c + Real.sigmoid ((w.G *ᵥ z u v) c) *
    ((Real.sqrt n)⁻¹ * ∑ k, (w.A *ᵥ z u k) c * (w.B *ᵥ z k v) c)

/-- One layer of equation (pair) without normalisation: the triangle update, then the residual MLP
on the updated state. -/
noncomputable def layer (w : Weights D H) (z : Fin n → Fin n → Fin D → ℝ) :
    Fin n → Fin n → Fin D → ℝ :=
  fun u v => w.triangle z u v + w.mlp (w.triangle z u v)

end Weights

/-! ### The paper's weights -/

/-- The squaring layer's weights, with `s = √n` in the paper. -/
noncomputable def squareWeights (s : ℝ) : Weights 6 2 where
  G := 0
  A := !![2 * s, 0, 0, 0, 0, 0;
          0, 2 * s, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0]
  B := !![1, 0, 0, 0, 0, 0;
          0, 1, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0]
  W₁ := !![1, 0, 0, 0, 0, 0;
           0, 1, 0, 0, 0, 0]
  b₁ := ![-1, -1]
  W₂ := !![-1, 0;
           0, -1;
           0, 0;
           0, 0;
           0, 0;
           0, 0]
  b₂ := 0

/-- The final layer's weights, with `s = √n` in the paper. -/
noncomputable def finalWeights (s : ℝ) : Weights 6 2 where
  G := 0
  A := !![0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 2 * s, 0, 0;
          0, 0, 0, 0, 0, 0]
  B := !![0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0;
          0, 1, 0, 0, 0, 0;
          0, 0, 0, 0, 0, 0]
  W₁ := !![0, 0, 1, 0, 1, 0;
           0, 0, 0, 0, 0, 0]
  b₁ := ![-1, 0]
  W₂ := !![0, 0;
           0, 0;
           0, 0;
           0, 0;
           0, 0;
           1, 0]
  b₂ := 0

/-- The read-out vector `w`, selecting channel `F`. -/
def readout : Fin 6 → ℝ := ![0, 0, 0, 0, 0, 1]

/-- The integer pair state as six real channels per pair, in the order `R, T, E, Γ, O, F`. -/
noncomputable def embed (z : PairState n) : Fin n → Fin n → Fin 6 → ℝ :=
  fun u v => ![(z.R u v : ℝ), z.T u v, z.E u v, z.Γ u v, z.O u v, z.F u v]

/-- `L` squaring layers and the final layer of equation (pair), with the paper's weights. -/
noncomputable def network (n L : ℕ) (z : Fin n → Fin n → Fin 6 → ℝ) : Fin n → Fin n → Fin 6 → ℝ :=
  (finalWeights (Real.sqrt n)).layer ((squareWeights (Real.sqrt n)).layer^[L] z)

/-! ### Each real layer is the integer layer -/

lemma sqrt_ne_zero_of_fin (u : Fin n) : Real.sqrt n ≠ 0 := by
  have : (0 : ℝ) < n := by exact_mod_cast u.pos
  exact (Real.sqrt_pos.2 this).ne'

/-- The gate `σ(0) = 1/2`, the scale `1/s` and the weight `2s` multiply to one on a triangle term
(in the form `simp` leaves it). -/
lemma gate_scale (s x : ℝ) (hs : s ≠ 0) : 2⁻¹ * (s⁻¹ * (2 * (s * x))) = x := by
  field_simp

/-- A squaring layer of equation (pair) with the paper's weights is the integer squaring layer. -/
theorem squareWeights_layer (z : PairState n) :
    (squareWeights (Real.sqrt n)).layer (embed z) = embed (squareLayer z) := by
  funext u v c
  have hs := sqrt_ne_zero_of_fin u
  fin_cases c <;>
    simp [Weights.layer, Weights.triangle, Weights.mlp, squareWeights, embed, squareLayer, clip,
      relu, Matrix.mulVec, dotProduct, Fin.sum_univ_six, mul_assoc, ← Finset.mul_sum,
      gate_scale _ _ hs, sub_eq_add_neg]

/-- The final layer of equation (pair) with the paper's weights is the integer final layer. -/
theorem finalWeights_layer (z : PairState n) :
    (finalWeights (Real.sqrt n)).layer (embed z) = embed (finalLayer z) := by
  funext u v c
  have hs := sqrt_ne_zero_of_fin u
  fin_cases c <;>
    simp [Weights.layer, Weights.triangle, Weights.mlp, finalWeights, embed, finalLayer,
      relu, Matrix.mulVec, dotProduct, Fin.sum_univ_six, mul_assoc, ← Finset.mul_sum,
      gate_scale _ _ hs, sub_eq_add_neg]
  -- channel `F`: the two sides add `O`, the triangle term and `E` in different orders
  congr 1
  ring

/-! ### Pair states search with `L + 1` layers -/

/-- `L` squaring layers of equation (pair) are `L` integer squaring layers. -/
theorem iterate_squareWeights_layer (L : ℕ) (z : PairState n) :
    (squareWeights (Real.sqrt n)).layer^[L] (embed z) = embed (squareLayer^[L] z) :=
  ((Function.Semiconj.iterate_right (f := embed) (fun z => (squareWeights_layer z).symm) L) z).symm

/-- The network of equation (pair) with the paper's weights is the integer construction. -/
theorem network_eq (A : Fin n → Fin n → Bool) (g : Fin n) (L : ℕ) :
    network n L (embed (init A g)) = embed (run A g L) := by
  rw [network, iterate_squareWeights_layer, finalWeights_layer, run]

/-- **Pair states search in logarithmic depth, `L + 1` layers, explicit weights.** After `L` squaring
layers and the final layer of equation (pair), with the paper's weights (which depend on `n` only
through `√n` and not on the graph), the read-out `w · z_{cv} - 1/2` is positive exactly when `v` is a
child of `c` from which the goal `g` can be reached in at most `2^L` steps. -/
theorem network_score_pos_iff (A : Fin n → Fin n → Bool) (g c v : Fin n) (L : ℕ) :
    0 < readout ⬝ᵥ network n L (embed (init A g)) c v - 1 / 2 ↔
      A c v = true ∧ ReachWithin A (2 ^ L) v g := by
  rw [network_eq, ← run_F_eq_one_iff]
  have hF : readout ⬝ᵥ embed (run A g L) c v = ((run A g L).F c v : ℝ) := by
    simp [readout, embed, dotProduct, Fin.sum_univ_six]
  rw [hF, run_F]
  by_cases h : A c v = true ∧ ReachWithin A (2 ^ L) v g
  · simp only [ind_pos h]
    norm_num
  · simp [ind_neg h]

/-- **Pair states search in logarithmic depth, `L + 1` layers.** For every number of vertices `n` and every
`L ≥ 0` there are weights for `L + 1` layers of equation (pair) (without the normalisation, ReLU MLPs of
width 2 with biases) and a read-out vector `w`, chosen before the graph, such that for every graph on
`n` vertices with adjacency `A`, every goal `g`, current vertex `c` and vertex `v`, the read-out
`w · z_{cv} - 1/2` is positive exactly when `v` is a child of `c` from which `g` can be reached in at
most `2^L` steps. The pair state starts from `embed (init A g)`: `R = [I + A > 0]`, `T = Rᵀ`, `E = A`,
`Γ_{uv} = [v = g]`, `O = F = 0`. The `L` squaring layers share their weights. -/
theorem pair_states_search (n L : ℕ) :
    ∃ (sq fin : Weights 6 2) (w : Fin 6 → ℝ),
      ∀ (A : Fin n → Fin n → Bool) (g c v : Fin n),
        0 < w ⬝ᵥ fin.layer (sq.layer^[L] (embed (init A g))) c v - 1 / 2 ↔
          A c v = true ∧ ReachWithin A (2 ^ L) v g :=
  ⟨squareWeights (Real.sqrt n), finalWeights (Real.sqrt n), readout,
    fun A g c v => network_score_pos_iff A g c v L⟩

end PairSearch
