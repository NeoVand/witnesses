import PairSearch.Network

/-!
# Theorem 1(i), depth law: `L` layers decide the goal within `2^L` steps of `c`

The construction of `PairSearch/Network.lean` (`pair_states_search`) uses `L + 1` layers to decide whether `v` is a child of `c`
from which `g` is reachable within `2^L` steps. This file proves the sharper count: `L ≥ 1` layers decide
whether `v` is a child of `c` from which `g` is reachable within `2^L - 1` steps, that is, whether the goal
is within `2^L` steps of `c` through `v`.

The pair state has five channels per ordered pair `(u, v)`: `R, T, E, H, F = 0, …, 4`. It starts from
`R = R^(0) = [I + A > 0]`, `T = (R^(0))ᵀ`, `E = A`, `H_{uv} = [v = g]`, `F = 0`.

- Every layer: the triangle term adds `∑_k R_{uk} R_{kv}` to `R`, `∑_k T_{uk} T_{kv}` to `T` and
  `∑_k H_{uk} T_{kv}` to `H`; three ReLU units with bias `-1` then subtract `ReLU(x - 1)` from each of
  `R`, `T`, `H`, so each becomes `min(1, x)`.
- The last layer has one more ReLU unit, `ReLU(x_H + (n + 1)(E - 1))`, added to `F`, where
  `x_H = H_{uv} + ∑_k H_{uk} T_{kv}` is the value of `H` before the threshold (`x_H ≤ n + 1`).
- After `m` layers, `H_{uv} = [v reaches g within 2^m - 1 steps]`, because a walk of at most
  `2^(m+1) - 1` steps splits into at most `2^m` steps followed by at most `2^m - 1` steps
  (`reachWithin_pow_succ_sub_one_iff`).

Main results:
- `iterate_sharpLayer`: what each channel holds after `m` layers.
- `runSharp_F_pos_iff`: after `L ≥ 1` layers, `F_{cv} ≥ 1` iff `A c v` and `g` is reachable from `v`
  within `2^L - 1` steps; otherwise `F_{cv} = 0` (`runSharp_F_nonneg`).
- `sharpWeights_layer`, `sharpLastWeights_layer`: the real layer of equation (pair) with these weights is
  the integer layer.
- `pair_states_search_sharp`: the theorem with the weights quantified before the graph.
-/

namespace PairSearch

open Finset Matrix

variable {n : ℕ}

/-! ### Reach within `2^(m+1) - 1` steps -/

section Reach

variable {A : Fin n → Fin n → Bool}

/-- Reach within zero steps is equality. -/
theorem reachWithin_zero_iff {u v : Fin n} : ReachWithin A 0 u v ↔ u = v := by
  constructor
  · rintro ⟨p, hp⟩
    cases p with
    | nil => rfl
    | cons h p => simp at hp
  · rintro rfl
    exact ReachWithin.refl 0 u

/-- **The sharpened splitting step.** `v` is reachable from `u` within `2^(m+1) - 1` steps iff some middle
vertex `k` is reachable from `u` within `2^m` steps and reaches `v` within `2^m - 1` steps, because
`2^(m+1) - 1 = 2^m + (2^m - 1)`. -/
theorem reachWithin_pow_succ_sub_one_iff {m : ℕ} {u v : Fin n} :
    ReachWithin A (2 ^ (m + 1) - 1) u v ↔
      ∃ k, ReachWithin A (2 ^ m) u k ∧ ReachWithin A (2 ^ m - 1) k v := by
  have h1 : 1 ≤ 2 ^ m := Nat.one_le_two_pow
  have h : 2 ^ (m + 1) - 1 = 2 ^ m + (2 ^ m - 1) := by rw [pow_succ]; omega
  rw [h]
  exact reachWithin_add_iff

end Reach

/-! ### Integer facts used by the last layer -/

lemma ind_le_one (P : Prop) : ind P ≤ 1 := by
  by_cases h : P
  · simp [ind_pos h]
  · simp [ind_neg h]

/-- A sum of `n` brackets is at most `n`. -/
lemma sum_ind_le_card (Q : Fin n → Prop) : ∑ k, ind (Q k) ≤ n := by
  calc ∑ k, ind (Q k) ≤ ∑ _k : Fin n, (1 : ℤ) := Finset.sum_le_sum fun k _ => ind_le_one (Q k)
    _ = n := by simp

/-- For a non-negative integer, `x ≥ 1` iff the threshold `x - ReLU(x - 1)` is `1`. -/
lemma one_le_iff_clip_eq_one {x : ℤ} (hx : 0 ≤ x) : 1 ≤ x ↔ clip x = 1 := by
  rw [clip_of_nonneg hx]
  split_ifs with h <;> simp [h]

/-- The last layer's extra unit `ReLU(x + N (E - 1))` with `E = [P]` and `0 ≤ x ≤ N`: it is `0` when
`E = 0` and `x` when `E = 1`, so it is at least `1` iff `P` and `x ≥ 1`. -/
lemma one_le_relu_gate {x N : ℤ} (P : Prop) (hx0 : 0 ≤ x) (hxN : x ≤ N) :
    1 ≤ relu (x + N * (ind P - 1)) ↔ P ∧ 1 ≤ x := by
  by_cases hP : P
  · rw [ind_pos hP]
    simp only [relu, sub_self, mul_zero, add_zero, le_max_iff, hP, true_and]
    omega
  · rw [ind_neg hP]
    simp only [relu, le_max_iff, hP, false_and, iff_false]
    omega

/-- An integer `x` has `x - 1/2 > 0` over the reals iff `x ≥ 1`. -/
lemma int_sub_half_pos_iff (x : ℤ) : 0 < (x : ℝ) - 1 / 2 ↔ 1 ≤ x := by
  constructor
  · intro h
    by_contra h'
    have hx : x ≤ 0 := by omega
    have : (x : ℝ) ≤ 0 := by exact_mod_cast hx
    linarith
  · intro h
    have : (1 : ℝ) ≤ x := by exact_mod_cast h
    linarith

/-! ### The five-channel pair state and its layers -/

/-- The pair state of the sharpened construction: five integer channels per ordered pair. -/
structure SharpState (n : ℕ) where
  /-- Reach within `2^m` steps, `R_{uv}`. -/
  R : Fin n → Fin n → ℤ
  /-- Its transpose, `T_{uv} = R_{vu}`. -/
  T : Fin n → Fin n → ℤ
  /-- The edges, `E = A`. -/
  E : Fin n → Fin n → ℤ
  /-- `H_{uv} = [v reaches g within 2^m - 1 steps]`. -/
  H : Fin n → Fin n → ℤ
  /-- The read-out channel. -/
  F : Fin n → Fin n → ℤ

/-- The initial state from the adjacency `A` and the goal `g`: `R = [I + A > 0]`, `T = Rᵀ`, `E = A`,
`H_{uv} = [v = g]`, `F = 0`. -/
noncomputable def initSharp (A : Fin n → Fin n → Bool) (g : Fin n) : SharpState n where
  R u v := ind (u = v ∨ A u v = true)
  T u v := ind (v = u ∨ A v u = true)
  E u v := ind (A u v = true)
  H _ v := ind (v = g)
  F _ _ := 0

/-- The value of channel `H` after the triangle term and before the threshold,
`x_H = H_{uv} + ∑_k H_{uk} T_{kv}`. -/
def SharpState.xH (z : SharpState n) (u v : Fin n) : ℤ := z.H u v + ∑ k, z.H u k * z.T k v

/-- The channels `R, T, E` and `F` as a six-channel state of the `L + 1`-layer construction (with `Γ = O = 0`),
so that the squaring step of `Layer.lean` can be reused. -/
def SharpState.toPair (z : SharpState n) : PairState n where
  R := z.R
  T := z.T
  E := z.E
  Γ _ _ := 0
  O _ _ := 0
  F := z.F

/-- A layer: the triangle term adds `∑_k R_{uk} R_{kv}` to `R`, `∑_k T_{uk} T_{kv}` to `T` and
`∑_k H_{uk} T_{kv}` to `H`; the MLP then subtracts `ReLU(x - 1)` from each of the three. -/
def sharpLayer (z : SharpState n) : SharpState n where
  R u v := clip (z.R u v + ∑ k, z.R u k * z.R k v)
  T u v := clip (z.T u v + ∑ k, z.T u k * z.T k v)
  E := z.E
  H u v := clip (z.xH u v)
  F := z.F

/-- The last layer: the same layer, and one more ReLU unit adds `ReLU(x_H + (n + 1)(E - 1))` to `F`. -/
def sharpLastLayer (z : SharpState n) : SharpState n where
  R := (sharpLayer z).R
  T := (sharpLayer z).T
  E := z.E
  H := (sharpLayer z).H
  F u v := z.F u v + relu (z.xH u v + ((n : ℤ) + 1) * (z.E u v - 1))

/-- `L` layers from the initial state: `L - 1` layers, then the last layer. -/
noncomputable def runSharp (A : Fin n → Fin n → Bool) (g : Fin n) (L : ℕ) : SharpState n :=
  sharpLastLayer (sharpLayer^[L - 1] (initSharp A g))

/-! ### One layer -/

/-- `x_H` as a sum of brackets, when `T` holds the transpose of `P` and `H_{uv} = [Q v]`. -/
lemma xH_eq {z : SharpState n} {P : Fin n → Fin n → Prop} {Q : Fin n → Prop}
    (hT : ∀ u v, z.T u v = ind (P v u)) (hH : ∀ u v, z.H u v = ind (Q v)) (u v : Fin n) :
    z.xH u v = ind (Q v) + ∑ k, ind (P v k ∧ Q k) := by
  simp only [SharpState.xH, hH, hT]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [mul_comm, ind_mul_ind]

/-- One layer. If `R` holds a reflexive relation `P`, `T` its transpose and `H_{uv} = [Q v]`, then
afterwards `R` holds `P ∘ P`, `T` its transpose, and `H_{uv} = [∃ k, P v k ∧ Q k]`. -/
theorem sharpLayer_spec {z : SharpState n} {P : Fin n → Fin n → Prop} {Q : Fin n → Prop}
    (hrefl : ∀ u, P u u) (hR : ∀ u v, z.R u v = ind (P u v)) (hT : ∀ u v, z.T u v = ind (P v u))
    (hH : ∀ u v, z.H u v = ind (Q v)) :
    (∀ u v, (sharpLayer z).R u v = ind (∃ k, P u k ∧ P k v)) ∧
      (∀ u v, (sharpLayer z).T u v = ind (∃ k, P v k ∧ P k u)) ∧
        (∀ u v, (sharpLayer z).H u v = ind (∃ k, P v k ∧ Q k)) := by
  obtain ⟨hR', hT'⟩ := squareLayer_spec (z := z.toPair) hrefl hR hT
  refine ⟨hR', hT', fun u v => ?_⟩
  show clip (z.xH u v) = _
  rw [xH_eq hT hH]
  exact clip_ind_add_sum _ fun h => ⟨v, hrefl v, h⟩

/-! ### After `m` layers -/

section Iterate

variable (A : Fin n → Fin n → Bool) (g : Fin n)

/-- After `m` layers, `R` holds reach within `2^m` steps, `T` its transpose, and
`H_{uv} = [g reachable from v within 2^m - 1 steps]`; `E` and `F` are unchanged. -/
theorem iterate_sharpLayer (m : ℕ) :
    (∀ u v, (sharpLayer^[m] (initSharp A g)).R u v = ind (ReachWithin A (2 ^ m) u v)) ∧
      (∀ u v, (sharpLayer^[m] (initSharp A g)).T u v = ind (ReachWithin A (2 ^ m) v u)) ∧
        (∀ u v, (sharpLayer^[m] (initSharp A g)).H u v = ind (ReachWithin A (2 ^ m - 1) v g)) ∧
          (sharpLayer^[m] (initSharp A g)).E = (initSharp A g).E ∧
            (sharpLayer^[m] (initSharp A g)).F = (initSharp A g).F := by
  induction m with
  | zero =>
    refine ⟨fun u v => ?_, fun u v => ?_, fun u v => ?_, rfl, rfl⟩
    · simp only [Function.iterate_zero, id, initSharp, pow_zero]
      exact ind_congr reachWithin_one_iff.symm
    · simp only [Function.iterate_zero, id, initSharp, pow_zero]
      exact ind_congr reachWithin_one_iff.symm
    · simp only [Function.iterate_zero, id, initSharp, pow_zero, Nat.sub_self]
      exact ind_congr reachWithin_zero_iff.symm
  | succ m ih =>
    obtain ⟨hR, hT, hH, hE, hF⟩ := ih
    obtain ⟨hR', hT', hH'⟩ := sharpLayer_spec (Q := fun v => ReachWithin A (2 ^ m - 1) v g)
      (fun u => ReachWithin.refl (2 ^ m) u) hR hT hH
    refine ⟨fun u v => ?_, fun u v => ?_, fun u v => ?_, ?_, ?_⟩ <;>
      rw [Function.iterate_succ_apply']
    · rw [hR', ind_congr reachWithin_pow_succ_iff]
    · rw [hT', ind_congr reachWithin_pow_succ_iff]
    · rw [hH', ind_congr reachWithin_pow_succ_sub_one_iff]
    · exact hE
    · exact hF

/-- **Theorem 1(i), channel `H`.** After `m` layers, `H_{uv} = [g can be reached from v in at
most 2^m - 1 steps]`. -/
theorem iterate_sharpLayer_H (m : ℕ) (u v : Fin n) :
    (sharpLayer^[m] (initSharp A g)).H u v = ind (ReachWithin A (2 ^ m - 1) v g) :=
  (iterate_sharpLayer A g m).2.2.1 u v

/-! ### The last layer -/

/-- The read-out channel after `m` layers and the last layer: `F_{cv} = ReLU(x_H + (n + 1)(E_{cv} - 1))`
is non-negative, and at least `1` exactly when `A c v` and `g` is reachable from `v` within
`2^(m+1) - 1` steps. -/
theorem lastLayer_F (m : ℕ) (c v : Fin n) :
    0 ≤ (sharpLastLayer (sharpLayer^[m] (initSharp A g))).F c v ∧
      (1 ≤ (sharpLastLayer (sharpLayer^[m] (initSharp A g))).F c v ↔
        A c v = true ∧ ReachWithin A (2 ^ (m + 1) - 1) v g) := by
  obtain ⟨-, hT, hH, hE, hF⟩ := iterate_sharpLayer A g m
  set z := sharpLayer^[m] (initSharp A g)
  have hx := xH_eq (Q := fun v => ReachWithin A (2 ^ m - 1) v g) hT hH c v
  -- `0 ≤ x_H ≤ n + 1`
  have hx0 : 0 ≤ z.xH c v := by
    rw [hx]
    exact add_nonneg (ind_nonneg _) (sum_ind_nonneg _)
  have hxN : z.xH c v ≤ (n : ℤ) + 1 := by
    rw [hx]
    have := ind_le_one (ReachWithin A (2 ^ m - 1) v g)
    have := sum_ind_le_card fun k => ReachWithin A (2 ^ m) v k ∧ ReachWithin A (2 ^ m - 1) k g
    omega
  -- `x_H ≥ 1` iff `g` is reachable from `v` within `2^(m+1) - 1` steps
  have hx1 : 1 ≤ z.xH c v ↔ ReachWithin A (2 ^ (m + 1) - 1) v g := by
    rw [one_le_iff_clip_eq_one hx0, hx,
      clip_ind_add_sum _ fun h => ⟨v, ReachWithin.refl _ v, h⟩, ind_eq_one_iff,
      reachWithin_pow_succ_sub_one_iff]
  have hFz : z.F c v = 0 := by rw [hF]; rfl
  have hEz : z.E c v = ind (A c v = true) := by rw [hE]; rfl
  simp only [sharpLastLayer, hFz, hEz, zero_add]
  refine ⟨le_max_right _ _, ?_⟩
  rw [one_le_relu_gate _ hx0 hxN, hx1]

/-- **Theorem 1(i) (integer form).** After `L ≥ 1` layers, `F_{cv} ≥ 1` exactly when `v` is a
child of `c` from which `g` can be reached in at most `2^L - 1` steps. -/
theorem runSharp_F_pos_iff (L : ℕ) (hL : 1 ≤ L) (c v : Fin n) :
    1 ≤ (runSharp A g L).F c v ↔ A c v = true ∧ ReachWithin A (2 ^ L - 1) v g := by
  obtain ⟨m, rfl⟩ : ∃ m, L = m + 1 := ⟨L - 1, by omega⟩
  exact (lastLayer_F A g m c v).2

/-- ... and `F_{cv} ≥ 0` always, so `F_{cv} = 0` otherwise (`F` is an integer). -/
theorem runSharp_F_nonneg (L : ℕ) (hL : 1 ≤ L) (c v : Fin n) : 0 ≤ (runSharp A g L).F c v := by
  obtain ⟨m, rfl⟩ : ∃ m, L = m + 1 := ⟨L - 1, by omega⟩
  exact (lastLayer_F A g m c v).1

end Iterate

/-! ### The real layer of equation (pair) with explicit weights -/

/-- The weights of every layer but the last, with `s = √n`. Channels `R, T, E, H, F = 0, …, 4`. `G = 0`;
`A` maps `R → R`, `T → T`, `H → H` with weight `2s`; `B` maps `R → R`, `T → T`, `T → H` with weight `1`;
three ReLU units with bias `-1`, `ReLU(R - 1)`, `ReLU(T - 1)`, `ReLU(H - 1)`, subtracted from `R`, `T`,
`H`; the fourth unit is zero. -/
noncomputable def sharpWeights (s : ℝ) : Weights 5 4 where
  G := 0
  A := !![2 * s, 0, 0, 0, 0;
          0, 2 * s, 0, 0, 0;
          0, 0, 0, 0, 0;
          0, 0, 0, 2 * s, 0;
          0, 0, 0, 0, 0]
  B := !![1, 0, 0, 0, 0;
          0, 1, 0, 0, 0;
          0, 0, 0, 0, 0;
          0, 1, 0, 0, 0;
          0, 0, 0, 0, 0]
  W₁ := !![1, 0, 0, 0, 0;
           0, 1, 0, 0, 0;
           0, 0, 0, 1, 0;
           0, 0, 0, 0, 0]
  b₁ := ![-1, -1, -1, 0]
  W₂ := !![-1, 0, 0, 0;
           0, -1, 0, 0;
           0, 0, 0, 0;
           0, 0, -1, 0;
           0, 0, 0, 0]
  b₂ := 0

/-- The last layer's weights, with `s = √n` and `N = n + 1`: the weights of `sharpWeights s`, and a fourth
ReLU unit `ReLU(H + N E - N)` added to `F`. -/
noncomputable def sharpLastWeights (s N : ℝ) : Weights 5 4 where
  G := 0
  A := !![2 * s, 0, 0, 0, 0;
          0, 2 * s, 0, 0, 0;
          0, 0, 0, 0, 0;
          0, 0, 0, 2 * s, 0;
          0, 0, 0, 0, 0]
  B := !![1, 0, 0, 0, 0;
          0, 1, 0, 0, 0;
          0, 0, 0, 0, 0;
          0, 1, 0, 0, 0;
          0, 0, 0, 0, 0]
  W₁ := !![1, 0, 0, 0, 0;
           0, 1, 0, 0, 0;
           0, 0, 0, 1, 0;
           0, 0, N, 1, 0]
  b₁ := ![-1, -1, -1, -N]
  W₂ := !![-1, 0, 0, 0;
           0, -1, 0, 0;
           0, 0, 0, 0;
           0, 0, -1, 0;
           0, 0, 0, 1]
  b₂ := 0

/-- The read-out vector, selecting channel `F`. -/
def readoutSharp : Fin 5 → ℝ := ![0, 0, 0, 0, 1]

/-- The integer pair state as five real channels per pair, in the order `R, T, E, H, F`. -/
noncomputable def embedSharp (z : SharpState n) : Fin n → Fin n → Fin 5 → ℝ :=
  fun u v => ![(z.R u v : ℝ), z.T u v, z.E u v, z.H u v, z.F u v]

/-- `L` layers of equation (pair) with these weights: `L - 1` layers with `sharpWeights √n`, then one
with `sharpLastWeights √n (n + 1)`. -/
noncomputable def sharpNetwork (n L : ℕ) (z : Fin n → Fin n → Fin 5 → ℝ) :
    Fin n → Fin n → Fin 5 → ℝ :=
  (sharpLastWeights (Real.sqrt n) (n + 1)).layer ((sharpWeights (Real.sqrt n)).layer^[L - 1] z)

/-- A layer of equation (pair) with `sharpWeights √n` is the integer layer `sharpLayer`. -/
theorem sharpWeights_layer (z : SharpState n) :
    (sharpWeights (Real.sqrt n)).layer (embedSharp z) = embedSharp (sharpLayer z) := by
  funext u v c
  have hs := sqrt_ne_zero_of_fin u
  fin_cases c <;>
    simp [Weights.layer, Weights.triangle, Weights.mlp, sharpWeights, embedSharp, sharpLayer,
      SharpState.xH, clip, relu, Matrix.mulVec, dotProduct, Fin.sum_univ_five, Fin.sum_univ_four,
      mul_assoc, ← Finset.mul_sum, gate_scale _ _ hs, sub_eq_add_neg]

/-- The last layer of equation (pair) with `sharpLastWeights √n (n + 1)` is the integer layer
`sharpLastLayer`. -/
theorem sharpLastWeights_layer (z : SharpState n) :
    (sharpLastWeights (Real.sqrt n) (n + 1)).layer (embedSharp z) = embedSharp (sharpLastLayer z) := by
  funext u v c
  have hs := sqrt_ne_zero_of_fin u
  fin_cases c <;>
    simp [Weights.layer, Weights.triangle, Weights.mlp, sharpLastWeights, embedSharp,
      sharpLastLayer, sharpLayer, SharpState.xH, clip, relu, Matrix.mulVec, dotProduct,
      Fin.sum_univ_five, Fin.sum_univ_four, mul_assoc, ← Finset.mul_sum, gate_scale _ _ hs,
      sub_eq_add_neg]
  congr 1
  ring

/-! ### Theorem 1(i) -/

/-- `m` layers of equation (pair) with `sharpWeights √n` are `m` integer layers. -/
theorem iterate_sharpWeights_layer (m : ℕ) (z : SharpState n) :
    (sharpWeights (Real.sqrt n)).layer^[m] (embedSharp z) = embedSharp (sharpLayer^[m] z) :=
  ((Function.Semiconj.iterate_right (f := embedSharp)
    (fun z => (sharpWeights_layer z).symm) m) z).symm

/-- The network of equation (pair) with these weights is the integer construction. -/
theorem sharpNetwork_eq (A : Fin n → Fin n → Bool) (g : Fin n) (L : ℕ) :
    sharpNetwork n L (embedSharp (initSharp A g)) = embedSharp (runSharp A g L) := by
  rw [sharpNetwork, iterate_sharpWeights_layer, sharpLastWeights_layer, runSharp]

/-- **Theorem 1(i), explicit weights.** After `L ≥ 1` layers of equation (pair) with the weights
`sharpWeights √n` (`L - 1` times) and `sharpLastWeights √n (n + 1)`, which depend on `n` and not on the
graph, the read-out `w · z_{cv} - 1/2` is positive exactly when `v` is a child of `c` from which the goal
`g` can be reached in at most `2^L - 1` steps. -/
theorem sharp_network_score_pos_iff (A : Fin n → Fin n → Bool) (g c v : Fin n) (L : ℕ)
    (hL : 1 ≤ L) :
    0 < readoutSharp ⬝ᵥ sharpNetwork n L (embedSharp (initSharp A g)) c v - 1 / 2 ↔
      A c v = true ∧ ReachWithin A (2 ^ L - 1) v g := by
  rw [sharpNetwork_eq]
  have hF : readoutSharp ⬝ᵥ embedSharp (runSharp A g L) c v = ((runSharp A g L).F c v : ℝ) := by
    simp [readoutSharp, embedSharp, dotProduct, Fin.sum_univ_five]
  rw [hF, int_sub_half_pos_iff, runSharp_F_pos_iff A g L hL]

/-- **Theorem 1(i).** For every number of vertices `n` and every `L ≥ 1` there are weights for `L`
layers of equation (pair) (without the normalisation; gate `σ(0) = 1/2`, scale `1/√n`, ReLU MLPs of width
4 with biases) and a read-out vector `w`, chosen before the graph, such that for every graph on `n`
vertices with adjacency `A`, every goal `g`, current vertex `c` and vertex `v`, the read-out
`w · z_{cv} - 1/2` is positive exactly when `v` is a child of `c` from which `g` can be reached in at most
`2^L - 1` steps: the goal is within `2^L` steps of `c` through `v`. The pair state starts from
`embedSharp (initSharp A g)`: `R = [I + A > 0]`, `T = Rᵀ`, `E = A`, `H_{uv} = [v = g]`, `F = 0`. The first
`L - 1` layers share their weights; the last layer has its own. -/
theorem pair_states_search_sharp (n L : ℕ) (hL : 1 ≤ L) :
    ∃ (sq fin : Weights 5 4) (w : Fin 5 → ℝ),
      ∀ (A : Fin n → Fin n → Bool) (g c v : Fin n),
        0 < w ⬝ᵥ fin.layer (sq.layer^[L - 1] (embedSharp (initSharp A g))) c v - 1 / 2 ↔
          A c v = true ∧ ReachWithin A (2 ^ L - 1) v g :=
  ⟨sharpWeights (Real.sqrt n), sharpLastWeights (Real.sqrt n) (n + 1), readoutSharp,
    fun A g c v => sharp_network_score_pos_iff A g c v L hL⟩

/-- **Theorem 1(i), one looped layer.** The weights of `pair_states_search_sharp` do not depend on `L`: one
layer `sq`, repeated `L - 1` times, and one last layer `fin` decide the next step within `2^L` steps for
every `L ≥ 1`. So running the same layer more times searches further, with nothing retrained. -/
theorem pair_states_search_looped (n : ℕ) :
    ∃ (sq fin : Weights 5 4) (w : Fin 5 → ℝ),
      ∀ L, 1 ≤ L → ∀ (A : Fin n → Fin n → Bool) (g c v : Fin n),
        0 < w ⬝ᵥ fin.layer (sq.layer^[L - 1] (embedSharp (initSharp A g))) c v - 1 / 2 ↔
          A c v = true ∧ ReachWithin A (2 ^ L - 1) v g :=
  ⟨sharpWeights (Real.sqrt n), sharpLastWeights (Real.sqrt n) (n + 1), readoutSharp,
    fun L hL A g c v => sharp_network_score_pos_iff A g c v L hL⟩

end PairSearch
