import PairSearch.Reach
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The integer pair-state construction with `L + 1` layers

A first construction for Theorem 1(i) (depth law): `L + 1` layers decide reach within `2^L` steps.
`PairSearch/Sharp.lean` sharpens it to the form the paper states, `L` layers and reach within `2^L - 1`.

The pair state has six integer channels per ordered pair `(u, v)`: `R`, `T`, `E`, `Γ`, `O`, `F`. It
starts from `R = R^(0) = [I + A > 0]`, `T = (R^(0))ᵀ`, `E = A`, `Γ_{uv} = [v = g]`, `O = F = 0`.

- A **squaring layer** replaces `x = R_{uv} + ∑_k R_{uk} R_{kv}` by `x - ReLU(x - 1)`, and the same for
  `T` with `∑_k T_{uk} T_{kv}`; the other channels are unchanged.
- The **final layer** adds `∑_k Γ_{uk} T_{kv}` to `O` and then `ReLU(O_{uv} + E_{uv} - 1)` to `F`.

These are exactly the layers of equation (pair) with the paper's weights, read over the integers:
the gate `σ(0) = 1/2`, the scale `1/√n`, and the weight `2√n` of `A` multiply to exactly `1` on each
triangle term, `(1/2) · (1/√n) · 2√n = 1`. `PairSearch/Network.lean` proves this equality for the
real-valued layer with the sigmoid gate and the `√n` scale.

Main results:
- `iterate_squareLayer_R`, `iterate_squareLayer_T`: after `L` squaring layers, channel `R` is
  `[v reachable from u within 2^L steps]` and channel `T` is its transpose.
- `run_O`, `run_F`: after the final layer, `O_{cv} = [g reachable from v within 2^L steps]` and
  `F_{cv} = [A c v ∧ g reachable from v within 2^L steps]`.
- `run_F_eq_one_iff`: `F_{cv} = 1` iff `v` is a child of `c` from which `g` is reachable within `2^L`
  steps, and `F_{cv} = 0` otherwise (`run_F_eq_zero_iff`).
-/

namespace PairSearch

open Finset

variable {n : ℕ}

/-! ### The Iverson bracket and the threshold -/

open Classical in
/-- The Iverson bracket `[P]`: `1` if `P` holds, `0` otherwise. -/
noncomputable def ind (P : Prop) : ℤ := if P then 1 else 0

lemma ind_pos {P : Prop} (h : P) : ind P = 1 := by simp [ind, h]

lemma ind_neg {P : Prop} (h : ¬P) : ind P = 0 := by simp [ind, h]

lemma ind_nonneg (P : Prop) : 0 ≤ ind P := by
  by_cases h : P
  · simp [ind_pos h]
  · simp [ind_neg h]

lemma ind_mul_ind (P Q : Prop) : ind P * ind Q = ind (P ∧ Q) := by
  by_cases hP : P <;> by_cases hQ : Q <;> simp [ind_pos, ind_neg, hP, hQ]

lemma ind_congr {P Q : Prop} (h : P ↔ Q) : ind P = ind Q := by rw [propext h]

lemma ind_eq_one_iff {P : Prop} : ind P = 1 ↔ P := by
  by_cases h : P <;> simp [ind_pos, ind_neg, h]

lemma ind_eq_zero_iff {P : Prop} : ind P = 0 ↔ ¬P := by
  by_cases h : P <;> simp [ind_pos, ind_neg, h]

/-- ReLU on integers. -/
def relu (x : ℤ) : ℤ := max x 0

/-- The squaring layer's residual MLP on one channel: `x ↦ x - ReLU(x - 1)` (one ReLU unit with bias
`-1`, subtracted from the channel). -/
def clip (x : ℤ) : ℤ := x - relu (x - 1)

/-- For a non-negative integer, `x - ReLU(x - 1) = min(1, x) = [x ≥ 1]`. -/
lemma clip_of_nonneg {x : ℤ} (hx : 0 ≤ x) : clip x = if 1 ≤ x then 1 else 0 := by
  unfold clip relu
  rcases le_total (x - 1) 0 with h | h
  · rw [max_eq_right h]; split_ifs <;> omega
  · rw [max_eq_left h]; split_ifs <;> omega

lemma sum_ind_nonneg (Q : Fin n → Prop) : 0 ≤ ∑ k, ind (Q k) :=
  Finset.sum_nonneg fun k _ => ind_nonneg (Q k)

/-- A sum of brackets is at least one iff some bracket is one: "the sum over `k` asks whether any
middle vertex exists". -/
lemma one_le_sum_ind_iff (Q : Fin n → Prop) : 1 ≤ ∑ k, ind (Q k) ↔ ∃ k, Q k := by
  constructor
  · intro h
    by_contra hne
    push Not at hne
    have : ∑ k, ind (Q k) = 0 := Finset.sum_eq_zero fun k _ => ind_neg (hne k)
    omega
  · rintro ⟨k, hk⟩
    have := Finset.single_le_sum (f := fun k => ind (Q k)) (fun i _ => ind_nonneg (Q i))
      (Finset.mem_univ k)
    simp only [ind_pos hk] at this
    exact this

/-- The threshold step of the squaring layer: if the residual term `[P₀]` implies a middle vertex,
`x - ReLU(x - 1)` of `[P₀] + ∑_k [Q k]` is `[∃ k, Q k]`. -/
lemma clip_ind_add_sum {P₀ : Prop} (Q : Fin n → Prop) (h : P₀ → ∃ k, Q k) :
    clip (ind P₀ + ∑ k, ind (Q k)) = ind (∃ k, Q k) := by
  have h0 := ind_nonneg P₀
  have hs := sum_ind_nonneg Q
  rw [clip_of_nonneg (by omega)]
  by_cases hq : ∃ k, Q k
  · have := (one_le_sum_ind_iff Q).2 hq
    rw [ind_pos hq]
    split_ifs <;> omega
  · have hP : ¬P₀ := fun hp => hq (h hp)
    have := (one_le_sum_ind_iff Q).not.2 hq
    rw [ind_neg hP, ind_neg hq]
    split_ifs <;> omega

/-! ### The pair state and the two layers -/

/-- The pair state: six integer channels for every ordered pair `(u, v)` of vertices. -/
structure PairState (n : ℕ) where
  /-- Reach within `2^ℓ` steps, `R_{uv}`. -/
  R : Fin n → Fin n → ℤ
  /-- Its transpose, `T_{uv} = R_{vu}`. -/
  T : Fin n → Fin n → ℤ
  /-- The edges, `E = A`. -/
  E : Fin n → Fin n → ℤ
  /-- The goal marker, `Γ_{uv} = [v = g]`. -/
  Γ : Fin n → Fin n → ℤ
  /-- The join channel of the final layer. -/
  O : Fin n → Fin n → ℤ
  /-- The read-out channel. -/
  F : Fin n → Fin n → ℤ

/-- The initial state from the adjacency `A` and the goal `g`: `R = [I + A > 0]`, `T = Rᵀ`, `E = A`,
`Γ_{uv} = [v = g]`, `O = F = 0`. -/
noncomputable def init (A : Fin n → Fin n → Bool) (g : Fin n) : PairState n where
  R u v := ind (u = v ∨ A u v = true)
  T u v := ind (v = u ∨ A v u = true)
  E u v := ind (A u v = true)
  Γ _ v := ind (v = g)
  O _ _ := 0
  F _ _ := 0

/-- A squaring layer: the triangle term adds `∑_k R_{uk} R_{kv}` to `R` and `∑_k T_{uk} T_{kv}` to
`T`; the MLP then subtracts `ReLU(R_{uv} - 1)` from `R` and `ReLU(T_{uv} - 1)` from `T`. -/
def squareLayer (z : PairState n) : PairState n :=
  { z with
    R := fun u v => clip (z.R u v + ∑ k, z.R u k * z.R k v)
    T := fun u v => clip (z.T u v + ∑ k, z.T u k * z.T k v) }

/-- The final layer: the triangle term adds `∑_k Γ_{uk} T_{kv}` to `O`; the MLP then adds
`ReLU(O_{uv} + E_{uv} - 1)` to `F`. -/
def finalLayer (z : PairState n) : PairState n :=
  let O' : Fin n → Fin n → ℤ := fun u v => z.O u v + ∑ k, z.Γ u k * z.T k v
  { z with
    O := O'
    F := fun u v => z.F u v + relu (O' u v + z.E u v - 1) }

/-- `L` squaring layers followed by the final layer, from the initial state. -/
noncomputable def run (A : Fin n → Fin n → Bool) (g : Fin n) (L : ℕ) : PairState n :=
  finalLayer (squareLayer^[L] (init A g))

/-! ### One squaring layer squares a reflexive relation -/

/-- One squaring layer. If channel `R` holds a reflexive relation `P` and channel `T` its transpose,
then afterwards `R` holds `P ∘ P` (`∃ k, P u k ∧ P k v`) and `T` its transpose. -/
theorem squareLayer_spec {z : PairState n} {P : Fin n → Fin n → Prop} (hrefl : ∀ u, P u u)
    (hR : ∀ u v, z.R u v = ind (P u v)) (hT : ∀ u v, z.T u v = ind (P v u)) :
    (∀ u v, (squareLayer z).R u v = ind (∃ k, P u k ∧ P k v)) ∧
      (∀ u v, (squareLayer z).T u v = ind (∃ k, P v k ∧ P k u)) := by
  constructor
  · intro u v
    have hterm : ∀ k, z.R u k * z.R k v = ind (P u k ∧ P k v) := fun k => by
      rw [hR, hR, ind_mul_ind]
    simp only [squareLayer, hterm, hR u v]
    exact clip_ind_add_sum _ fun h => ⟨v, h, hrefl v⟩
  · intro u v
    have hterm : ∀ k, z.T u k * z.T k v = ind (P v k ∧ P k u) := fun k => by
      rw [hT, hT, mul_comm, ind_mul_ind]
    simp only [squareLayer, hterm, hT u v]
    exact clip_ind_add_sum _ fun h => ⟨u, h, hrefl u⟩

/-! ### After `L` squaring layers -/

section Iterate

variable (A : Fin n → Fin n → Bool) (g : Fin n)

/-- After `L` squaring layers, `R` and `T` hold reach within `2^L` steps and its transpose. -/
theorem iterate_squareLayer (L : ℕ) :
    (∀ u v, (squareLayer^[L] (init A g)).R u v = ind (ReachWithin A (2 ^ L) u v)) ∧
      (∀ u v, (squareLayer^[L] (init A g)).T u v = ind (ReachWithin A (2 ^ L) v u)) := by
  induction L with
  | zero =>
    refine ⟨fun u v => ?_, fun u v => ?_⟩ <;>
      simp only [Function.iterate_zero, id, init, pow_zero] <;>
      exact ind_congr reachWithin_one_iff.symm
  | succ L ih =>
    obtain ⟨hR, hT⟩ := squareLayer_spec (fun u => ReachWithin.refl (2 ^ L) u) ih.1 ih.2
    refine ⟨fun u v => ?_, fun u v => ?_⟩ <;> rw [Function.iterate_succ_apply']
    · rw [hR, ind_congr reachWithin_pow_succ_iff]
    · rw [hT, ind_congr reachWithin_pow_succ_iff]

/-- **Squaring layers, channel `R`.** After `L` squaring layers,
`R_{uv} = [v can be reached from u in at most 2^L steps]`. -/
theorem iterate_squareLayer_R (L : ℕ) (u v : Fin n) :
    (squareLayer^[L] (init A g)).R u v = ind (ReachWithin A (2 ^ L) u v) :=
  (iterate_squareLayer A g L).1 u v

/-- **Squaring layers, channel `T`.** After `L` squaring layers, `T` is the transpose of
`R`: `T_{uv} = [u can be reached from v in at most 2^L steps]`. -/
theorem iterate_squareLayer_T (L : ℕ) (u v : Fin n) :
    (squareLayer^[L] (init A g)).T u v = ind (ReachWithin A (2 ^ L) v u) :=
  (iterate_squareLayer A g L).2 u v

/-- Squaring layers leave the channels `E`, `Γ`, `O`, `F` unchanged. -/
theorem iterate_squareLayer_other (L : ℕ) :
    (squareLayer^[L] (init A g)).E = (init A g).E ∧ (squareLayer^[L] (init A g)).Γ = (init A g).Γ ∧
      (squareLayer^[L] (init A g)).O = (init A g).O ∧
        (squareLayer^[L] (init A g)).F = (init A g).F := by
  induction L with
  | zero => simp
  | succ L ih =>
    simp only [Function.iterate_succ_apply']
    exact ih

/-! ### The final layer -/

/-- **Final layer, channel `O`.** `O_{cv} = ∑_k Γ_{ck} T_{kv} = R_{vg}`:
`[g can be reached from v in at most 2^L steps]`. -/
theorem run_O (L : ℕ) (c v : Fin n) : (run A g L).O c v = ind (ReachWithin A (2 ^ L) v g) := by
  obtain ⟨-, hΓ, hO, -⟩ := iterate_squareLayer_other A g L
  simp only [run, finalLayer]
  rw [hO, hΓ]
  simp only [iterate_squareLayer_T]
  simp only [init, zero_add]
  rw [Finset.sum_eq_single g]
  · simp [ind_pos]
  · intro k _ hk
    rw [ind_neg hk, zero_mul]
  · simp

/-- **Final layer, channel `F`.**
`F_{cv} = ReLU(O_{cv} + E_{cv} - 1) = [A c v ∧ g can be reached from v in at most 2^L steps]`. -/
theorem run_F (L : ℕ) (c v : Fin n) :
    (run A g L).F c v = ind (A c v = true ∧ ReachWithin A (2 ^ L) v g) := by
  obtain ⟨hE, -, -, hF⟩ := iterate_squareLayer_other A g L
  have hO := run_O A g L c v
  simp only [run, finalLayer] at hO ⊢
  rw [hO, hF, hE]
  simp only [init, zero_add, relu]
  by_cases ha : A c v = true <;> by_cases hr : ReachWithin A (2 ^ L) v g <;>
    simp [ind_pos, ind_neg, ha, hr]

/-- **`L + 1` layers, integer form.** After `L` squaring layers and the final layer, `F_{cv} = 1` exactly
when `v` is a child of `c` from which `g` can be reached in at most `2^L` steps. -/
theorem run_F_eq_one_iff (L : ℕ) (c v : Fin n) :
    (run A g L).F c v = 1 ↔ A c v = true ∧ ReachWithin A (2 ^ L) v g := by
  rw [run_F, ind_eq_one_iff]

/-- ... and `F_{cv} = 0` otherwise. -/
theorem run_F_eq_zero_iff (L : ℕ) (c v : Fin n) :
    (run A g L).F c v = 0 ↔ ¬(A c v = true ∧ ReachWithin A (2 ^ L) v g) := by
  rw [run_F, ind_eq_zero_iff]

end Iterate

end PairSearch
