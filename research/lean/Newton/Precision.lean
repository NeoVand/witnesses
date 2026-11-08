import Newton.Characters

/-!
# How much precision the layer needs

The layer computes `e_k` from power sums by the recursion `newtonE`. If every power sum it uses is within
`η` of the true one, and the true power sums are bounded by `x` in modulus (for unit-modulus features,
`x = n`, the number of partners), then the computed `e_k` is within `H_k(x + η) - H_k(x)` of the true one,
where `H_k(x) = x (x + 1) ⋯ (x + k - 1) / k!` (`majorant_eq`). For the count of `count_eq` the error is at
most `(|G| - 1)/|G|` times this, so the count is recovered by rounding when `H_k(n + η) - H_k(n) < 1/2`.

- `majorant x k`: the recursion of Newton's identities with every power sum replaced by `x`.
- `majorant_eq`: `majorant x k = (∏_{i<k} (x + i)) / k!`.
- `norm_newtonE_le`: `‖newtonE p k‖ ≤ majorant x k` when `‖p l‖ ≤ x` for all `l`.
- `norm_newtonE_sub_le`: `‖newtonE p' k - newtonE p k‖ ≤ majorant (x + η) k - majorant x k`
  when moreover `‖p' l - p l‖ ≤ η` for all `l`.
- `count_error_le`: for the counting read-out of `count_odd`, feature errors `≤ δ` change `|G| · N` by at
  most `(|G| - 1) δ`.
-/

open Finset

namespace Newton

/-- `H_k(x)`: Newton's recursion with every power sum equal to `x`. -/
noncomputable def majorant (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | m + 1 => ((m : ℝ) + 1)⁻¹ * ∑ i : Fin (m + 1), majorant x i * x
decreasing_by exact i.isLt

lemma majorant_zero (x : ℝ) : majorant x 0 = 1 := by rw [majorant]

lemma majorant_succ (x : ℝ) (m : ℕ) :
    majorant x (m + 1) = ((m : ℝ) + 1)⁻¹ * ∑ i ∈ range (m + 1), majorant x i * x := by
  rw [majorant, Fin.sum_univ_eq_sum_range (fun i => majorant x i * x)]

/-- Hockey stick: `Σ_{i ≤ m} H_i(x) = H_m(x + 1)`, in closed form. -/
lemma sum_rising (x : ℝ) (m : ℕ) :
    ∑ i ∈ range (m + 1), (∏ j ∈ range i, (x + j)) / (i.factorial : ℝ) =
      (∏ j ∈ range m, (x + 1 + j)) / (m.factorial : ℝ) := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Finset.sum_range_succ, ih, Finset.prod_range_succ' (fun j => x + (j : ℝ)),
      Finset.prod_range_succ (fun j => x + 1 + (j : ℝ))]
    have hf : ((m + 1).factorial : ℝ) = (m + 1) * (m.factorial : ℝ) := by
      rw [Nat.factorial_succ]; push_cast; ring
    have hm : (m.factorial : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero m
    rw [hf]
    have h1 : ∏ j ∈ range m, (x + ((j + 1 : ℕ) : ℝ)) = ∏ j ∈ range m, (x + 1 + j) := by
      refine Finset.prod_congr rfl fun j _ => ?_
      push_cast; ring
    rw [h1]
    field_simp
    push_cast
    ring

lemma majorant_eq (x : ℝ) (k : ℕ) :
    majorant x k = (∏ i ∈ range k, (x + i)) / (k.factorial : ℝ) := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k with _ | m
    · simp [majorant_zero]
    · rw [majorant_succ]
      have : ∑ i ∈ range (m + 1), majorant x i * x =
          x * ∑ i ∈ range (m + 1), (∏ j ∈ range i, (x + j)) / (i.factorial : ℝ) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun i hi => ?_
        rw [ih i (by simpa [Nat.lt_succ_iff] using Finset.mem_range.1 hi)]
        ring
      rw [this, sum_rising, Finset.prod_range_succ']
      have hf : ((m + 1).factorial : ℝ) = (m + 1) * (m.factorial : ℝ) := by
        rw [Nat.factorial_succ]; push_cast; ring
      have hm : (m.factorial : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero m
      have h1 : ∏ j ∈ range m, (x + ((j + 1 : ℕ) : ℝ)) = ∏ j ∈ range m, (x + 1 + j) := by
        refine Finset.prod_congr rfl fun j _ => ?_
        push_cast; ring
      rw [h1, hf]
      push_cast
      field_simp
      ring

lemma majorant_nonneg {x : ℝ} (hx : 0 ≤ x) (k : ℕ) : 0 ≤ majorant x k := by
  rw [majorant_eq]
  apply div_nonneg _ (by positivity)
  exact Finset.prod_nonneg fun i _ => by positivity

/-- The computed `e_k` is bounded by the majorant. -/
theorem norm_newtonE_le (p : ℕ → ℂ) {x : ℝ} (hx : 0 ≤ x) (hp : ∀ l, ‖p l‖ ≤ x) (k : ℕ) :
    ‖newtonE p k‖ ≤ majorant x k := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k with _ | m
    · simp [newtonE_zero, majorant_zero]
    · rw [newtonE_succ, majorant_succ, norm_mul, norm_inv]
      have hm : ‖((m : ℂ) + 1)‖ = (m : ℝ) + 1 := by
        rw [show ((m : ℂ) + 1) = ((m + 1 : ℕ) : ℂ) by push_cast; ring, Complex.norm_natCast]
        push_cast; ring
      rw [hm]
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i hi => ?_)
      have hi' : i < m + 1 := Finset.mem_range.1 hi
      rw [norm_mul, norm_mul, norm_pow, norm_neg, norm_one, one_pow, one_mul]
      exact mul_le_mul (ih i hi') (hp _) (norm_nonneg _) (majorant_nonneg hx i)

/-- **Precision.** Power sums within `η` give `e_k` within `H_k(x + η) - H_k(x)`. -/
theorem norm_newtonE_sub_le (p p' : ℕ → ℂ) {x η : ℝ} (hx : 0 ≤ x) (hη : 0 ≤ η)
    (hp : ∀ l, ‖p l‖ ≤ x) (hd : ∀ l, ‖p' l - p l‖ ≤ η) (k : ℕ) :
    ‖newtonE p' k - newtonE p k‖ ≤ majorant (x + η) k - majorant x k := by
  have hp' : ∀ l, ‖p' l‖ ≤ x + η := fun l => by
    calc ‖p' l‖ = ‖(p' l - p l) + p l‖ := by ring_nf
      _ ≤ ‖p' l - p l‖ + ‖p l‖ := norm_add_le _ _
      _ ≤ η + x := add_le_add (hd l) (hp l)
      _ = x + η := by ring
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k with _ | m
    · simp [newtonE_zero, majorant_zero]
    · rw [newtonE_succ, newtonE_succ, majorant_succ, majorant_succ, ← mul_sub, ← mul_sub,
        ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, norm_mul, norm_inv]
      have hm : ‖((m : ℂ) + 1)‖ = (m : ℝ) + 1 := by
        rw [show ((m : ℂ) + 1) = ((m + 1 : ℕ) : ℂ) by push_cast; ring, Complex.norm_natCast]
        push_cast; ring
      rw [hm]
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i hi => ?_)
      have hi' : i < m + 1 := Finset.mem_range.1 hi
      have split : (-1 : ℂ) ^ (m - i) * newtonE p' i * p' (m + 1 - i) -
          (-1) ^ (m - i) * newtonE p i * p (m + 1 - i) =
          (-1) ^ (m - i) * ((newtonE p' i - newtonE p i) * p' (m + 1 - i) +
            newtonE p i * (p' (m + 1 - i) - p (m + 1 - i))) := by ring
      rw [split, norm_mul, norm_pow, norm_neg, norm_one, one_pow, one_mul]
      refine (norm_add_le _ _).trans ?_
      rw [norm_mul, norm_mul]
      have a1 : ‖newtonE p' i - newtonE p i‖ * ‖p' (m + 1 - i)‖ ≤
          (majorant (x + η) i - majorant x i) * (x + η) :=
        (mul_le_mul_of_nonneg_left (hp' _) (norm_nonneg _)).trans
          (mul_le_mul_of_nonneg_right (ih i hi') (by linarith))
      have a2 : ‖newtonE p i‖ * ‖p' (m + 1 - i) - p (m + 1 - i)‖ ≤ majorant x i * η :=
        mul_le_mul (norm_newtonE_le p hx hp i) (hd _) (norm_nonneg _) (majorant_nonneg hx i)
      linarith

/-- For the counting read-out (`count_odd`): if each channel's computed feature is within `δ` of the true
one, the computed value of `|G| · N` is within `2 |R| δ = (|G| - 1) δ`. -/
theorem count_error_le {α : Type*} (R : Finset α) (f f' : α → ℂ) (c : ℝ) {δ : ℝ}
    (hδ : ∀ a ∈ R, ‖f' a - f a‖ ≤ δ) :
    |(c + 2 * ∑ a ∈ R, (f' a).re) - (c + 2 * ∑ a ∈ R, (f a).re)| ≤ 2 * R.card * δ := by
  have : (c + 2 * ∑ a ∈ R, (f' a).re) - (c + 2 * ∑ a ∈ R, (f a).re) =
      2 * ∑ a ∈ R, (f' a - f a).re := by
    simp only [Complex.sub_re, Finset.sum_sub_distrib]
    ring
  rw [this, abs_mul, abs_two]
  have h : |∑ a ∈ R, (f' a - f a).re| ≤ R.card * δ := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc ∑ a ∈ R, |(f' a - f a).re| ≤ ∑ a ∈ R, δ :=
          Finset.sum_le_sum fun a ha => (Complex.abs_re_le_norm _).trans (hδ a ha)
      _ = R.card * δ := by rw [Finset.sum_const, nsmul_eq_mul]
  linarith

section Robust

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]
variable {ι : Type*} [Fintype ι]

omit [DecidableEq G] in
lemma norm_psum_char_le (ψ : AddChar G ℂ) (a : ι → G) (l : ℕ) :
    ‖psum (fun j => ψ (a j)) l‖ ≤ Fintype.card ι := by
  refine (norm_sum_le _ _).trans ?_
  simp [AddChar.norm_apply]

/-- **Precision of the count (odd order).** With the channels `R` of `count_odd`, if the layer works with
power sums `p' ψ l` that are each within `η` of the true ones, its value of `|G| · N` is within
`(|G| - 1) (H_k(n + η) - H_k(n))` of the true one; so rounding `(·)/|G|` recovers `N` when
`H_k(n + η) - H_k(n) < 1/2`. -/
theorem count_robust (hodd : Odd (Fintype.card G)) (k : ℕ) :
    ∃ R : Finset (AddChar G ℂ), 2 * R.card = Fintype.card G - 1 ∧
      ∀ (q : G) (a : ι → G) (p' : AddChar G ℂ → ℕ → ℂ) (η : ℝ), 0 ≤ η →
        (∀ ψ ∈ R, ∀ l, ‖p' ψ l - psum (fun j => ψ (a j)) l‖ ≤ η) →
        |((Fintype.card ι).choose k + 2 * ∑ ψ ∈ R, (ψ q * newtonE (p' ψ) k).re) -
            (Fintype.card G : ℝ) *
              (((univ : Finset ι).powersetCard k).filter (fun S => q + ∑ j ∈ S, a j = 0)).card| ≤
          ((Fintype.card G : ℝ) - 1) *
            (majorant (Fintype.card ι + η) k - majorant (Fintype.card ι) k) := by
  obtain ⟨R, hR, hcount⟩ := count_odd (ι := ι) hodd k
  refine ⟨R, hR, fun q a p' η hη hp' => ?_⟩
  rw [hcount q a]
  have hfeat : ∀ ψ ∈ R, ‖ψ q * newtonE (p' ψ) k - feature ψ q a k‖ ≤
      majorant (Fintype.card ι + η) k - majorant (Fintype.card ι) k := by
    intro ψ hψ
    rw [feature, ← mul_sub, norm_mul, AddChar.norm_apply, one_mul]
    exact norm_newtonE_sub_le _ _ (Nat.cast_nonneg _) hη (norm_psum_char_le ψ a) (hp' ψ hψ) k
  have h := count_error_le R (fun ψ => feature ψ q a k) (fun ψ => ψ q * newtonE (p' ψ) k)
    ((Fintype.card ι).choose k : ℝ) hfeat
  have hc : (2 * R.card : ℝ) = (Fintype.card G : ℝ) - 1 := by
    have h1 : 1 ≤ Fintype.card G := Fintype.card_pos
    have : ((2 * R.card : ℕ) : ℝ) = ((Fintype.card G - 1 : ℕ) : ℝ) := by rw [hR]
    push_cast [Nat.cast_sub h1] at this
    linarith
  rw [← hc]
  exact h

end Robust

end Newton
