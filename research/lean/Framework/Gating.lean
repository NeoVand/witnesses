import Mathlib

/-!
# Theorem 2 (size law): gated existence survives growth while `β Δ ≥ log |F|`

A family `F` of witnesses, a score `s w` for each, a level `a` and a margin `Δ`: every witness either reaches the
level (`a ≤ s w`, a true witness) or stays `Δ` below it (`s w ≤ a - Δ`). An exponential gate at inverse temperature
`β` weighs each witness by `exp (β (s w - a))`; the family's partition function is the sum `Z`.

- `gated_existence`: if `|F| · exp (-β Δ) < 1`, then "some witness is true" holds exactly when `Z ≥ 1`.
- `gated_existence_log`: the same under `log |F| < β Δ` (the size law).
- `background_le`: with no true witness, `Z ≤ |F| · exp (-β Δ)` (the background bound).
- `tight`: the bound is attained: with every score at `a - Δ`, `Z = |F| · exp (-β Δ)`, so the threshold `Z ≥ 1` is wrong
  on that input once `|F| · exp (-β Δ) ≥ 1`.
- `softmax_existence`: the normalised read-out `Z_true / (Z_true + Z_false) ≥ 1/2` is right under the same law;
  `softmax_size_law` states it on `F` itself under `log |F| ≤ β Δ`.
- `strict_existence`: for `|F| ≥ 2` and `log |F| ≤ β Δ` (equality included), existence holds exactly when `Z > 1`.
- `size_law_iff` (Theorem 2 of the paper): for `|F| ≥ 2`, `Δ > 0`, `β ≥ 0` and scores free within the
  margin, some threshold on `Z` (`θ ≤ Z` or `θ < Z`) decides existence on every input iff `log |F| ≤ β Δ`.
- `size_law_equality`: at `β Δ = log |F|`, the strict threshold `Z > 1` decides existence, it is the only strict
  threshold that does, and no non-strict threshold does.

Built by `lake build` in this project (Lean 4.35, Mathlib v4.35.0).
-/

open Finset Real

namespace Framework

variable {ι : Type*}

lemma background_le (F : Finset ι) (s : ι → ℝ) (a Δ β : ℝ) (hβ : 0 ≤ β)
    (hneg : ∀ w ∈ F, s w ≤ a - Δ) :
    ∑ w ∈ F, exp (β * (s w - a)) ≤ (F.card : ℝ) * exp (-(β * Δ)) := by
  have hle : ∀ w ∈ F, exp (β * (s w - a)) ≤ exp (-(β * Δ)) := by
    intro w hw
    apply exp_le_exp.mpr
    have h := mul_le_mul_of_nonneg_left (show s w - a ≤ -Δ by linarith [hneg w hw]) hβ
    linarith
  have := sum_le_card_nsmul F (fun w => exp (β * (s w - a))) _ hle
  simpa [nsmul_eq_mul] using this

theorem gated_existence (F : Finset ι) (s : ι → ℝ) (a Δ β : ℝ) (hβ : 0 ≤ β)
    (hmargin : ∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ)
    (hlaw : (F.card : ℝ) * exp (-(β * Δ)) < 1) :
    (∃ w ∈ F, a ≤ s w) ↔ 1 ≤ ∑ w ∈ F, exp (β * (s w - a)) := by
  constructor
  · rintro ⟨w, hw, hs⟩
    have h1 : 1 ≤ exp (β * (s w - a)) := by
      rw [← exp_zero]; exact exp_le_exp.mpr (mul_nonneg hβ (by linarith))
    exact h1.trans (single_le_sum (f := fun i => exp (β * (s i - a))) (fun i _ => (exp_pos _).le) hw)
  · intro hZ
    by_contra hno
    simp only [not_exists, not_and, not_le] at hno
    have hneg : ∀ w ∈ F, s w ≤ a - Δ := by
      intro w hw
      rcases hmargin w hw with h | h
      · exact absurd h (not_le.mpr (hno w hw))
      · exact h
    have := background_le F s a Δ β hβ hneg
    linarith

theorem gated_existence_log (F : Finset ι) (s : ι → ℝ) (a Δ β : ℝ) (hβ : 0 ≤ β)
    (hF : F.Nonempty) (hmargin : ∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ)
    (hlaw : log (F.card : ℝ) < β * Δ) :
    (∃ w ∈ F, a ≤ s w) ↔ 1 ≤ ∑ w ∈ F, exp (β * (s w - a)) := by
  apply gated_existence F s a Δ β hβ hmargin
  have hc : (0 : ℝ) < F.card := by exact_mod_cast hF.card_pos
  have h1 : (F.card : ℝ) = exp (log (F.card : ℝ)) := (exp_log hc).symm
  rw [h1, ← exp_add]
  rw [← exp_zero]
  exact exp_lt_exp.mpr (by linarith)

theorem tight (F : Finset ι) (a Δ β : ℝ) :
    ∑ _w ∈ F, exp (β * ((a - Δ) - a)) = (F.card : ℝ) * exp (-(β * Δ)) := by
  rw [sum_const, nsmul_eq_mul]
  congr 2
  ring

/-- The normalised (softmax) read-out of existence: the weight on true witnesses is at least one half whenever
some witness is true and the false ones obey the background bound. -/
theorem softmax_existence (T N : Finset ι) (s : ι → ℝ) (a Δ β : ℝ) (hβ : 0 ≤ β)
    (htrue : ∃ w ∈ T, a ≤ s w) (hfalse : ∀ w ∈ N, s w ≤ a - Δ)
    (hlaw : (N.card : ℝ) * exp (-(β * Δ)) ≤ 1) :
    1 / 2 ≤ (∑ w ∈ T, exp (β * (s w - a))) /
      ((∑ w ∈ T, exp (β * (s w - a))) + ∑ w ∈ N, exp (β * (s w - a))) := by
  obtain ⟨w, hw, hs⟩ := htrue
  set ZT := ∑ w ∈ T, exp (β * (s w - a))
  set ZN := ∑ w ∈ N, exp (β * (s w - a))
  have hT : 1 ≤ ZT := by
    have h1 : 1 ≤ exp (β * (s w - a)) := by
      rw [← exp_zero]; exact exp_le_exp.mpr (mul_nonneg hβ (by linarith))
    exact h1.trans (single_le_sum (f := fun i => exp (β * (s i - a))) (fun i _ => (exp_pos _).le) hw)
  have hN : ZN ≤ 1 := (background_le N s a Δ β hβ hfalse).trans hlaw
  have hN0 : 0 ≤ ZN := sum_nonneg (fun i _ => (exp_pos _).le)
  rw [le_div_iff₀ (by linarith)]
  linarith

/-! ## The necessity half and the equality case

"Scores free within the margin" means that the inputs are all score functions `s` with
`a ≤ s w ∨ s w ≤ a - Δ` for every `w ∈ F`. A threshold on `Z` is `θ ≤ Z` or the strict `θ < Z`. -/

/-- `|F| e^{-x} ≤ 1` is the log form `log |F| ≤ x`. -/
lemma card_mul_exp_le_one_iff (F : Finset ι) (hF : F.Nonempty) (x : ℝ) :
    (F.card : ℝ) * exp (-x) ≤ 1 ↔ log (F.card : ℝ) ≤ x := by
  have hc : (0 : ℝ) < F.card := by exact_mod_cast hF.card_pos
  rw [Real.log_le_iff_le_exp hc, exp_neg, mul_inv_le_iff₀ (exp_pos x), one_mul]

/-- For `β > 0` the true inputs come arbitrarily close to `Z = 1`: one score at the level `a` and the rest far
below give a true input, free within the margin, with `Z < c` for any `c > 1`. -/
lemma exists_true_input_lt (F : Finset ι) (a Δ β c : ℝ) (hβ : 0 < β) (hc : 1 < c)
    {w₀ : ι} (hw₀ : w₀ ∈ F) :
    ∃ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) ∧ (∃ w ∈ F, a ≤ s w) ∧
      ∑ w ∈ F, exp (β * (s w - a)) < c := by
  classical
  have hlim : Filter.Tendsto (fun M : ℝ => (F.card : ℝ) * exp (-(β * M))) Filter.atTop (nhds 0) := by
    have h1 : Filter.Tendsto (fun M : ℝ => β * M) Filter.atTop Filter.atTop :=
      Filter.tendsto_id.const_mul_atTop hβ
    simpa using (Real.tendsto_exp_neg_atTop_nhds_zero.comp h1).const_mul (F.card : ℝ)
  obtain ⟨M, hMΔ, hM⟩ := ((Filter.eventually_ge_atTop Δ).and
    (hlim.eventually (gt_mem_nhds (show (0 : ℝ) < c - 1 by linarith)))).exists
  refine ⟨fun w => if w = w₀ then a else a - M, ?_, ⟨w₀, hw₀, by simp⟩, ?_⟩
  · intro w _
    by_cases h : w = w₀
    · left; simp [h]
    · right; simp only [h, ite_false]; linarith
  · have hle : ∀ w ∈ F, exp (β * ((if w = w₀ then a else a - M) - a)) ≤
        exp (-(β * M)) + if w = w₀ then 1 else 0 := by
      intro w _
      by_cases h : w = w₀
      · simp only [h, ite_true, sub_self, mul_zero, exp_zero]
        linarith [exp_pos (-(β * M))]
      · simp only [h, ite_false, add_zero]
        exact le_of_eq (congrArg exp (by ring))
    calc ∑ w ∈ F, exp (β * ((if w = w₀ then a else a - M) - a))
        ≤ ∑ w ∈ F, (exp (-(β * M)) + if w = w₀ then 1 else 0) := sum_le_sum hle
      _ = (F.card : ℝ) * exp (-(β * M)) + 1 := by
          rw [sum_add_distrib, sum_const, nsmul_eq_mul, sum_ite_eq' F w₀ (fun _ => (1 : ℝ))]
          simp [hw₀]
      _ < c := by linarith

/-- Sufficiency with the strict threshold `Z > 1`: for `|F| ≥ 2` and `log |F| ≤ β Δ` (equality included), some
witness is true exactly when `Z > 1`. Every true input passes because its other terms are positive. -/
theorem strict_existence (F : Finset ι) (s : ι → ℝ) (a Δ β : ℝ) (hβ : 0 ≤ β) (hF : 2 ≤ F.card)
    (hmargin : ∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ)
    (hlaw : log (F.card : ℝ) ≤ β * Δ) :
    (∃ w ∈ F, a ≤ s w) ↔ 1 < ∑ w ∈ F, exp (β * (s w - a)) := by
  classical
  have hFne : F.Nonempty := card_pos.mp (by omega)
  constructor
  · rintro ⟨w, hw, hs⟩
    have h1 : 1 ≤ exp (β * (s w - a)) := by
      rw [← exp_zero]; exact exp_le_exp.mpr (mul_nonneg hβ (by linarith))
    have hrest : 0 < ∑ i ∈ F.erase w, exp (β * (s i - a)) := by
      apply sum_pos (fun i _ => exp_pos _)
      apply card_pos.mp
      rw [card_erase_of_mem hw]; omega
    have hsplit : exp (β * (s w - a)) + ∑ i ∈ F.erase w, exp (β * (s i - a)) =
        ∑ i ∈ F, exp (β * (s i - a)) := add_sum_erase F (fun i => exp (β * (s i - a))) hw
    rw [← hsplit]
    linarith
  · intro hZ
    by_contra hno
    simp only [not_exists, not_and, not_le] at hno
    have hneg : ∀ w ∈ F, s w ≤ a - Δ := by
      intro w hw
      rcases hmargin w hw with h | h
      · exact absurd h (not_le.mpr (hno w hw))
      · exact h
    have h1 := background_le F s a Δ β hβ hneg
    have h2 := (card_mul_exp_le_one_iff F hFne _).mpr hlaw
    linarith

/-- **Size law** (Theorem 2 of the paper). For `|F| ≥ 2`, `Δ > 0`, `β ≥ 0` and scores free within the margin,
some threshold on `Z` (`θ ≤ Z` or `θ < Z`) decides existence on every input if and only if `log |F| ≤ β Δ`. -/
theorem size_law_iff (F : Finset ι) (a Δ β : ℝ) (hF : 2 ≤ F.card) (hΔ : 0 < Δ) (hβ : 0 ≤ β) :
    (∃ θ : ℝ,
      (∀ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) →
        ((∃ w ∈ F, a ≤ s w) ↔ θ ≤ ∑ w ∈ F, exp (β * (s w - a)))) ∨
      (∀ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) →
        ((∃ w ∈ F, a ≤ s w) ↔ θ < ∑ w ∈ F, exp (β * (s w - a))))) ↔
    log (F.card : ℝ) ≤ β * Δ := by
  have hFne : F.Nonempty := card_pos.mp (by omega)
  constructor
  · rintro ⟨θ, hθ⟩
    by_contra hlaw
    -- the all-false input at the margin reaches `Z = |F| e^{-βΔ} > 1`
    have hf_adm : ∀ w ∈ F, a ≤ (fun _ => a - Δ : ι → ℝ) w ∨ (fun _ => a - Δ : ι → ℝ) w ≤ a - Δ :=
      fun _ _ => Or.inr le_rfl
    have hf_false : ¬ ∃ w ∈ F, a ≤ (fun _ => a - Δ : ι → ℝ) w := by
      rintro ⟨w, _, h⟩; simp only at h; linarith
    have hZf : ∑ w ∈ F, exp (β * ((fun _ => a - Δ : ι → ℝ) w - a)) = (F.card : ℝ) * exp (-(β * Δ)) :=
      tight F a Δ β
    have hZf1 : 1 < (F.card : ℝ) * exp (-(β * Δ)) :=
      not_le.mp fun h => hlaw ((card_mul_exp_le_one_iff F hFne _).mp h)
    -- a true input with `Z` no larger
    obtain ⟨st, ht_adm, ht_true, hZt⟩ : ∃ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) ∧
        (∃ w ∈ F, a ≤ s w) ∧ ∑ w ∈ F, exp (β * (s w - a)) ≤ (F.card : ℝ) * exp (-(β * Δ)) := by
      rcases hβ.eq_or_lt with h0 | hpos
      · -- at `β = 0` every input has `Z = |F|`
        subst h0
        obtain ⟨w, hw⟩ := hFne
        exact ⟨fun _ => a, fun _ _ => Or.inl le_rfl, ⟨w, hw, le_rfl⟩, by simp⟩
      · obtain ⟨w₀, hw₀⟩ := hFne
        obtain ⟨s, h1, h2, h3⟩ := exists_true_input_lt F a Δ β _ hpos hZf1 hw₀
        exact ⟨s, h1, h2, h3.le⟩
    rcases hθ with h | h
    · exact hf_false ((h _ hf_adm).mpr ((((h st ht_adm).mp ht_true).trans hZt).trans_eq hZf.symm))
    · exact hf_false ((h _ hf_adm).mpr ((((h st ht_adm).mp ht_true).trans_le hZt).trans_eq hZf.symm))
  · intro hlaw
    exact ⟨1, Or.inr fun s hs => strict_existence F s a Δ β hβ hF hs hlaw⟩

/-- **Size law at equality.** When `β Δ = log |F|` (with `|F| ≥ 2`, `Δ > 0`, `β ≥ 0` and scores free within the
margin), the strict threshold `Z > 1` decides existence on every input, it is the only strict threshold that does,
and no non-strict threshold `θ ≤ Z` does. -/
theorem size_law_equality (F : Finset ι) (a Δ β : ℝ) (hF : 2 ≤ F.card) (hΔ : 0 < Δ) (hβ : 0 ≤ β)
    (heq : β * Δ = log (F.card : ℝ)) :
    (∀ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) →
        ((∃ w ∈ F, a ≤ s w) ↔ 1 < ∑ w ∈ F, exp (β * (s w - a)))) ∧
    (∀ θ : ℝ, (∀ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) →
        ((∃ w ∈ F, a ≤ s w) ↔ θ < ∑ w ∈ F, exp (β * (s w - a)))) → θ = 1) ∧
    ¬ ∃ θ : ℝ, ∀ s : ι → ℝ, (∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) →
        ((∃ w ∈ F, a ≤ s w) ↔ θ ≤ ∑ w ∈ F, exp (β * (s w - a))) := by
  have hFne : F.Nonempty := card_pos.mp (by omega)
  have hc : (0 : ℝ) < F.card := by exact_mod_cast hFne.card_pos
  have hlog : 0 < log (F.card : ℝ) := Real.log_pos (by exact_mod_cast (show 1 < F.card by omega))
  have hpos : 0 < β := by
    rcases hβ.eq_or_lt with h0 | h
    · rw [← h0, zero_mul] at heq; linarith
    · exact h
  obtain ⟨w₀, hw₀⟩ := hFne
  -- the all-false input at the margin has `Z = 1`
  have hf_adm : ∀ w ∈ F, a ≤ (fun _ => a - Δ : ι → ℝ) w ∨ (fun _ => a - Δ : ι → ℝ) w ≤ a - Δ :=
    fun _ _ => Or.inr le_rfl
  have hf_false : ¬ ∃ w ∈ F, a ≤ (fun _ => a - Δ : ι → ℝ) w := by
    rintro ⟨w, _, h⟩; simp only at h; linarith
  have hZf : ∑ w ∈ F, exp (β * ((fun _ => a - Δ : ι → ℝ) w - a)) = 1 := by
    rw [tight F a Δ β, heq, exp_neg, exp_log hc]
    exact mul_inv_cancel₀ hc.ne'
  refine ⟨fun s hs => strict_existence F s a Δ β hβ hF hs heq.ge, ?_, ?_⟩
  · intro θ hθ
    have h1 : ¬ θ < 1 := fun h => hf_false ((hθ _ hf_adm).mpr (h.trans_eq hZf.symm))
    refine le_antisymm ?_ (not_lt.mp h1)
    by_contra h2
    obtain ⟨s, hs_adm, hs_true, hZs⟩ := exists_true_input_lt F a Δ β θ hpos (not_le.mp h2) hw₀
    exact lt_asymm ((hθ s hs_adm).mp hs_true) hZs
  · rintro ⟨θ, hθ⟩
    have h1 : ¬ θ ≤ 1 := fun h => hf_false ((hθ _ hf_adm).mpr (h.trans_eq hZf.symm))
    obtain ⟨s, hs_adm, hs_true, hZs⟩ := exists_true_input_lt F a Δ β θ hpos (not_le.mp h1) hw₀
    exact not_le.mpr hZs ((hθ s hs_adm).mp hs_true)

/-- The softmax read-out under the size law, stated on `F` itself: if `log |F| ≤ β Δ` and some witness is true,
the softmax weight on the true witnesses is at least one half. -/
theorem softmax_size_law (F : Finset ι) (s : ι → ℝ) (a Δ β : ℝ) (hβ : 0 ≤ β) (hF : F.Nonempty)
    (hmargin : ∀ w ∈ F, a ≤ s w ∨ s w ≤ a - Δ) (hlaw : log (F.card : ℝ) ≤ β * Δ)
    (htrue : ∃ w ∈ F, a ≤ s w) :
    1 / 2 ≤ (∑ w ∈ F with a ≤ s w, exp (β * (s w - a))) / ∑ w ∈ F, exp (β * (s w - a)) := by
  rw [← sum_filter_add_sum_filter_not F (fun w => a ≤ s w) (fun w => exp (β * (s w - a)))]
  apply softmax_existence (F.filter (fun w => a ≤ s w)) (F.filter (fun w => ¬ a ≤ s w)) s a Δ β hβ
  · obtain ⟨w, hw, h⟩ := htrue
    exact ⟨w, mem_filter.mpr ⟨hw, h⟩, h⟩
  · intro w hw
    rw [mem_filter] at hw
    exact (hmargin w hw.1).resolve_left hw.2
  · have hcard : ((F.filter (fun w => ¬ a ≤ s w)).card : ℝ) ≤ F.card := by
      exact_mod_cast card_filter_le _ _
    exact (mul_le_mul_of_nonneg_right hcard (exp_pos _).le).trans
      ((card_mul_exp_le_one_iff F hF _).mpr hlaw)

end Framework

#print axioms Framework.gated_existence
#print axioms Framework.gated_existence_log
#print axioms Framework.background_le
#print axioms Framework.tight
#print axioms Framework.softmax_existence
#print axioms Framework.strict_existence
#print axioms Framework.size_law_iff
#print axioms Framework.size_law_equality
#print axioms Framework.softmax_size_law
