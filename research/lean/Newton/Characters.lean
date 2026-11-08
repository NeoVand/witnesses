import Mathlib.Analysis.Fourier.FiniteAbelian.PontryaginDuality
import Mathlib.RingTheory.MvPolynomial.Symmetric.NewtonIdentities

/-!
# Newton attention counts zero-sum subsets through characters

Source: Proposition 2 of the paper (characters count set pairs at linear cost) and its proof, with the
general count over any finite abelian group and any subset size (`count_eq`).

Setting. `G` is a finite abelian group (written additively), `q : G` is the query, the partners are a
family `a : ι → G` indexed by a finite type `ι` (so partners may repeat as elements of `G`; distinctness
is distinctness of indices). A character is `ψ : AddChar G ℂ`.

- `Newton.esymm x k` is the elementary symmetric polynomial of degree `k` of a family `x : ι → ℂ`, written
  as the sum over the `k`-subsets of indices of the product of their values.
- `Newton.newtonE p k` is what the layer computes: the recursion of Newton's identities,
  `e₀ = 1`, `m eₘ = Σ_{l=1}^{m} (-1)^{l-1} e_{m-l} p_l`, applied to the power sums `p`.
- `newtonE_psum`: the recursion applied to the power sums of `x` gives `esymm x k` (Newton's identities,
  from Mathlib's `MvPolynomial.mul_esymm_eq_sum`).
- `count_eq`: `|G| · #{S : |S| = k, q + Σ_{j∈S} a j = 0} = Σ_ψ ψ(q) e_k(ψ(a₁), …, ψ(aₙ))`.
- `subset_sum_eq`: for every `h : G → ℂ`, `Σ_{|S|=k} h(q + Σ_S a) = Σ_ψ ĥ(ψ) ψ(q) e_k(ψ ∘ a)`, with
  `ĥ(ψ) = |G|⁻¹ Σ_y h(y) ψ(-y)`.
- `layer_exact`: for real `h`, a Newton layer of order `k + 1` whose channels hold one character from
  each conjugate pair `{ψ, ψ⁻¹}` in the support of `ĥ` (and each real character once), with a real linear
  read-out of the real and imaginary parts of its features and a constant term, computes the sum exactly.
- `exists_channels`: such channel sets exist, with `2 |R| + |Q| = #{ψ ≠ 1 : ĥ ψ ≠ 0}`, `Q` the real
  characters among them; so the layer has `|R| + |Q|` channels, the number of conjugate pairs.
- `count_odd`: for `|G|` odd and the count (`h = 𝟙_{0}`), no non-trivial character is real, and
  `|G| · N = C(n, k) + 2 Re Σ_{ψ ∈ R} ψ(q) e_k(ψ ∘ a)` with `2 |R| = |G| - 1`.
- `set_count`, `set4_count`: SET (`G = (ℤ/3)⁴`, `k = 2`) and four-card SET (`k = 3`) with 40 channels.
-/

open Finset

namespace Newton

/-! ### Elementary symmetric polynomials, power sums and Newton's identities -/

section Symmetric

variable {ι : Type*} [Fintype ι]

/-- `e_k(x)`: the sum over the `k`-subsets `S` of the index set of `∏_{j ∈ S} x j`. -/
noncomputable def esymm (x : ι → ℂ) (k : ℕ) : ℂ :=
  ∑ S ∈ (univ : Finset ι).powersetCard k, ∏ j ∈ S, x j

/-- The power sum `p_m(x) = Σ_j (x j)^m`. -/
noncomputable def psum (x : ι → ℂ) (m : ℕ) : ℂ := ∑ j, x j ^ m

/-- The layer's recursion (Newton's identities solved for `e_m`): `e₀ = 1` and
`e_{m+1} = (m+1)⁻¹ Σ_{i=0}^{m} (-1)^{m-i} e_i p_{m+1-i}`, which is
`m e_m = Σ_{l=1}^{m} (-1)^{l-1} e_{m-l} p_l` with `l = m - i`. -/
noncomputable def newtonE (p : ℕ → ℂ) : ℕ → ℂ
  | 0 => 1
  | m + 1 => ((m : ℂ) + 1)⁻¹ * ∑ i : Fin (m + 1), (-1) ^ (m - (i : ℕ)) * newtonE p i * p (m + 1 - i)
decreasing_by exact i.isLt

lemma newtonE_zero (p : ℕ → ℂ) : newtonE p 0 = 1 := by rw [newtonE]

lemma newtonE_succ (p : ℕ → ℂ) (m : ℕ) :
    newtonE p (m + 1) =
      ((m : ℂ) + 1)⁻¹ * ∑ i ∈ range (m + 1), (-1) ^ (m - i) * newtonE p i * p (m + 1 - i) := by
  rw [newtonE, Fin.sum_univ_eq_sum_range (fun i => (-1) ^ (m - i) * newtonE p i * p (m + 1 - i))]

/-- Newton's identities for a family of complex numbers, in the form
`k e_k = (-1)^(k+1) Σ_{i<k} (-1)^i e_i p_{k-i}` (Mathlib's `MvPolynomial.mul_esymm_eq_sum`, evaluated). -/
theorem newton_identity (x : ι → ℂ) (k : ℕ) :
    (k : ℂ) * esymm x k =
      (-1) ^ (k + 1) * ∑ i ∈ range k, (-1) ^ i * esymm x i * psum x (k - i) := by
  classical
  have h := congrArg (MvPolynomial.eval x) (MvPolynomial.mul_esymm_eq_sum ι ℂ k)
  have hE : ∀ m, MvPolynomial.eval x (MvPolynomial.esymm ι ℂ m) = esymm x m := by
    intro m
    simp [MvPolynomial.esymm, esymm, map_sum, map_prod]
  have hP : ∀ m, MvPolynomial.eval x (MvPolynomial.psum ι ℂ m) = psum x m := by
    intro m
    simp [MvPolynomial.psum, psum, map_sum]
  simp only [map_mul, map_natCast, map_pow, map_neg, map_one, map_sum, hE, hP] at h
  rw [h, Finset.sum_filter, Nat.sum_antidiagonal_eq_sum_range_succ_mk, Finset.sum_range_succ]
  simp only [lt_irrefl, ite_false, add_zero]
  congr 1
  refine Finset.sum_congr rfl fun i hi => ?_
  simp [Finset.mem_range.1 hi]

/-- Newton's identities, under the name the paper uses. -/
alias newton_identities := newton_identity

/-- The layer's recursion on the power sums of `x` computes the elementary symmetric polynomials. -/
theorem newtonE_psum (x : ι → ℂ) (k : ℕ) : newtonE (psum x) k = esymm x k := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k with _ | m
    · rw [newtonE_zero]
      simp [esymm]
    · rw [newtonE_succ]
      have hm : ((m : ℂ) + 1) ≠ 0 := by exact_mod_cast Nat.succ_ne_zero m
      have hid := newton_identity x (m + 1)
      push_cast at hid
      rw [inv_mul_eq_iff_eq_mul₀ hm, hid, Finset.mul_sum]
      refine Finset.sum_congr rfl fun i hi => ?_
      have hi' : i < m + 1 := Finset.mem_range.1 hi
      rw [ih i hi']
      have hsign : ((-1 : ℂ) ^ (m - i)) = (-1) ^ (m + 1 + 1) * (-1) ^ i := by
        rw [← pow_add]
        have : m + 1 + 1 + i = (m - i) + 2 * (i + 1) := by omega
        rw [this, pow_add, pow_mul]
        simp
      rw [hsign]
      ring

lemma newtonE_one (p : ℕ → ℂ) : newtonE p 1 = p 1 := by
  rw [show (1 : ℕ) = 0 + 1 from rfl, newtonE_succ]
  simp [newtonE_zero]

lemma newtonE_two (p : ℕ → ℂ) : newtonE p 2 = (p 1 ^ 2 - p 2) / 2 := by
  rw [show (2 : ℕ) = 1 + 1 from rfl, newtonE_succ]
  simp [Finset.sum_range_succ, newtonE_zero, newtonE_one]
  ring

lemma newtonE_three (p : ℕ → ℂ) :
    newtonE p 3 = (p 1 ^ 3 - 3 * p 1 * p 2 + 2 * p 3) / 6 := by
  rw [show (3 : ℕ) = 2 + 1 from rfl, newtonE_succ]
  simp [Finset.sum_range_succ, newtonE_zero, newtonE_one, newtonE_two]
  ring

/-- The closed forms the layer uses for orders two and three: `e₁ = p₁`, `e₂ = (p₁² - p₂)/2`,
`e₃ = (p₁³ - 3 p₁ p₂ + 2 p₃)/6`. -/
theorem esymm_one (x : ι → ℂ) : esymm x 1 = psum x 1 := by
  rw [← newtonE_psum, newtonE_one]

theorem esymm_two (x : ι → ℂ) : esymm x 2 = (psum x 1 ^ 2 - psum x 2) / 2 := by
  rw [← newtonE_psum, newtonE_two]

theorem esymm_three (x : ι → ℂ) :
    esymm x 3 = (psum x 1 ^ 3 - 3 * psum x 1 * psum x 2 + 2 * psum x 3) / 6 := by
  rw [← newtonE_psum, newtonE_three]

lemma esymm_one_const (k : ℕ) : esymm (fun _ : ι => (1 : ℂ)) k = (Fintype.card ι).choose k := by
  simp [esymm, Finset.card_powersetCard]

end Symmetric

/-! ### Characters turn subset sums into products -/

section Characters

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]
variable {ι : Type*} [Fintype ι]

omit [Fintype G] [DecidableEq G] [Fintype ι] in
lemma map_finset_sum (ψ : AddChar G ℂ) (S : Finset ι) (a : ι → G) :
    ψ (∑ j ∈ S, a j) = ∏ j ∈ S, ψ (a j) := by
  classical
  induction S using Finset.induction_on with
  | empty => simp
  | insert j S hj ih => rw [Finset.sum_insert hj, Finset.prod_insert hj, AddChar.map_add_eq_mul, ih]

omit [Fintype G] [DecidableEq G] in
/-- One channel holding `ψ` scores every `k`-subset: `ψ(q) e_k(ψ ∘ a) = Σ_{|S|=k} ψ(q + Σ_S a)`. -/
theorem char_mul_esymm (ψ : AddChar G ℂ) (q : G) (a : ι → G) (k : ℕ) :
    ψ q * esymm (fun j => ψ (a j)) k =
      ∑ S ∈ (univ : Finset ι).powersetCard k, ψ (q + ∑ j ∈ S, a j) := by
  simp only [esymm, Finset.mul_sum, AddChar.map_add_eq_mul, map_finset_sum]

/-- **General count.** `|G| · #{S : |S| = k, q + Σ_{j ∈ S} a_j = 0} = Σ_ψ ψ(q) e_k(ψ(a_1), …, ψ(a_n))`. -/
theorem count_eq (q : G) (a : ι → G) (k : ℕ) :
    (Fintype.card G : ℂ) *
        (((univ : Finset ι).powersetCard k).filter (fun S => q + ∑ j ∈ S, a j = 0)).card =
      ∑ ψ : AddChar G ℂ, ψ q * esymm (fun j => ψ (a j)) k := by
  simp_rw [char_mul_esymm]
  rw [Finset.sum_comm]
  simp_rw [AddChar.sum_apply_eq_ite]
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_comm]

/-- The Fourier coefficient `ĥ(ψ) = |G|⁻¹ Σ_y h(y) ψ(-y)` (`ψ(-y)` is the conjugate of `ψ(y)`). -/
noncomputable def coeff (h : G → ℂ) (ψ : AddChar G ℂ) : ℂ :=
  (Fintype.card G : ℂ)⁻¹ * ∑ y, h y * ψ (-y)

/-- Fourier inversion on a finite abelian group. -/
theorem inversion (h : G → ℂ) (x : G) : ∑ ψ : AddChar G ℂ, coeff h ψ * ψ x = h x := by
  have hG : (Fintype.card G : ℂ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  have step : ∀ ψ : AddChar G ℂ, coeff h ψ * ψ x =
      (Fintype.card G : ℂ)⁻¹ * ∑ y, h y * ψ (x - y) := by
    intro ψ
    rw [coeff, mul_assoc, Finset.sum_mul]
    congr 1
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [mul_assoc, ← AddChar.map_add_eq_mul, neg_add_eq_sub]
  simp_rw [step]
  rw [← Finset.mul_sum, Finset.sum_comm]
  simp_rw [← Finset.mul_sum, AddChar.sum_apply_eq_ite, sub_eq_zero]
  simp only [mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
  field_simp

/-- **General subset sum.** For every `h : G → ℂ`,
`Σ_{|S|=k} h(q + Σ_{j∈S} a_j) = Σ_ψ ĥ(ψ) ψ(q) e_k(ψ ∘ a)`. -/
theorem subset_sum_eq (h : G → ℂ) (q : G) (a : ι → G) (k : ℕ) :
    ∑ S ∈ (univ : Finset ι).powersetCard k, h (q + ∑ j ∈ S, a j) =
      ∑ ψ : AddChar G ℂ, coeff h ψ * (ψ q * esymm (fun j => ψ (a j)) k) := by
  simp_rw [char_mul_esymm, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun S _ => ?_
  rw [inversion]

end Characters

/-! ### The layer: one channel per conjugate pair, a real linear read-out -/

section Layer

open scoped Classical ComplexConjugate

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]
variable {ι : Type*} [Fintype ι]

/-- The feature of order `k + 1` of a channel that gives each element `x` of `G` the complex feature
`φ x`, at the query `q` with partners `a`: `φ(q) e_k`, where `e_k` is computed by the layer's recursion
from the power sums `p_m = Σ_j φ(a_j)^m`. -/
noncomputable def feature (φ : G → ℂ) (q : G) (a : ι → G) (k : ℕ) : ℂ :=
  φ q * newtonE (psum fun j => φ (a j)) k

omit [Fintype G] [DecidableEq G] in
lemma feature_char (ψ : AddChar G ℂ) (q : G) (a : ι → G) (k : ℕ) :
    feature ψ q a k = ψ q * esymm (fun j => ψ (a j)) k := by
  rw [feature, newtonE_psum]

/-- `h` lifted to `ℂ`. -/
noncomputable def lift (h : G → ℝ) : G → ℂ := fun x => (h x : ℂ)

omit [DecidableEq G] in
lemma coeff_inv (h : G → ℝ) (ψ : AddChar G ℂ) :
    coeff (lift h) ψ⁻¹ = conj (coeff (lift h) ψ) := by
  simp [coeff, lift, map_sum, AddChar.map_neg_eq_conj]

omit [DecidableEq G] in
lemma feature_inv (ψ : AddChar G ℂ) (q : G) (a : ι → G) (k : ℕ) :
    feature ⇑(ψ⁻¹) q a k = conj (feature ψ q a k) := by
  simp [feature_char, esymm, map_sum, map_prod, AddChar.map_neg_eq_conj, AddChar.inv_apply]

/-- The read-out of one channel: a real linear map of the real and imaginary parts of its feature. -/
def readout (w z : ℂ) : ℝ := w.re * z.re - w.im * z.im

lemma readout_eq (w z : ℂ) : readout w z = (w * z).re := by simp [readout, Complex.mul_re]

omit [DecidableEq G] in
lemma readout_inv (h : G → ℝ) (ψ : AddChar G ℂ) (q : G) (a : ι → G) (k : ℕ) :
    readout (coeff (lift h) ψ⁻¹) (feature ⇑(ψ⁻¹) q a k) = readout (coeff (lift h) ψ) (feature ψ q a k) := by
  rw [readout_eq, readout_eq, coeff_inv, feature_inv, ← map_mul, Complex.conj_re]

/-- **The layer.** Let `h : G → ℝ`. Let `R` hold one character from each conjugate pair
`{ψ, ψ⁻¹}` of non-real characters (`R` and its inverses are disjoint, `1 ∉ R`), and `Q` real non-trivial
characters, such that every character with `ĥ ψ ≠ 0` is `1`, in `R`, the inverse of one in `R`, or in
`Q`. Then a Newton layer of order `k + 1` with one channel per character of `R ∪ Q`, whose features are
computed from power sums by Newton's identities, and a real linear read-out with a constant term,
computes `Σ_{|S| = k} h(q + Σ_{j∈S} a_j)` exactly, for every query `q` and partners `a`. The read-out
weights are `2 ĥ(ψ)` on `R`, `ĥ(ψ)` on `Q`, and the constant is `ĥ(1) C(n, k)`. -/
theorem layer_exact (h : G → ℝ) (R Q : Finset (AddChar G ℂ))
    (hR1 : (1 : AddChar G ℂ) ∉ R) (hRdisj : Disjoint R (R.image (·⁻¹)))
    (hQ : ∀ ψ ∈ Q, ψ⁻¹ = ψ ∧ ψ ≠ 1)
    (hcover : ∀ ψ : AddChar G ℂ, coeff (lift h) ψ ≠ 0 → ψ = 1 ∨ ψ ∈ R ∨ ψ⁻¹ ∈ R ∨ ψ ∈ Q)
    (q : G) (a : ι → G) (k : ℕ) :
    ∑ S ∈ (univ : Finset ι).powersetCard k, h (q + ∑ j ∈ S, a j) =
      (coeff (lift h) 1).re * (Fintype.card ι).choose k
        + ∑ ψ ∈ R, 2 * readout (coeff (lift h) ψ) (feature ψ q a k)
        + ∑ ψ ∈ Q, readout (coeff (lift h) ψ) (feature ψ q a k) := by
  set f : AddChar G ℂ → ℝ := fun ψ => readout (coeff (lift h) ψ) (feature ψ q a k) with hf
  -- the sum as the real part of the character expansion
  have hC : ((∑ S ∈ (univ : Finset ι).powersetCard k, h (q + ∑ j ∈ S, a j) : ℝ) : ℂ) =
      ∑ ψ : AddChar G ℂ, coeff (lift h) ψ * feature ψ q a k := by
    push_cast
    have := subset_sum_eq (lift h) q a k
    simp only [lift] at this
    rw [this]
    simp [feature_char]
  have hre : ∑ S ∈ (univ : Finset ι).powersetCard k, h (q + ∑ j ∈ S, a j) = ∑ ψ, f ψ := by
    have := congrArg Complex.re hC
    rw [Complex.ofReal_re] at this
    rw [this, Complex.re_sum]
    simp [hf, readout_eq]
  rw [hre]
  -- restrict to the support
  have hsub : ∑ ψ, f ψ = ∑ ψ ∈ (({1} ∪ R) ∪ R.image (·⁻¹)) ∪ Q, f ψ := by
    symm
    refine Finset.sum_subset (Finset.subset_univ _) fun ψ _ hψ => ?_
    by_cases h0 : coeff (lift h) ψ = 0
    · simp [hf, readout, h0]
    · exfalso
      apply hψ
      rcases hcover ψ h0 with h1 | h1 | h1 | h1
      · simp [h1]
      · simp [h1]
      · simp only [Finset.mem_union, Finset.mem_singleton, Finset.mem_image]
        exact Or.inl (Or.inr ⟨ψ⁻¹, h1, inv_inv ψ⟩)
      · simp [h1]
  rw [hsub]
  have hd1 : Disjoint ({1} : Finset (AddChar G ℂ)) R := Finset.disjoint_singleton_left.2 hR1
  have hd2 : Disjoint ({1} ∪ R) (R.image (·⁻¹)) := by
    rw [Finset.disjoint_union_left]
    refine ⟨Finset.disjoint_singleton_left.2 ?_, hRdisj⟩
    simp only [Finset.mem_image, not_exists, not_and]
    intro χ hχ h1
    exact hR1 (by rw [inv_eq_one] at h1; rwa [h1] at hχ)
  have hd3 : Disjoint (({1} ∪ R) ∪ R.image (·⁻¹)) Q := by
    rw [Finset.disjoint_right]
    intro ψ hψ
    obtain ⟨hinv, hne⟩ := hQ ψ hψ
    have hnotR : ψ ∉ R := by
      intro hR
      exact Finset.disjoint_left.1 hRdisj hR (Finset.mem_image.2 ⟨ψ, hR, hinv⟩)
    simp only [Finset.mem_union, Finset.mem_singleton, Finset.mem_image, not_or, not_exists, not_and]
    refine ⟨⟨hne, hnotR⟩, fun χ hχ hχψ => hnotR ?_⟩
    have : χ = ψ := by rw [← hinv, ← hχψ, inv_inv]
    rwa [← this]
  rw [Finset.sum_union hd3, Finset.sum_union hd2, Finset.sum_union hd1, Finset.sum_singleton,
    Finset.sum_image (fun x _ y _ hxy => inv_injective hxy)]
  have hinvsum : ∑ ψ ∈ R, f ψ⁻¹ = ∑ ψ ∈ R, f ψ :=
    Finset.sum_congr rfl fun ψ _ => by simp only [hf]; exact readout_inv h ψ q a k
  rw [hinvsum]
  have h1 : f 1 = (coeff (lift h) 1).re * (Fintype.card ι).choose k := by
    simp only [hf, readout_eq, feature_char, AddChar.one_apply, one_mul]
    rw [esymm_one_const]
    simp
  show f 1 + ∑ ψ ∈ R, f ψ + ∑ ψ ∈ R, f ψ + ∑ ψ ∈ Q, f ψ =
    _ + ∑ ψ ∈ R, 2 * f ψ + ∑ ψ ∈ Q, f ψ
  rw [h1, ← Finset.mul_sum]
  ring

omit [DecidableEq G] in
/-- The conjugate of a character in the support of `ĥ` is in the support. -/
lemma coeff_inv_ne_zero (h : G → ℝ) (ψ : AddChar G ℂ) (hψ : coeff (lift h) ψ ≠ 0) :
    coeff (lift h) ψ⁻¹ ≠ 0 := by
  rw [coeff_inv]
  exact (map_ne_zero _).2 hψ

omit [DecidableEq G] in
/-- **Channel count.** For every real `h` there are channel sets `R`, `Q` as in `layer_exact`, with `Q`
the real non-trivial characters in the support of `ĥ` and `2 |R| + |Q| = #{ψ ≠ 1 : ĥ ψ ≠ 0}`. The layer
then has `|R| + |Q|` channels: one per conjugate pair of non-trivial characters in the support. -/
theorem exists_channels (h : G → ℝ) : ∃ R Q : Finset (AddChar G ℂ),
    (1 : AddChar G ℂ) ∉ R ∧ Disjoint R (R.image (·⁻¹)) ∧ (∀ ψ ∈ Q, ψ⁻¹ = ψ ∧ ψ ≠ 1) ∧
    (∀ ψ : AddChar G ℂ, coeff (lift h) ψ ≠ 0 → ψ = 1 ∨ ψ ∈ R ∨ ψ⁻¹ ∈ R ∨ ψ ∈ Q) ∧
    Q = univ.filter (fun ψ => ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0 ∧ ψ⁻¹ = ψ) ∧
    2 * R.card + Q.card = (univ.filter fun ψ : AddChar G ℂ => ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0).card := by
  let e := Fintype.equivFin (AddChar G ℂ)
  set Sp := univ.filter fun ψ : AddChar G ℂ => ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0 with hSp
  set R := Sp.filter fun ψ => e ψ < e ψ⁻¹ with hRdef
  set Q := univ.filter (fun ψ : AddChar G ℂ => ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0 ∧ ψ⁻¹ = ψ) with hQdef
  have memSp : ∀ ψ, ψ ∈ Sp ↔ ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0 := fun ψ => by simp [hSp]
  have memR : ∀ ψ, ψ ∈ R ↔ (ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0) ∧ e ψ < e ψ⁻¹ := fun ψ => by
    simp [hRdef, memSp]
  have memQ : ∀ ψ, ψ ∈ Q ↔ ψ ≠ 1 ∧ coeff (lift h) ψ ≠ 0 ∧ ψ⁻¹ = ψ := fun ψ => by simp [hQdef]
  have hRdisj : Disjoint R (R.image (·⁻¹)) := by
    rw [Finset.disjoint_left]
    intro ψ hψ hψ'
    obtain ⟨χ, hχ, hχψ⟩ := Finset.mem_image.1 hψ'
    have a1 := ((memR ψ).1 hψ).2
    have a2 := ((memR χ).1 hχ).2
    rw [← hχψ, inv_inv] at a1
    exact lt_asymm a1 a2
  refine ⟨R, Q, fun h1 => ((memR 1).1 h1).1.1 rfl, hRdisj, fun ψ hψ => ⟨((memQ ψ).1 hψ).2.2,
    ((memQ ψ).1 hψ).1⟩, ?_, rfl, ?_⟩
  · intro ψ hψ
    by_cases h1 : ψ = 1
    · exact Or.inl h1
    by_cases hq : ψ⁻¹ = ψ
    · exact Or.inr (Or.inr (Or.inr ((memQ ψ).2 ⟨h1, hψ, hq⟩)))
    have hne : e ψ ≠ e ψ⁻¹ := fun he => hq (e.injective he).symm
    rcases lt_or_gt_of_ne hne with hlt | hgt
    · exact Or.inr (Or.inl ((memR ψ).2 ⟨⟨h1, hψ⟩, hlt⟩))
    · refine Or.inr (Or.inr (Or.inl ((memR ψ⁻¹).2 ⟨⟨fun h' => h1 (inv_eq_one.1 h'),
        coeff_inv_ne_zero h ψ hψ⟩, ?_⟩)))
      rwa [inv_inv]
  · -- Sp is the disjoint union of R, its inverses and Q
    have hSpeq : Sp = (R ∪ R.image (·⁻¹)) ∪ Q := by
      ext ψ
      simp only [Finset.mem_union, Finset.mem_image]
      constructor
      · intro hψ
        obtain ⟨h1, hc⟩ := (memSp ψ).1 hψ
        by_cases hq : ψ⁻¹ = ψ
        · exact Or.inr ((memQ ψ).2 ⟨h1, hc, hq⟩)
        have hne : e ψ ≠ e ψ⁻¹ := fun he => hq (e.injective he).symm
        rcases lt_or_gt_of_ne hne with hlt | hgt
        · exact Or.inl (Or.inl ((memR ψ).2 ⟨⟨h1, hc⟩, hlt⟩))
        · refine Or.inl (Or.inr ⟨ψ⁻¹, (memR ψ⁻¹).2 ⟨⟨fun h' => h1 (inv_eq_one.1 h'),
            coeff_inv_ne_zero h ψ hc⟩, ?_⟩, inv_inv ψ⟩)
          rwa [inv_inv]
      · rintro ((hψ | ⟨χ, hχ, rfl⟩) | hψ)
        · exact (memSp ψ).2 ((memR ψ).1 hψ).1
        · obtain ⟨⟨h1, hc⟩, _⟩ := (memR χ).1 hχ
          exact (memSp _).2 ⟨fun h' => h1 (inv_eq_one.1 h'), coeff_inv_ne_zero h χ hc⟩
        · obtain ⟨h1, hc, _⟩ := (memQ ψ).1 hψ
          exact (memSp ψ).2 ⟨h1, hc⟩
    have hdQ : Disjoint (R ∪ R.image (·⁻¹)) Q := by
      rw [Finset.disjoint_right]
      intro ψ hψ
      obtain ⟨_, _, hinv⟩ := (memQ ψ).1 hψ
      simp only [Finset.mem_union, Finset.mem_image, not_or, not_exists, not_and]
      have hnot : ψ ∉ R := fun hR => by
        have := ((memR ψ).1 hR).2
        rw [hinv] at this
        exact lt_irrefl _ this
      refine ⟨hnot, fun χ hχ hχψ => hnot ?_⟩
      have : χ = ψ := by rw [← hinv, ← hχψ, inv_inv]
      rwa [← this]
    rw [hSpeq, Finset.card_union_of_disjoint hdQ, Finset.card_union_of_disjoint hRdisj,
      Finset.card_image_of_injective _ inv_injective]
    ring

omit [DecidableEq G] in
/-- For a group of odd order, the only real character is the trivial one. -/
theorem inv_eq_self_of_odd (hodd : Odd (Fintype.card G)) (ψ : AddChar G ℂ) (hψ : ψ⁻¹ = ψ) :
    ψ = 1 := by
  obtain ⟨m, hm⟩ := hodd
  ext x
  have two : ∀ y : G, ψ (2 • y) = 1 := fun y => by
    rw [two_nsmul, AddChar.map_add_eq_mul]
    have : ψ⁻¹ y = ψ y := by rw [hψ]
    rw [AddChar.inv_apply] at this
    nth_rewrite 1 [← this]
    rw [← AddChar.map_add_eq_mul, neg_add_cancel, AddChar.map_zero_eq_one]
  have hx : x = (m + 1) • (2 • x) := by
    rw [← mul_nsmul', show (m + 1) * 2 = Fintype.card G + 1 by omega, add_nsmul, card_nsmul_eq_zero,
      zero_add, one_nsmul]
  rw [hx, AddChar.map_nsmul_eq_pow, two, one_pow, AddChar.one_apply]

/-- **Corollary (counting, odd order).** For a finite abelian group of odd order there is a set `R` of
`(|G| - 1)/2` characters, one from each conjugate pair of non-trivial characters, such that for every
query `q` and partners `a`, the number `N` of `k`-subsets of partners with `q + Σ_S a = 0` satisfies
`|G| · N = C(n, k) + 2 Σ_{ψ ∈ R} Re(ψ(q) e_k(ψ ∘ a))`, with `e_k` computed from power sums. -/
theorem count_odd (hodd : Odd (Fintype.card G)) (k : ℕ) :
    ∃ R : Finset (AddChar G ℂ), 2 * R.card = Fintype.card G - 1 ∧
      ∀ (q : G) (a : ι → G),
        (Fintype.card G : ℝ) *
            (((univ : Finset ι).powersetCard k).filter (fun S => q + ∑ j ∈ S, a j = 0)).card =
          (Fintype.card ι).choose k + 2 * ∑ ψ ∈ R, (feature ψ q a k).re := by
  set δ : G → ℝ := fun x => if x = 0 then 1 else 0 with hδ
  have hcoeff : ∀ ψ : AddChar G ℂ, coeff (lift δ) ψ = (Fintype.card G : ℂ)⁻¹ := fun ψ => by
    have : ∑ y, lift δ y * ψ (-y) = 1 := by
      rw [Finset.sum_eq_single (0 : G)]
      · simp [lift, hδ]
      · intro y _ hy
        simp [lift, hδ, hy]
      · intro h0
        exact absurd (Finset.mem_univ _) h0
    rw [coeff, this, mul_one]
  have hG : (Fintype.card G : ℂ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  obtain ⟨R, Q, hR1, hRdisj, hQ, hcover, hQdef, hcard⟩ := exists_channels δ
  have hQe : Q = ∅ := by
    rw [Finset.eq_empty_iff_forall_notMem]
    intro ψ hψ
    obtain ⟨hinv, hne⟩ := hQ ψ hψ
    exact hne (inv_eq_self_of_odd hodd ψ hinv)
  refine ⟨R, ?_, fun q a => ?_⟩
  · rw [hQe, Finset.card_empty, add_zero] at hcard
    rw [hcard]
    have : (univ.filter fun ψ : AddChar G ℂ => ψ ≠ 1 ∧ coeff (lift δ) ψ ≠ 0) = univ.erase 1 := by
      ext ψ
      simp [hcoeff, hG]
    rw [this, Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, AddChar.card_eq]
  · have hl := layer_exact δ R Q hR1 hRdisj hQ hcover q a k
    rw [hQe, Finset.sum_empty, add_zero] at hl
    have hsum : ∑ S ∈ (univ : Finset ι).powersetCard k, δ (q + ∑ j ∈ S, a j) =
        (((univ : Finset ι).powersetCard k).filter (fun S => q + ∑ j ∈ S, a j = 0)).card := by
      simp [hδ]
    rw [← hsum, hl]
    have hre : ∀ ψ : AddChar G ℂ, readout (coeff (lift δ) ψ) (feature ψ q a k) =
        (Fintype.card G : ℝ)⁻¹ * (feature ψ q a k).re := fun ψ => by
      rw [readout, hcoeff]
      have h1 : ((Fintype.card G : ℂ)⁻¹).re = (Fintype.card G : ℝ)⁻¹ := by
        rw [show ((Fintype.card G : ℂ)⁻¹) = (((Fintype.card G : ℝ)⁻¹ : ℝ) : ℂ) by push_cast; rfl,
          Complex.ofReal_re]
      have h2 : ((Fintype.card G : ℂ)⁻¹).im = 0 := by
        rw [show ((Fintype.card G : ℂ)⁻¹) = (((Fintype.card G : ℝ)⁻¹ : ℝ) : ℂ) by push_cast; rfl,
          Complex.ofReal_im]
      rw [h1, h2]
      ring
    have h1 : ((Fintype.card G : ℂ)⁻¹).re = (Fintype.card G : ℝ)⁻¹ := by
      rw [show ((Fintype.card G : ℂ)⁻¹) = (((Fintype.card G : ℝ)⁻¹ : ℝ) : ℂ) by push_cast; rfl,
        Complex.ofReal_re]
    simp only [hre]
    rw [hcoeff, h1]
    have hG' : (Fintype.card G : ℝ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
    rw [show ∑ ψ ∈ R, 2 * ((Fintype.card G : ℝ)⁻¹ * (feature ψ q a k).re) =
        2 * (Fintype.card G : ℝ)⁻¹ * ∑ ψ ∈ R, (feature ψ q a k).re by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun _ _ => by ring]
    field_simp

end Layer

/-! ### SET, four-card SET and Match-k -/

section Corollaries

/-- The cards of SET: four attributes, each with three values. -/
abbrev Card := Fin 4 → ZMod 3

lemma card_Card : Fintype.card Card = 81 := by simp [Card]

lemma odd_card_Card : Odd (Fintype.card Card) := by rw [card_Card]; decide

/-- **Proposition 2 (SET).** There are 40 characters of `(ℤ/3)⁴`, one from each conjugate
pair of non-trivial characters, such that for every card in hand `q` and table `a`, the number `N` of
unordered pairs of table cards that complete a set with `q` satisfies
`81 N = C(n, 2) + 2 Re Σ_{c=1}^{40} χ_c(q) e_{2,c}`. -/
theorem set_count {ι : Type*} [Fintype ι] :
    ∃ R : Finset (AddChar Card ℂ), R.card = 40 ∧
      ∀ (q : Card) (a : ι → Card),
        (81 : ℝ) * (((univ : Finset ι).powersetCard 2).filter (fun S => q + ∑ j ∈ S, a j = 0)).card =
          (Fintype.card ι).choose 2 + 2 * ∑ ψ ∈ R, (feature ψ q a 2).re := by
  obtain ⟨R, hR, hcount⟩ := count_odd (ι := ι) odd_card_Card 2
  refine ⟨R, by rw [card_Card] at hR; omega, fun q a => ?_⟩
  have := hcount q a
  rwa [card_Card, Nat.cast_ofNat] at this

/-- **Four-card SET** (`k = 3`): the same 40 channels count the triples of table cards that sum with the
card in hand to zero. -/
theorem set4_count {ι : Type*} [Fintype ι] :
    ∃ R : Finset (AddChar Card ℂ), R.card = 40 ∧
      ∀ (q : Card) (a : ι → Card),
        (81 : ℝ) * (((univ : Finset ι).powersetCard 3).filter (fun S => q + ∑ j ∈ S, a j = 0)).card =
          (Fintype.card ι).choose 3 + 2 * ∑ ψ ∈ R, (feature ψ q a 3).re := by
  obtain ⟨R, hR, hcount⟩ := count_odd (ι := ι) odd_card_Card 3
  refine ⟨R, by rw [card_Card] at hR; omega, fun q a => ?_⟩
  have := hcount q a
  rwa [card_Card, Nat.cast_ofNat] at this

/-- **Match-(k+1) modulo an odd `M`**: `(M - 1)/2` channels count the `k`-subsets of partners whose sum
with the query is `0 mod M`. -/
theorem match_count (M : ℕ) [NeZero M] (hM : Odd M) (k : ℕ) {ι : Type*} [Fintype ι] :
    ∃ R : Finset (AddChar (ZMod M) ℂ), 2 * R.card = M - 1 ∧
      ∀ (q : ZMod M) (a : ι → ZMod M),
        (M : ℝ) * (((univ : Finset ι).powersetCard k).filter (fun S => q + ∑ j ∈ S, a j = 0)).card =
          (Fintype.card ι).choose k + 2 * ∑ ψ ∈ R, (feature ψ q a k).re := by
  have hc : Fintype.card (ZMod M) = M := ZMod.card M
  obtain ⟨R, hR, hcount⟩ := count_odd (ι := ι) (G := ZMod M) (by rw [hc]; exact hM) k
  refine ⟨R, by rw [hR, hc], fun q a => ?_⟩
  have := hcount q a
  rwa [hc] at this

end Corollaries

end Newton
