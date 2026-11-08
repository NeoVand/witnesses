import Mathlib.Order.Basic
import Mathlib.Tactic.Ring

/-!
# Walks, reach within `m` steps, and the splitting lemma

Source: the proof of Theorem 1(i) (depth law) in the paper's Proofs, the sentence before equation (square):
"a walk of at most `2^(ℓ+1)` steps splits at a middle vertex `k` into two walks of at most `2^ℓ` steps".

The graph has vertices `Fin n` and adjacency `A : Fin n → Fin n → Bool`; `A u v = true` is the edge
`u → v`, so `v` is a child of `u`. A walk may repeat vertices and may have length zero.

- `Walk.exists_split`: a walk splits at any step `i ≤ length` into a prefix of length `i` and a suffix.
- `Walk.split_add`: a walk of length `≤ a + b` splits into walks of length `≤ a` and `≤ b`.
- `Walk.split_pow`: the paper's key lemma, `a = b = 2^ℓ`.
- `reachWithin_pow_succ_iff`: the relational form, reach within `2^(ℓ+1)` is the Boolean square of
  reach within `2^ℓ`.
- `reachWithin_one_iff`: reach within one step is `[I + A > 0]`, the paper's `R^(0)`.
-/

namespace PairSearch

variable {n : ℕ}

/-- A directed walk from `u` to `w` in the graph with adjacency `A`: a sequence of edges. -/
inductive Walk (A : Fin n → Fin n → Bool) : Fin n → Fin n → Type
  /-- The walk of zero steps at `u`. -/
  | nil (u : Fin n) : Walk A u u
  /-- An edge `u → v` followed by a walk from `v` to `w`. -/
  | cons {u v w : Fin n} (h : A u v = true) (p : Walk A v w) : Walk A u w

namespace Walk

variable {A : Fin n → Fin n → Bool}

/-- The number of steps (edges) of a walk. -/
def length : {u v : Fin n} → Walk A u v → ℕ
  | _, _, nil _ => 0
  | _, _, cons _ p => p.length + 1

/-- Concatenation of walks. -/
def append : {u v w : Fin n} → Walk A u v → Walk A v w → Walk A u w
  | _, _, _, nil _, q => q
  | _, _, _, cons h p, q => cons h (p.append q)

@[simp] lemma length_nil (u : Fin n) : (nil u : Walk A u u).length = 0 := rfl

@[simp] lemma length_cons {u v w : Fin n} (h : A u v = true) (p : Walk A v w) :
    (cons h p).length = p.length + 1 := rfl

@[simp] lemma length_append : ∀ {u v w : Fin n} (p : Walk A u v) (q : Walk A v w),
    (p.append q).length = p.length + q.length
  | _, _, _, nil _, q => by simp [append]
  | _, _, _, cons h p, q => by simp [append, length_append p q]; omega

/-- A walk splits at step `i`, for any `i ≤ length`, into a prefix of `i` steps and the rest. -/
theorem exists_split : ∀ {u v : Fin n} (p : Walk A u v) (i : ℕ), i ≤ p.length →
    ∃ (k : Fin n) (p₁ : Walk A u k) (p₂ : Walk A k v),
      p₁.length = i ∧ p₂.length = p.length - i
  | u, _, nil _, i, hi => ⟨u, nil u, nil u, by simp at hi ⊢; omega, by simp⟩
  | u, _, cons h p, 0, _ => ⟨u, nil u, cons h p, rfl, by simp⟩
  | _, _, cons h p, i + 1, hi => by
      obtain ⟨k, p₁, p₂, h₁, h₂⟩ := exists_split p i (by simp at hi; omega)
      exact ⟨k, cons h p₁, p₂, by simp [h₁], by simp [h₂]⟩

/-- A walk of at most `a + b` steps splits at a middle vertex into walks of at most `a` and at most
`b` steps. -/
theorem split_add {u v : Fin n} (p : Walk A u v) {a b : ℕ} (hp : p.length ≤ a + b) :
    ∃ (k : Fin n) (p₁ : Walk A u k) (p₂ : Walk A k v), p₁.length ≤ a ∧ p₂.length ≤ b := by
  obtain ⟨k, p₁, p₂, h₁, h₂⟩ := p.exists_split (min a p.length) (min_le_right _ _)
  exact ⟨k, p₁, p₂, by omega, by omega⟩

/-- **Key lemma** (proof of Theorem 1(i), before equation (square)). A walk of at most `2^(ℓ+1)` steps
splits at a middle vertex `k` into two walks of at most `2^ℓ` steps. -/
theorem split_pow {u v : Fin n} (p : Walk A u v) {ℓ : ℕ} (hp : p.length ≤ 2 ^ (ℓ + 1)) :
    ∃ (k : Fin n) (p₁ : Walk A u k) (p₂ : Walk A k v), p₁.length ≤ 2 ^ ℓ ∧ p₂.length ≤ 2 ^ ℓ :=
  p.split_add (by rw [pow_succ] at hp; omega)

end Walk

/-- `v` can be reached from `u` in at most `m` steps (zero steps included): some walk from `u` to `v`
has length at most `m`. -/
def ReachWithin (A : Fin n → Fin n → Bool) (m : ℕ) (u v : Fin n) : Prop :=
  ∃ p : Walk A u v, p.length ≤ m

section ReachWithin

variable {A : Fin n → Fin n → Bool}

lemma ReachWithin.refl (m : ℕ) (u : Fin n) : ReachWithin A m u u := ⟨Walk.nil u, by simp⟩

lemma ReachWithin.mono {m m' : ℕ} (h : m ≤ m') {u v : Fin n} (hr : ReachWithin A m u v) :
    ReachWithin A m' u v :=
  let ⟨p, hp⟩ := hr; ⟨p, hp.trans h⟩

lemma ReachWithin.trans {a b : ℕ} {u k v : Fin n} (h₁ : ReachWithin A a u k)
    (h₂ : ReachWithin A b k v) : ReachWithin A (a + b) u v :=
  let ⟨p, hp⟩ := h₁; let ⟨q, hq⟩ := h₂; ⟨p.append q, by simp; omega⟩

/-- Reach within `a + b` steps is reach within `a` steps followed by reach within `b` steps. -/
theorem reachWithin_add_iff {a b : ℕ} {u v : Fin n} :
    ReachWithin A (a + b) u v ↔ ∃ k, ReachWithin A a u k ∧ ReachWithin A b k v := by
  constructor
  · rintro ⟨p, hp⟩
    obtain ⟨k, p₁, p₂, h₁, h₂⟩ := p.split_add hp
    exact ⟨k, ⟨p₁, h₁⟩, ⟨p₂, h₂⟩⟩
  · rintro ⟨k, h₁, h₂⟩
    exact h₁.trans h₂

/-- **Equation (square), relational form.** `v` is reachable from `u` within `2^(ℓ+1)` steps iff some
middle vertex `k` is reachable from `u` within `2^ℓ` steps and reaches `v` within `2^ℓ` steps. -/
theorem reachWithin_pow_succ_iff {ℓ : ℕ} {u v : Fin n} :
    ReachWithin A (2 ^ (ℓ + 1)) u v ↔
      ∃ k, ReachWithin A (2 ^ ℓ) u k ∧ ReachWithin A (2 ^ ℓ) k v := by
  rw [pow_succ, mul_two]
  exact reachWithin_add_iff

/-- Reach within one step is equality or an edge: the paper's `R^(0) = [I + A > 0]`. -/
theorem reachWithin_one_iff {u v : Fin n} : ReachWithin A 1 u v ↔ u = v ∨ A u v = true := by
  constructor
  · rintro ⟨p, hp⟩
    cases p with
    | nil => exact Or.inl rfl
    | cons h p =>
      cases p with
      | nil => exact Or.inr h
      | cons _ _ => simp at hp
  · rintro (rfl | h)
    · exact ReachWithin.refl 1 u
    · exact ⟨Walk.cons h (Walk.nil v), by simp⟩

end ReachWithin

end PairSearch
