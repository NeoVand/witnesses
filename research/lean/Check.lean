import PairSearch.Sharp
import Newton.Characters
import Framework.Gating

/-!
# The declarations the paper names

Each declaration named beside a theorem or proposition of the paper, or in its Proofs, is shown with
`#check` and with `#print axioms`. Every one depends only on `propext`, `Classical.choice` and
`Quot.sound`; none on `sorryAx`.

- Theorem 1(i), depth law: `PairSearch.pair_states_search_sharp`, `PairSearch.pair_states_search_looped`.
- Theorem 2, size law: `Framework.size_law_iff`, `Framework.size_law_equality`, `Framework.gated_existence`,
  `Framework.gated_existence_log`, `Framework.background_le`, `Framework.tight`, `Framework.softmax_existence`.
- Proposition 2, characters count set pairs: `Newton.set_count`, `Newton.newton_identities`, and the general
  count over any finite abelian group and any subset size, `Newton.count_eq`.
-/

/-! ### Theorem 1(i): depth law -/

#check @PairSearch.pair_states_search_sharp
#print axioms PairSearch.pair_states_search_sharp
#check @PairSearch.pair_states_search_looped
#print axioms PairSearch.pair_states_search_looped

/-! ### Theorem 2: size law -/

#check @Framework.size_law_iff
#print axioms Framework.size_law_iff
#check @Framework.size_law_equality
#print axioms Framework.size_law_equality
#check @Framework.gated_existence
#print axioms Framework.gated_existence
#check @Framework.gated_existence_log
#print axioms Framework.gated_existence_log
#check @Framework.background_le
#print axioms Framework.background_le
#check @Framework.tight
#print axioms Framework.tight
#check @Framework.softmax_existence
#print axioms Framework.softmax_existence

/-! ### Proposition 2: characters count set pairs at linear cost -/

#check @Newton.set_count
#print axioms Newton.set_count
#check @Newton.newton_identities
#print axioms Newton.newton_identities
#check @Newton.count_eq
#print axioms Newton.count_eq

/-! ### A concrete instance of Theorem 1(i)

The path `0 → 1 → 2 → 3 → 4` with goal `4`. From `c = 0` the child `v = 1` is three steps from the goal, so
two layers (reach within `2^2 - 1 = 3` steps of `v`) accept it and one layer (within `2^1 - 1 = 1`) does not. -/

namespace PairSearch.Example

open PairSearch Matrix

/-- The path graph on five vertices, `u → u + 1`. -/
def path5 : Fin 5 → Fin 5 → Bool := fun u v => v.val = u.val + 1

example : 0 < readoutSharp ⬝ᵥ sharpNetwork 5 2 (embedSharp (initSharp path5 4)) 0 1 - 1 / 2 :=
  (sharp_network_score_pos_iff path5 4 0 1 2 (by norm_num)).2
    ⟨by decide, ⟨Walk.cons (v := 2) (by decide)
      (Walk.cons (v := 3) (by decide) (Walk.cons (by decide) (Walk.nil 4))), by simp⟩⟩

example : ¬ 0 < readoutSharp ⬝ᵥ sharpNetwork 5 1 (embedSharp (initSharp path5 4)) 0 1 - 1 / 2 := by
  rw [sharp_network_score_pos_iff _ _ _ _ _ le_rfl, show (2 : ℕ) ^ 1 - 1 = 1 from rfl,
    reachWithin_one_iff]
  decide

end PairSearch.Example
