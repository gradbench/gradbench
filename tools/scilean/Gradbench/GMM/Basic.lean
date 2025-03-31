import SciLean

import Gradbench.GMM.Data

open SciLean Scalar

namespace Gradbench.GMM

set_option pp.deepTerms true
set_option pp.proofs false

set_default_scalar Float

open VectorType in
/-- unlack `logdiag` and `lt` to lower triangular matrix -/
def unpackQ {d : Nat} (logdiag : Float^[d]) (lt : Float^[((d-1)*d/2)]) : Float^[d,d]  :=
  ⊞ (ij : Idx d × Idx d) =>
    let' (i,j) := ij
    if h : i < j then 0
       else if h' : i == j then exp (logdiag[i])
       else
         let idx : Idx ((d-1)*d/2) := ⟨d.toUSize*j.1 + i.1 - j.1 - 1 - (j.1 * (j.1+1))/2,
                                       have := h; have := h'; sorry_proof⟩
         (lt[idx])


abbrev_data_synth unpackQ in logdiag lt : HasRevFDeriv Float by
  unfold unpackQ; dsimp
  data_synth => enter[3]; lsimp [↓let_ite_normalize]

def logWishartPrior {k d : Nat} (Qs : Float^[d,d]^[k]) (qsums : Float^[k]) (wishartGamma : Float) (wishartM : Nat) :=
    let p := d
    let n := p + wishartM + 1
    let c := (n * p) * (log wishartGamma - 0.5 * log 2) - (logMultiGamma (0.5 * n.toFloat) p)
    let frobenius : Float := ‖Qs‖₂²
    let sumQs : Float := qsums.sum
    0.5 * wishartGamma * wishartGamma * frobenius - wishartM * sumQs - k * c


abbrev_data_synth logWishartPrior in Qs qsums : HasRevFDeriv Float by
  unfold logWishartPrior;
  data_synth => enter[3]; lsimp


open VectorType
def gmmObjective {d k n : Nat}
      (alphas: Float^[k]) (means: Float^[k,d])
      (logdiag : Float^[k,d]) (lt : Float^[k,((d-1)*d)/2])
      (x : Float^[n,d]) (wishartGamma : Float) (wishartM: Nat) :=
    let C := -(n * d * 0.5 * log (2 * π))

    -- qsAndSums
    let Qs := ⊞ i => unpackQ (logdiag.row i) (lt.row i)
    let qsums := ⊞ i => (logdiag.row i).sum

    let slse : Float :=
      ∑ᴵ (i : Idx n), (⊞ (j : Idx k) =>
          alphas[j]
          +
          qsums[j]
          -
          0.5 * ‖Qs[j] * ((x.row i) - (means.row j))‖₂²).logsumexp

    C + slse - n * alphas.logsumexp + logWishartPrior Qs qsums wishartGamma wishartM


abbrev_data_synth gmmObjective in alphas means logdiag lt : HasRevFDeriv Float by
  unfold gmmObjective
  data_synth => enter[3]; lsimp

set_option linter.unusedVariables false in
def gmmJacobian {d k n : Nat}
      (alphas: Float^[k]) (means: Float^[k,d])
      (logdiag : Float^[k,d]) (lt : Float^[k,((d-1)*d)/2])
      (x : Float^[n,d]) (wishartGamma : Float) (wishartM: Nat)
      {f'} (deriv : HasRevFDeriv Float (fun (a,m,l,lt) =>
              gmmObjective (d:=d) (k:=k) (n:=n) a m l lt x wishartGamma wishartM) f')
      {grad} (simp : grad = (f' (alphas,means,logdiag,lt)).2 1) := grad


def objective (data : GMMData) : Float :=
  let ⟨m,gamma,alpha,means,logdiag,lt,x⟩ := data
  gmmObjective alpha means logdiag lt x gamma m

def jacobian (data : GMMData) : GMMGradientData :=
  let ⟨m,gamma,alpha,means,logdiag,lt,x⟩ := data
  let (alpha,means,logdiag,lt) :=
    gmmJacobian alpha means logdiag lt x gamma m
      (deriv := by data_synth)
      (simp := by conv => rhs; lsimp)
  ⟨alpha,means,logdiag,lt⟩
