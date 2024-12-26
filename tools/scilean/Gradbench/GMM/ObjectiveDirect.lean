import SciLean

import Gradbench.GMM.Data

open SciLean Scalar

namespace Gradbench.GMM

set_default_scalar Float

/-- unlack `logdiag` and `lt` to lower triangular matrix -/
def unpackQ {d : Nat} (logdiag : Float^[d]) (lt : Float^[((d-1)*d/2)]) : Float^[d,d]  :=
  ⊞ i j =>
    if i < j then 0
       else if i == j then exp logdiag[i]
       else
         let idx : Fin ((d-1)*d/2) := ⟨d*j.1 + i.1 - j.1 - 1 - (j.1 * (j.1+1))/2, sorry⟩
         lt[idx]


@[extern "gradbench_lgamma"]
opaque lgamma : Float → Float

def logGammaDistrib (a : Float) (p : Nat) :=
  0.25 * p * (p - 1) * log π +
  ∑ (j : Fin p), lgamma (a + 0.5 * j.1)


def logWishartPrior {k d : Nat} (Qs : Float^[d,d]^[k]) (qsums : Float^[k]) (wishartGamma : Float) (wishartM p : Nat) :=
    let n := p + wishartM + 1
    let c := (n * p) * (log wishartGamma - 0.5 * log 2) - (logGammaDistrib (0.5 * n) p)
    let frobenius : Float := ‖Qs.uncurry‖₂²
    let sumQs : Float := qsums.sum
    0.5 * wishartGamma * wishartGamma * frobenius - wishartM * sumQs - k * c

def gmmObjective {d k n : Nat} (alphas: Float^[k]) (means: Float^[d]^[k]) (logdiag : Float^[d]^[k]) (lt : Float^[((d-1)*d)/2]^[k])
      (x : Float^[d]^[n]) (wishartGamma : Float) (wishartM: Nat) :=
    let C := -(n * d * 0.5 * log (2 * π))

    -- qsAndSums
    let Qs := ⊞ i => unpackQ logdiag[i] lt[i]
    let qsums := ⊞ i => logdiag[i].sum

    let slse : Float :=
      -- maybe Qs[j]ᵀ
      ∑ i, (⊞ j => alphas[j] + qsums[j] - 0.5 * ‖Qs[j]ᵀ  * (x[i] - means[j])‖₂²).logsumexp
    C + slse  - n * alphas.logsumexp + logWishartPrior Qs qsums wishartGamma wishartM d


def objective (data : GMMDataRaw) : Float :=
  let ⟨d,k,n,data⟩ := data.toGMMData
  gmmObjective data.alpha data.means data.logdiag data.lt data.x data.gamma data.m
