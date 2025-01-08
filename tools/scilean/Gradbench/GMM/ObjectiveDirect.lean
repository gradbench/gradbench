import SciLean
import SciLean.Analysis.SpecialFunctions.MultiGamma

import Gradbench.GMM.Data

open SciLean Scalar

namespace Gradbench.GMM


local macro (priority:=high+1) "Float^[" M:term ", " N:term "]" : term =>
  `(FloatMatrix' .RowMajor .normal (Fin $M) (Fin $N))

local macro (priority:=high+1) "Float^[" N:term "]" : term =>
  `(FloatVector (Fin $N))

-- notation: `v[i] := vᵢ`
local macro (priority:=high+10) id:ident noWs "[" i:term "]" " := " v:term : doElem =>
   `(doElem| $id:ident := VectorType.set $id $i $v)

-- notation: `A[i,:] := r`
local macro (priority:=high+10) id:ident noWs "[" i:term "," ":" "]" " := " v:term : doElem =>
   `(doElem| $id:ident := MatrixType.updateRow $id $i $v)

-- notation: `⊞ i => vᵢ`
open Lean Parser Term in
local macro (priority:=high+10) "⊞" i:funBinder " => " b:term : term =>
   `(term| VectorType.fromVec  fun $i => $b)

instance {n : Nat} : GetElem (Float^[n]) (Fin n) Float (fun _ _ => True) :=
  ⟨fun x i _ => VectorType.toVec x i⟩

set_default_scalar Float

/-- unlack `logdiag` and `lt` to lower triangular matrix -/
def unpackQ {d : Nat} (logdiag : Float^[d]) (lt : Float^[((d-1)*d/2)]) : Float^[d,d]  :=
  MatrixType.fromMatrix fun (i j : Fin d) =>
    if i < j then 0
       else if i == j then exp logdiag[i]
       else
         let idx : Fin ((d-1)*d/2) := ⟨d*j.1 + i.1 - j.1 - 1 - (j.1 * (j.1+1))/2, sorry⟩
         lt[idx]


def logWishartPrior {k d : Nat} (Qs : Float^[d,d]^[k]) (qsums : Float^[k]) (wishartGamma : Float) (wishartM : Nat) :=
    let p := d
    let n := p + wishartM + 1
    let c := (n * p) * (log wishartGamma - 0.5 * log 2) - (logMultiGamma (0.5 * n.toFloat) p)
    let frobenius : Float := ‖Qs‖₂²
    let sumQs : Float := VectorType.sum qsums
    0.5 * wishartGamma * wishartGamma * frobenius - wishartM * sumQs - k * c

open VectorType
def gmmObjective {d k n : Nat}
      (alphas: Float^[k]) (means: Float^[k,d])
      (logdiag : Float^[k,d]) (lt : Float^[k,((d-1)*d)/2])
      (x : Float^[n,d]) (wishartGamma : Float) (wishartM: Nat) :=
    let C := -(n * d * 0.5 * log (2 * π))

    -- qsAndSums
    let Qs := ⊞ i => unpackQ (MatrixType.row logdiag i) (MatrixType.row lt i)
    let qsums := VectorType.fromVec fun i => VectorType.sum (MatrixType.row logdiag i)

    let slse : Float :=
      ∑ (i : Fin n), logsumexp (VectorType.fromVec (X:=FloatVector _)
        fun (j : Fin k) =>
          toVec alphas j
          +
          toVec qsums j
          -
          0.5 * ‖MatrixType.gemv 1 1 Qs[j] ((MatrixType.row x i) - (MatrixType.row means j)) 0‖₂²)

    C + slse  - n * VectorType.logsumexp alphas + logWishartPrior Qs qsums wishartGamma wishartM


def objective (data : GMMDataRaw) : Float :=
  let ⟨d,k,n,data⟩ := data.toGMMData
  gmmObjective data.alpha data.means data.logdiag data.lt data.x data.gamma data.m
