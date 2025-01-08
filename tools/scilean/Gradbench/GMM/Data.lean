import Lean
import SciLean

open Lean ToJson FromJson SciLean

namespace Gradbench.GMM


local macro (priority:=high+1) "Float^[" M:term ", " N:term "]" : term =>
  `(FloatMatrix' .RowMajor .normal (Fin $M) (Fin $N))

local macro (priority:=high+1) "Float^[" N:term "]" : term =>
  `(FloatVector (Fin $N))


structure GMMDataRaw where
  runs : Nat
  d : Nat
  k : Nat
  n : Nat

  m : Nat
  gamma : Float

  alpha : Array Float
  means : Array (Array Float)
  icf : Array (Array Float)
  x : Array (Array Float)
deriving ToJson, FromJson


structure GMMData (d k n : Nat) where
  m : Nat
  gamma : Float

  alpha : Float^[k]
  means : Float^[k,d]
  logdiag : Float^[k,d]
  lt : Float^[k,((d-1)*d)/2]
  x : Float^[n,d]


def GMMDataRaw.toGMMData (d : GMMDataRaw) : (d k n : Nat) × GMMData d k n :=
  ⟨d.d,d.k,d.n,{
    m := d.m
    gamma := d.gamma
    alpha := VectorType.fromVec fun (i : Fin d.k) => d.alpha[i.1]!
    means := MatrixType.fromMatrix fun (i : Fin d.k) (j : Fin d.d) => (d.means.get! i.1 |>.get! j.1)
    logdiag := MatrixType.fromMatrix fun (i : Fin d.k) (j : Fin d.d) => (d.icf.get! i.1 |>.get! j.1)
    lt := MatrixType.fromMatrix fun (i : Fin d.k) (j : Fin (((d.d-1)*d.d)/2)) => (d.icf.get! i.1 |>.get! (d.d+j.1))
    x := MatrixType.fromMatrix fun (i : Fin d.n) (j : Fin d.d) => (d.x.get! i.1 |>.get! j.1)
   }⟩
