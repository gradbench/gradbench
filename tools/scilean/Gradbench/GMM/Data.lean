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

structure GMMData where
  {d k n : Nat}
  m : Nat
  gamma : Float

  alpha : Float^[k]
  means : Float^[k,d]
  logdiag : Float^[k,d]
  lt : Float^[k,((d-1)*d)/2]
  x : Float^[n,d]

def GMMDataRaw.toGMMData (d : GMMDataRaw) : GMMData :={
  d := d.d
  k := d.k
  n := d.n
  m := d.m
  gamma := d.gamma
  alpha := VectorType.fromVec fun (i : Fin d.k) => d.alpha[i.1]!
  means := MatrixType.fromMatrix fun (i : Fin d.k) (j : Fin d.d) => (d.means.get! i.1 |>.get! j.1)
  logdiag := MatrixType.fromMatrix fun (i : Fin d.k) (j : Fin d.d) => (d.icf.get! i.1 |>.get! j.1)
  lt := MatrixType.fromMatrix fun (i : Fin d.k) (j : Fin (((d.d-1)*d.d)/2)) => (d.icf.get! i.1 |>.get! (d.d+j.1))
  x := MatrixType.fromMatrix fun (i : Fin d.n) (j : Fin d.d) => (d.x.get! i.1 |>.get! j.1)
}

instance : FromJson GMMData where
  fromJson? json := do
    let data : GMMDataRaw ← fromJson? json
    return data.toGMMData

structure GMMGradientData where
  {d k : Nat}
  alpha : Float^[k]
  means : Float^[k,d]
  logdiag : Float^[k,d]
  lt : Float^[k,((d-1)*d)/2]

open VectorType IndexType in
def GMMGradientData.toArray (data : GMMGradientData) : Array Float :=
  let k := data.k; let d := data.d
  let alphaData := Array.ofFn (fun i => toVec data.alpha i)
  let meansData := Array.ofFn (fun idx =>
    toVec data.means (fromFin idx))
  let icfData := Array.ofFn (fun idx =>
    let (i,j) : Fin k × (Fin d ⊕ Fin (((d-1)*d)/2)) := fromFin idx
    match j with
    | .inl j => toVec data.logdiag (i,j)
    | .inr j => toVec data.lt (i,j))
  (alphaData ++ meansData ++ icfData)

instance : ToJson GMMGradientData where
  toJson data := toJson data.toArray
