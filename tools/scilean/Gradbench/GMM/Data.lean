import Lean
import SciLean

open Lean ToJson FromJson SciLean

namespace Gradbench.GMM


structure GMMDataRaw where
  d : Nat
  k : Nat
  n : Nat

  m : Nat
  gamma : Float

  alpha : Array Float
  mu : Array (Array Float)
  q : Array (Array Float)
  l : Array (Array Float)
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
  alpha := ⊞ (i : Idx d.k) => d.alpha[i.1]!
  means := ⊞ (i : Idx d.k) (j : Idx d.d) => (d.mu.get! i.1.toNat |>.get! j.1.toNat)
  logdiag := ⊞ (i : Idx d.k) (j : Idx d.d) => (d.q.get! i.1.toNat |>.get! j.1.toNat)
  lt := ⊞ (i : Idx d.k) (j : Idx (((d.d-1)*d.d)/2)) => (d.l.get! i.1.toNat |>.get! j)
  x := ⊞ (i : Idx d.n) (j : Idx d.d) => (d.x.get! i.1.toNat |>.get! j.1.toNat)
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
  let alphaData := Array.ofFn (fun i => data.alpha[i.toIdx])
  let meansData := Array.ofFn (fun idx =>
    data.means[fromIdx idx.toIdx])
  let icfData := Array.ofFn (fun idx =>
    let (i,j) : Idx k × (Idx d ⊕ Idx (((d-1)*d)/2)) := fromIdx idx.toIdx
    match j with
    | .inl j => data.logdiag[i,j]
    | .inr j => data.lt[i,j])
  (alphaData ++ meansData ++ icfData)

instance : ToJson GMMGradientData where
  toJson data := toJson data.toArray
