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

instance : ToJson GMMGradientData where
  toJson data :=
    let alphaData := Array.ofFn (fun i => data.alpha[i.toIdx])
    let muData := Array.ofFn (fun i => Array.ofFn (fun j => data.means[(i.toIdx, j.toIdx)]))
    let qData := Array.ofFn (fun i => Array.ofFn (fun j => data.logdiag[(i.toIdx, j.toIdx)]))
    let lData := Array.ofFn (fun i => Array.ofFn (fun j => data.lt[(i.toIdx, j.toIdx)]))
    Json.mkObj [("alpha", toJson alphaData),
                ("mu", toJson muData),
                ("q", toJson qData),
                ("l", toJson lData)]
