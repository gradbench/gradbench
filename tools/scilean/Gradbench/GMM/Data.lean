import Lean
import SciLean

open Lean ToJson FromJson SciLean

namespace Gradbench.GMM


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
  means : Float^[d]^[k]
  logdiag : Float^[d]^[k]
  lt : Float^[((d-1)*d)/2]^[k]
  x : Float^[d]^[k]


def GMMDataRaw.toGMMData (d : GMMDataRaw) : (d k n : Nat) × GMMData d k n :=
  ⟨d.d,d.k,d.n,{
    m := d.m
    gamma := d.gamma
    alpha := ⊞ (i : Fin d.k) => d.alpha[i.1]!
    means := ⊞ (i : Fin d.k) (j : Fin d.d) => (d.means.get! i.1 |>.get! j.1) |>.curry
    logdiag := ⊞ (i : Fin d.k) (j : Fin d.d) => (d.icf.get! i.1 |>.get! j.1) |>.curry
    lt := ⊞ (i : Fin d.k) (j : Fin (((d.d-1)*d.d)/2)) => (d.icf.get! i.1 |>.get! (d.d+j.1)) |>.curry
    x := ⊞ (i : Fin d.k) (j : Fin d.d) => (d.x.get! i.1 |>.get! j.1) |>.curry
   }⟩
