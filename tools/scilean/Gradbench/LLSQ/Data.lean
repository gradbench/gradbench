import SciLean

open SciLean Lean

namespace Gradbench.LLSQ

structure LLSQInputRaw where
  x  : Array Float
  n : Nat
deriving ToJson, FromJson

structure LLSQInput where
  {m : Nat}
  x : Float^[m]
  n : Nat

open IndexType in
def LLSQInputRaw.toLLSQInput (data : LLSQInputRaw) : LLSQInput :=
  let m := data.x.size
  {
    m := m
    x := ⊞ (i : Idx m) => data.x[i.1]!
    n := data.n
  }

instance : FromJson LLSQInput where
  fromJson? json := do
    let data : LLSQInputRaw ← fromJson? json
    return data.toLLSQInput

instance : ToJson (DataArray Float) where
  toJson x := toJson (Array.ofFn (fun i => x.get i.toIdx))
