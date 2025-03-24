import SciLean

open SciLean Scalar Lean ToJson FromJson

namespace Gradbench.LSTM

set_default_scalar Float

structure LSTMInputRaw where
  main_params  : Array (Array Float)
  extra_params : Array (Array Float)
  state        : Array (Array Float)
  sequence     : Array (Array Float)
deriving ToJson, FromJson

structure LSTMInput where
  {d stlen lenseq : ℕ}
  mainParams : Float^[4,d]^[stlen,2]
  extraParams : Float^[3,d]
  state : Float^[d]^[stlen,2]
  sequence : Float^[d]^[lenseq]

open IndexType in
def LSTMInputRaw.toLSTMInput (data : LSTMInputRaw) : LSTMInput :=
  let d := data.main_params[0]!.size/4
  let stlen := data.main_params.size/2
  let lenseq := data.sequence.size
  {
    d := d
    stlen := stlen
    lenseq := lenseq
    mainParams := ⊞ (i : Idx stlen × Idx 2) => ⊞ (j : Idx 4 × Idx d) =>
      (data.main_params.get! (toIdx i).toFin)[(toIdx j).toFin]!
    extraParams := ⊞ (i : Idx 3) (j : Idx d) =>
      (data.extra_params.get! i.toFin)[j.toFin]!
    state := ⊞ (i : Idx stlen × Idx 2) => ⊞ (j : Idx d) =>
      (data.state.get! (toIdx i).toFin)[(toIdx j).toFin]!
    sequence := ⊞ (i : Idx lenseq) => ⊞ (j : Idx d) =>
      (data.sequence.get! i.toFin)[j.toFin]!
  }

instance : FromJson LSTMInput where
  fromJson? json := do
    let data : LSTMInputRaw ← fromJson? json
    return data.toLSTMInput


structure LSTMOutput where
  {d slen : ℕ}
  mainParams : Float^[4,d]^[slen,2]
  extraParams : Float^[3,d]

open IndexType in
def LSTMOutput.toArray (data : LSTMOutput) : Array Float :=
  let a₁ := Array.ofFn (fun i =>
     let (i₁,i₂,j₁,j₂) := fromIdx i.toIdx
     MatrixType.toMatrix data.mainParams[i₁,i₂] j₁ j₂)
  let a₂ := Array.ofFn (fun i =>
     let (i,j) := fromIdx i.toIdx
     MatrixType.toMatrix data.extraParams i j)
  a₁ ++ a₂

instance : ToJson (LSTMOutput) where
  toJson x := toJson x.toArray
