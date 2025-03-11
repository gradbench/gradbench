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
    mainParams := ⊞ (i : Fin stlen × Fin 2) => ⊞ (j : Fin 4 × Fin d) =>
      (data.main_params.get! (toFin i))[(toFin j)]!
    extraParams := ⊞ (i : Fin 3) (j : Fin d) =>
      (data.extra_params.get! i)[j]!
    state := ⊞ (i : Fin stlen × Fin 2) => ⊞ (j : Fin d) =>
      (data.state.get! (toFin i))[(toFin j)]!
    sequence := ⊞ (i : Fin lenseq) => ⊞ (j : Fin d) =>
      (data.sequence.get! i)[j]!
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
     let (i₁,i₂,j₁,j₂) := fromFin i
     MatrixType.toMatrix data.mainParams[i₁,i₂] j₁ j₂)
  let a₂ := Array.ofFn (fun i =>
     let (i,j) := fromFin i
     MatrixType.toMatrix data.extraParams i j)
  a₁ ++ a₂

instance : ToJson (LSTMOutput) where
  toJson x := toJson x.toArray
