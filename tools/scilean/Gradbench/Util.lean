import Lean

open Lean
open Except FromJson ToJson

namespace Gradbench

structure Definition where
  id : Int
  kind : String
  module : String
deriving FromJson

structure Params where
  id : Int
  kind : String
  module : String
  function : String
  input : Json
deriving FromJson

structure Timing where
  name : String
  nanoseconds : Nat
deriving ToJson

structure Output where
  output : Json
  timings : List Timing
deriving Inhabited

structure Response where
  id : Int
  success : Bool
  output : Json
  timings : List Timing
deriving ToJson

structure Runs where
  min_runs : Nat
  min_seconds : Float
deriving ToJson, FromJson

def wrap [FromJson a] [ToJson b] (f : a -> b) (x' : Json)
    : Except String (IO Output) := do
  let runs : Runs := fromJson? x' |>.toOption |>.getD { min_runs := 1, min_seconds := 0.0 }
  let x : a <- fromJson? x'
  return do
    let mut totalRuns := 0
    let mut totalNs : Nat := 0
    let mut timings : List Timing := []
    let mut y' : Json := default
    while (totalRuns < runs.min_runs) || (totalNs.toFloat < 10^9 * runs.min_seconds) do
      let start <- IO.monoNanosNow
      let y ← pure (f x)
      let done <- IO.monoNanosNow
      if totalRuns = 0 then
        y' := toJson y
      let time := done - start
      totalNs := totalNs + time
      totalRuns := totalRuns + 1
      timings := { name := "evaluate", nanoseconds := time } :: timings
    return { output := y', timings }
