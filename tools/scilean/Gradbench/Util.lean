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

def wrap [FromJson a] [ToJson b] (f : a -> b) (x' : Json)
    : Except String (IO Output) := do
  let min_runs ← x'.getObjVal? "min_runs" >>= Json.getNat?
  let min_seconds ← x'.getObjVal? "min_seconds" >>= Json.getNat?
  let x : a <- fromJson? x'
  return do
    let mut totalRuns := 0
    let mut totalNs := 0
    let mut timings : List Timing := []
    let mut y' : Json := default
    while (totalRuns < min_runs) || (totalNs < 10^9 * min_seconds) do
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
