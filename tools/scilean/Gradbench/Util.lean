import Lean

open Lean
open Except FromJson ToJson

namespace Gradbench

structure Definition where
  id: Int
  kind: String
  module: String
deriving FromJson

structure Params where
  id: Int
  kind: String
  module: String
  function: String
  input: Json
deriving FromJson

structure Timing where
  name: String
  nanoseconds: Nat
deriving ToJson

structure Output where
  output: Json
  timings: List Timing
deriving Inhabited

structure Response where
  id: Int
  output: Json
  timings: List Timing
deriving ToJson

def wrap [FromJson a] [ToJson b] (f : a -> b) (x' : Json)
    : Except String (IO Output) := do
  let x : a <- fromJson? x'
  return do
    let start <- IO.monoNanosNow
    let y ← pure (f x)
    let done <- IO.monoNanosNow
    let y' := toJson y
    let timings := [{ name := "evaluate", nanoseconds := done - start }]
    return { output := y', timings }
