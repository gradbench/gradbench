import Lean
import SciLean

open Lean
open Except FromJson ToJson

def singleton (xs : List a) : Except String a :=
  match xs with
  | [x] => ok x
  | _ => error "expected exactly one element"

def all [Monad m] (xs : List (m a)) : m (List a) := do
  let ys <- xs.foldl (fun res x => do
    let ys <- res
    let y <- x
    return y :: ys
  ) (pure [])
  return ys.reverse

partial def readToEnd (stream : IO.FS.Stream) : IO String := do
  let rec loop (s : String) := do
    let line <- stream.getLine
    if line.isEmpty then
      return s
    else
      loop (s ++ line)
  loop ""

def square (x : Float) : Float := x * x

def double (x : Float) : Float := x + x

def resolve (name : String) : Except String (Float -> Float) :=
  match name with
  | "square" => ok square
  | "double" => ok double
  | _ => error "unknown function"

structure Argument where
  value: Float
deriving FromJson

structure Params where
  name: String
  arguments: List Argument
deriving FromJson

structure Output where
  ret: Float
  nanoseconds: Int

instance : ToJson Output where
  toJson o := Json.mkObj [
    ("return", toJson o.ret),
    ("nanoseconds", toJson o.nanoseconds),
  ]

def run (params : Params) : Except String (IO Output) := do
  let f <- resolve params.name
  let arg <- singleton params.arguments
  return do
    let start <- IO.monoNanosNow
    let y := f arg.value
    let done <- IO.monoNanosNow
    let output : Output := { ret := y, nanoseconds := done - start }
    return output

def main : IO UInt32 := do
  let stdin <- IO.getStdin
  let s <- readToEnd stdin
  let result := do
    let cfg <- Json.parse s
    let list <- Json.getObjVal? cfg "inputs"
    let inputs : List Params <- fromJson? list
    all (inputs.map run)
  match result with
  | Except.error e => do
      IO.eprintln e
      return 1
  | Except.ok runs => do
      let outputs <- all runs
      let json := toJson outputs
      IO.println json
      return 0
