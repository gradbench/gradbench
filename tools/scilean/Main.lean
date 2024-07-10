import Lean
import SciLean
import Mathlib.Lean.CoreM

open Lean
open Except FromJson ToJson

-- This informs SciLean that we want to work primarly with `Float`
-- take it as a magic incantation that makes some of the following notation work
set_default_scalar Float

open SciLean

def square (x : Float) : Float := x * x

def double (x : Float) : Float :=
  -- take derivative of square at `x`
  -- it is just a syntactic sugar for `cderiv Float square x 1`
  (∂ square x)
    rewrite_by
      -- the derivative `cderiv` is noncomputable function thus `∂ square x` is just
      -- a symbolic expression that we need to rewrite using tactics into a computable form
      unfold square -- right now autodiff can't see through new definitions so we have to unfold them manually
      autodiff

--- Few examples of differentiation

-- specification that we want to differentiate `x*x` w.r.t. `x`
#check ∂ (x : Float), x * x

-- because it is just a spec, we can evaluate it, the following fails
-- #eval ∂ (x:=3.14), x * x

-- to get the actual derivative we have to rewrite the term with autodiff
#check (∂ (x : Float), x * x) rewrite_by autodiff

-- now we can evaluate
#eval (∂ (x:=3.14), x * x) rewrite_by autodiff

-- short hand using `!` that automatically calls autodiff
#eval (∂! (x:=3.14), x * x)

-- similarly we can compute gradient
#eval (∇! (x:=(3.14,0.3)), ‖x‖₂²)

-- similarly we can compute forward mode AD
#check (∂>! (x : Float×Float), ‖x‖₂²*x.1)

-- similarly we can compute reverse mode ad
#check (<∂! (x : Float×Float), ‖x‖₂²*x.1)

-- If you want to know what is the notation actually doing just set an option
set_option pp.notation false
#check (<∂ (x : Float×Float), ‖x‖₂²)

-- Function measuring time to compute the derivative
open Lean Meta Qq in
def benchAD : MetaM Unit := do

  let e := q(∂ square)

  let start <- IO.monoNanosNow
  let (e',_) ← rewriteByConv e (← `(conv| (unfold square; autodiff)))
  let done <- IO.monoNanosNow

  IO.println s!"differentiating took {1e-6*(done-start).toFloat}ms\n\n{← ppExpr e}\n==>\n{← ppExpr e'}"

def ranBenchAD : IO Unit :=
  let run : CoreM Unit := do
    let _ ← benchAD.run {} {}

  -- Here you have to specify all the lean files that need to be loaded
  CoreM.withImportModules #[`SciLean,`Main] run
    (searchPath:= compile_time_search_path%)

structure Definition where
  id: Int
  kind: String
  module: String
  source: String
deriving FromJson

structure Params where
  id: Int
  kind: String
  module: String
  name: String
  input: Json
deriving FromJson

structure Nanoseconds where
  evaluate: Nat
deriving ToJson

structure Output where
  output: Json
  nanoseconds: Nanoseconds

structure Response where
  id: Int
  output: Json
  nanoseconds: Nanoseconds
deriving ToJson

def wrap [FromJson a] [ToJson b] (f : a -> b) (x' : Json)
    : Except String (IO Output) := do
  let x : a <- fromJson? x'
  return do
    let start <- IO.monoNanosNow
    let y := f x
    let done <- IO.monoNanosNow
    let y' := toJson y
    let nanoseconds := { evaluate := done - start }
    return { output := y', nanoseconds }

def resolve (module : String)
    : Option (String -> Option (Json -> Except String (IO Output))) :=
  match module with
  | "gradbench" => some fun
    | "square" => some (wrap square)
    | "double" => some (wrap double)
    | _ => none
  | _ => none

partial def loop (stdin : IO.FS.Stream) (stdout : IO.FS.Stream) :
    IO (Except String Unit) := do
  let line <- stdin.getLine
  if line.isEmpty then
    return ok ()
  else
    let result := do
      let message <- Json.parse line
      let kind <- Json.getObjVal? message "kind"
      if kind == "define" then
        let definition : Definition <- fromJson? message
        let success := (resolve definition.module).isSome
        return do
          return Json.mkObj [("id", definition.id), ("success", toJson success)]
      else if kind == "evaluate" then
        let params : Params <- fromJson? message
        let resolved <- match resolve params.module with
          | some mod => ok mod
          | none => error "module not found"
        let function <- match resolved params.name with
          | some func => ok func
          | none => error "function not found"
        let action <- function params.input
        return do
          let { output, nanoseconds } <- action
          let response : Response := { id := params.id, output, nanoseconds }
          return toJson response
      else
        let id <- Json.getObjVal? message "id"
        return do
          return Json.mkObj [("id", id)]
    match result with
    | error err => return error err
    | ok action => do
      let response <- action
      IO.println response
      stdout.flush
      loop stdin stdout

def main : IO UInt32 := do
  let stdin <- IO.getStdin
  let stdout <- IO.getStdout
  let result <- loop stdin stdout
  match result with
  | error err => do
    IO.eprintln err
    return 1
  | ok _ => return 0
