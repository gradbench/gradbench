import Lean
import SciLean

import Gradbench

open Lean
open Except FromJson ToJson

open Gradbench


def resolve (module : String)
    : Option (String -> Option (Json -> Except String (IO Output))) :=
  match module with
  | "hello" => some hello
  | "gmm" => some gmm
  | "kmeans" => some kmeans
  -- | "lstm" => some lstm
  | "llsq" => some llsq
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
      if kind == "start" then
        let id <- Json.getObjVal? message "id"
        return do
          return Json.mkObj [("id", id), ("tool", "scilean")]
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
        let function <- match resolved params.function with
          | some func => ok func
          | none => error "function not found"
        let action <- function params.input
        return do
          let id := params.id
          let { output, timings } <- action
          let success := true
          let response : Response := { id, success, output, timings }
          return toJson response
      else
        let id <- Json.getObjVal? message "id"
        return do
          return Json.mkObj [("id", id)]
    match result with
    | error err => return error err
    | ok action => do
      let response <- action
      -- IO.eprintln (Json.compress response)
      IO.println (Json.compress response)
      stdout.flush
      loop stdin stdout


open SciLean.VectorType in
def main : IO UInt32 := do
  let stdin <- IO.getStdin
  let stdout <- IO.getStdout
  let result <- loop stdin stdout
  match result with
  | error err => do
    IO.eprintln err
    return 1
  | ok _ => return 0
