import Lean

structure Argument where
  value: Float
deriving Lean.FromJson, Repr

instance : ToString Argument where
  toString : Argument -> String
    | { value } => s!"\{ value := {value} }"

structure Input where
  name: String
  arguments: List Argument
deriving Lean.FromJson, Repr

instance : ToString Input where
  toString : Input -> String
    | { name, arguments } => s!"\{ name := {name}, arguments := {arguments} }"

partial def readToEnd (stream : IO.FS.Stream) : IO String := do
  let rec loop (s : String) := do
    let line ← stream.getLine
    if line.isEmpty then
      return s
    else
      loop (s ++ line)
  loop ""

def main : IO Unit := do
  let stdin <- IO.getStdin
  let s <- readToEnd stdin
  let result: Except String (List Input) := do
    let cfg <- Lean.Json.parse s
    let inputs <- Lean.Json.getObjVal? cfg "inputs"
    Lean.FromJson.fromJson? inputs
  IO.println s!"{result}"
