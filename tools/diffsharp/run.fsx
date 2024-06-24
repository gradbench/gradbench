#r "nuget: FSharp.Data"

open FSharp.Data
open System

#load "functions.fsx"

// Correctly fetch a field from a JsonValue
let run (pars : JsonValue) =
    pars.GetProperty("name")

// Load JSON from standard input
let cfg = JsonValue.Load(Console.In)

// Iterating over an array expected under the "inputs" field
for item in cfg.GetProperty("inputs").AsArray() do
    let result = run item
    printfn "%A" result
