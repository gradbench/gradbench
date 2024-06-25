#r "nuget: FSharp.Data"
#r "nuget: DiffSharp-cpu"
#load "functions.fsx"

open DiffSharp
open FSharp.Data
open FSharp.Data.JsonExtensions
open System
open System.Diagnostics
open functions

let run (pars : JsonValue) =
    let inputs = pars?arguments
    let values = [for item in inputs -> dsharp.tensor (item?value.AsFloat())]
    let name = pars?name.AsString()
    if name = "square" then square values.Head
    else double values.Head

let createJsonData cfg =
    let data = 
        (cfg?inputs.AsArray() |> Array.map (fun entry ->
            let stopwatch = Stopwatch.StartNew()
            let result = run entry
            stopwatch.Stop()
            FSharp.Data.JsonValue.Record [|
                ("return",  JsonValue.Float (float result));
                ("nanoseconds",  JsonValue.Float (float stopwatch.Elapsed.TotalMilliseconds * 1000000.0))
            |]
        ))
    let json = JsonValue.Record [|
                    ("outputs",  JsonValue.Array data)
                |]
    json

let cfg = JsonValue.Load(Console.In)
let json = createJsonData cfg
printfn "%A" json