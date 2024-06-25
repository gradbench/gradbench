#r "nuget: FSharp.Data"
#r "nuget: DiffSharp-lite"
#load "functions.fsx"

open DiffSharp
open FSharp.Data
open FSharp.Data.JsonExtensions
open System
open System.Text.Json
open System.Diagnostics
open functions

let run (pars : JsonValue) =
    let inputs = pars?arguments
    let values = [for item in inputs -> dsharp.tensor (item?value.AsFloat())]
    let name = pars?name.AsString()
    
    if name = "square" then 
        let stopwatch = Stopwatch.StartNew()
        let result = square (values.Head)
        stopwatch.Stop()
        (float result, float stopwatch.ElapsedTicks)
    else 
        let stopwatch = Stopwatch.StartNew()
        let result = double (values.Head)
        stopwatch.Stop()
        (float result, float stopwatch.ElapsedTicks)

let createJsonData cfg =
    let data = 
        (cfg?inputs.AsArray() |> Array.map (fun entry ->
            let (result, time) = run entry
            FSharp.Data.JsonValue.Record [|
                ("return",  JsonValue.Float result); 
                ("nanoseconds",  JsonValue.Float time)
            |]
        ))
    let json = JsonValue.Record [|
                    ("outputs",  JsonValue.Array data)
                |]
    json


assert (Stopwatch.Frequency = 1000000000L) //Ensure frequency gives ticks per second

let cfg = JsonValue.Load(Console.In)
let json = createJsonData cfg

let jsonString = json.ToString(JsonSaveOptions.CompactSpaceAfterComma)
printfn "%s" jsonString