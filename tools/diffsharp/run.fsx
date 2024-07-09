#r "nuget: FSharp.Data"
#r "nuget: DiffSharp-lite"
#load "modules.fsx"

open DiffSharp
open FSharp.Data
open FSharp.Data.JsonExtensions
open System
open System.Diagnostics
open modules

let tensor (x) =
    dsharp.tensor x

let run (pars: JsonValue) =
    let arg = tensor (pars?input.AsFloat())
    let name = pars?name.AsString()
    let moduleName = pars.GetProperty("module").AsString()
    let func =  Option.get (resolve moduleName (Some name) )
    let stopwatch = Stopwatch.StartNew()
    let result = func arg
    stopwatch.Stop()
    (float result, decimal stopwatch.ElapsedTicks)

let createJsonData message =
    let id = message?id
    let kind = message.GetProperty("kind").AsString()

    let response =
        match kind with
        | "evaluate" ->
            let (result, time) = run message
            [| ("id", id)
               ("output", JsonValue.Float result)
               ("nanoseconds", JsonValue.Record [| ("evaluate", JsonValue.Number time) |]) |]
        | "define" ->
            let moduleName = message.GetProperty("module").AsString()
            let success = Option.isSome (resolve moduleName None)
            [| ("id", id)
               ("success", JsonValue.Boolean success) |]
        | _ ->
            [| ("id", id) |]

    JsonValue.Record response

assert (Stopwatch.Frequency = 1000000000L) //Ensure one tick is one nanosecond

let mutable line = Console.ReadLine()

while not (isNull line) do
    let message = JsonValue.Parse(line)
    let json = createJsonData message
    let jsonString = json.ToString(JsonSaveOptions.CompactSpaceAfterComma)
    printfn "%s" jsonString
    line <- Console.ReadLine()
