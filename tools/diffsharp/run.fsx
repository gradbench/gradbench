#r "nuget: FSharp.Data"
#r "nuget: DiffSharp-lite"
#load "functions.fsx"

open DiffSharp
open FSharp.Data
open FSharp.Data.JsonExtensions
open System
open System.Diagnostics
open functions

let run (pars: JsonValue) =
    let arg = dsharp.tensor (pars?input.AsFloat())
    let name = pars?name.AsString()
    let func = if name = "square" then square else double
    let stopwatch = Stopwatch.StartNew()
    let result = func arg
    stopwatch.Stop()
    (float result, decimal stopwatch.ElapsedTicks)

let createJsonData message =
    let id = message?id

    if message?kind.AsString() = "evaluate" then
        let (result, time) = run message

        JsonValue.Record
            [| ("id", id)
               ("output", JsonValue.Float result)
               ("nanoseconds", JsonValue.Record [| ("evaluate", JsonValue.Number time) |]) |]
    else
        JsonValue.Record [| ("id", id) |]


assert (Stopwatch.Frequency = 1000000000L) //Ensure one tick is one nanosecond

let mutable line = Console.ReadLine()

while not (isNull line) do
    let message = JsonValue.Parse(line)
    let json = createJsonData message
    let jsonString = json.ToString(JsonSaveOptions.CompactSpaceAfterComma)
    printfn "%s" jsonString
    line <- Console.ReadLine()
