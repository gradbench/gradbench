#r "nuget: FSharp.Data"
#r "nuget: DiffSharp-lite"
#load "modules.fsx"

open DiffSharp
open FSharp.Data
open FSharp.Data.JsonExtensions
open System
open System.Diagnostics
open modules

let run (pars: JsonValue) =
    let arg = dsharp.tensor (pars?input.AsFloat())
    let name = pars?name.AsString()
    let moduleName = pars.GetProperty("module").AsString()
    let stopwatch = Stopwatch.StartNew()
    let result = runModule moduleName name arg
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
    elif message?kind.AsString() = "define" then
        let moduleName = message.GetProperty("module").AsString()
        let success = moduleExists moduleName
        JsonValue.Record [| ("id", id)
                            ("succes", JsonValue.Boolean success) |]
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
