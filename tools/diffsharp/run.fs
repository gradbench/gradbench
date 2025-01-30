module run

open Newtonsoft.Json.Linq
open System
open System.Diagnostics
open modules

let run (pars: JToken) =
    let arg = pars.["input"]
    let name = pars.["function"].ToString()
    let moduleName = pars.["module"].ToString()

    let resolved =
        match resolve moduleName with
        | Some module_ -> module_
        | _ -> failwith "module not found"

    let func =
        match resolved name with
        | Some func_ -> func_
        | _ -> failwith "function not found"

    func arg

let createJsonData (message: JToken) =
    let id = message.["id"]
    let kind = message.["kind"].ToString()

    let response =
        match kind with
        | "evaluate" ->
            let (result, time) = run message

            let timings =
                JArray([ JObject(JProperty("name", "evaluate"), JProperty("nanoseconds", time)) ])

            JObject(JProperty("id", id), JProperty("output", result), JProperty("timings", timings))
        | "define" ->
            let moduleName = message.["module"].ToString()
            let success = Option.isSome (resolve moduleName)
            JObject(JProperty("id", id), JProperty("success", success))
        | _ -> JObject(JProperty("id", id))

    response

assert (Stopwatch.Frequency = 1000000000L) // Ensure one tick is one nanosecond

let mutable line = Console.ReadLine()

while not (isNull line) do
    let message = JToken.Parse(line)
    let json = createJsonData message
    let jsonString = json.ToString(Newtonsoft.Json.Formatting.None)
    printfn "%s" jsonString
    line <- Console.ReadLine()
