module modules

#load "gradbench.fsx"

open DiffSharp
open Newtonsoft.Json
open Newtonsoft.Json.Linq
open System.Diagnostics

// Converting functions for DiffSharp Tensors
let jsonToTensor (json: JToken) =
    match json.Type with
    | JTokenType.Array ->
        let array = json.ToObject<float[]>()
        dsharp.tensor array
    | JTokenType.Float | JTokenType.Integer ->
        let value = json.ToObject<float>()
        dsharp.tensor value
    | _ -> failwith "cannot convert JSON to Tensor"

let tensorToJson (tensor: DiffSharp.Tensor) =
    JToken.Parse(JsonConvert.SerializeObject(tensor))

let wrap (f: 'a -> 'b) (jsonToA: JToken -> 'a) (bToJson: 'b -> JToken) =
    fun (x:JToken)->
        try
            let input: 'a = jsonToA x
            let timer = Stopwatch.StartNew()
            let result = f input
            timer.Stop()
            let y = bToJson result
            (y , int32 timer.ElapsedTicks)
        with
        | ex -> failwith ex.Message

let resolve (moduleName: string) =
    match moduleName with
    | "gradbench" ->
        Some(fun x ->
            match x with
            | "double" -> Some(wrap gradbench.double jsonToTensor tensorToJson)
            | "square" -> Some(wrap gradbench.square jsonToTensor tensorToJson)
            | _ -> None)
    | _ -> None
