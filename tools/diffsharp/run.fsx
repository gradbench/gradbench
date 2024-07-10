#r "nuget: FSharp.Data"
#r "nuget: Newtonsoft.Json"
#r "nuget: DiffSharp-lite"

#load "modules.fsx"

open DiffSharp
open FSharp.Data
open FSharp.Data.JsonExtensions
open Newtonsoft.Json
open Newtonsoft.Json.Linq
open System
open System.Diagnostics
open modules

type TensorConverter() =
    inherit JsonConverter()
    override this.CanConvert(objectType: Type) =
        objectType = typeof<Tensor>

    // serializing, convert to JSON
    override this.WriteJson(writer: JsonWriter, value: obj, serializer: JsonSerializer) =
        match value with
        | :? Tensor as tensor ->
            let array = tensor.toArray()
            serializer.Serialize(writer, array)
        | _ -> serializer.Serialize(writer, value)

    // deserializing, convert to Tensor
    override this.ReadJson(reader: JsonReader, objectType: Type, existingValue: obj, serializer: JsonSerializer) =
        let token = JToken.Load(reader)
        match token.Type with
        | JTokenType.Array ->
            let array = token.ToObject<float[]>()
            dsharp.tensor array :> obj
        | JTokenType.Float | JTokenType.Integer ->
            let value = token.ToObject<float>()
            dsharp.tensor value :> obj
        | _ -> failwith "can no convert JSON to Tensor"

let settings: JsonSerializerSettings = JsonSerializerSettings()
settings.Converters.Add(TensorConverter())

let wrap (f: 'a -> 'b) =
    fun (x:JsonValue)->
        try
            let x_: 'a = JsonConvert.DeserializeObject<'a>(x.ToString(), settings)
            let stopwatch = Stopwatch.StartNew()
            let y_ = f x_
            stopwatch.Stop()
            let yJson = JsonConvert.SerializeObject(y_, settings)
            let y = JsonValue.Parse(yJson)
            (y , Decimal stopwatch.ElapsedTicks)
        with
        | ex -> failwith ex.Message

let run (pars: JsonValue) =
    let arg = pars?input
    let name = pars?name.AsString()
    let moduleName = pars.GetProperty("module").AsString()

    let resolved =
        match resolve moduleName with
        | Some module_ -> module_
        | _ -> failwith "module not found"

    let func =
        match resolved name with
        | Some func_ -> (wrap func_)
        | _ -> failwith "function not found"

    func arg //add wrap here

let createJsonData message =
    let id = message?id
    let kind = message.GetProperty("kind").AsString()

    let response =
        match kind with
        | "evaluate" ->
            let (result, time) = run message
            [| ("id", id)
               ("output", result)
               ("nanoseconds", JsonValue.Record [| ("evaluate", JsonValue.Number time) |]) |]
        | "define" ->
            let moduleName = message.GetProperty("module").AsString()
            let success = Option.isSome (resolve moduleName)
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
