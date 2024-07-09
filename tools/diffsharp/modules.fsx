module modules

#load "gradbench.fsx"

let resolve (moduleName: string) (nameOpt: string option) =
    match nameOpt with
    | None ->
        match moduleName with
        | "gradbench" -> Some (fun (x: DiffSharp.Tensor) -> x) // needs to return func of same type
        | _ -> None
    | Some name ->
        match moduleName, name with
        | "gradbench", "square" -> Some gradbench.square
        | "gradbench", "double" -> Some gradbench.double
        | _ -> None
