module modules

#load "gradbench.fsx"

let resolve (moduleName: string) =
    match moduleName with
    | "gradbench" ->
        Some(fun x ->
            match x with
            | "double" -> Some(gradbench.double)
            | "square" -> Some(gradbench.square)
            | _ -> None)
    | _ -> None
