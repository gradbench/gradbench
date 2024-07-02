module modules

#load "gradbench.fsx"

let runModule (moduleName: string) (name: string) (arg: DiffSharp.Tensor) =
    match moduleName with
    | "gradbench" ->
        let func = if name = "square" then gradbench.square else gradbench.double
        func arg
    // add other modules
    | _ -> failwith "Unsupported module"

let moduleExists moduleName =
    let modulePath = sprintf "%s.fsx" moduleName
    System.IO.File.Exists(modulePath)
