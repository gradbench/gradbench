module functions

#r "nuget: DiffSharp-cpu"

open DiffSharp

let square x = x * x

let double x = ( dsharp.grad square ) x
