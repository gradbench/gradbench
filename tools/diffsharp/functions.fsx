module functions

#r "nuget: DiffSharp-lite"

open DiffSharp

let square x = x * x

let double x = ( dsharp.grad square ) x
