import Gradbench.KMeans.Basic

namespace Gradbench

open Lean ToJson FromJson

def kmeans : String → Option (Json → Except String (IO Output))
  | "cost" => some (wrap KMeans.cost)
  | "dir" => some (wrap KMeans.dir)
  | _ => none
