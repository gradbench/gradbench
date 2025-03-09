import Gradbench.KMeans.Basic

namespace Gradbench

open KMeans Lean ToJson FromJson

def kmeans : String → Option (Json → Except String (IO Output))
  | "cost" => some (wrap objective)
  | "dir" => some (wrap dir)
  | _ => none
