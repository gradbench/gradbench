import Gradbench.Util
import Gradbench.LSTM.Objective

namespace Gradbench

open Lean ToJson FromJson

open SciLean LSTM

def lstm : String → Option (Json → Except String (IO Output))
  | "objective" => some (wrap objective)
  | "jacobian" => some (wrap jacobian)
  | _ => none
