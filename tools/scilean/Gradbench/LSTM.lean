import Gradbench.Util
import Gradbench.LSTM.Basic

namespace Gradbench

open Lean ToJson FromJson

open SciLean

def lstm : String → Option (Json → Except String (IO Output))
  | "objective" => some (wrap LSTM.objective)
  | "jacobian" => some (wrap LSTM.jacobian)
  | _ => none
