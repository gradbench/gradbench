import Gradbench.Util
import Gradbench.GMM.ObjectiveDirect

namespace Gradbench

open Lean ToJson FromJson

set_default_scalar Float

open SciLean GMM

def gmm : String → Option (Json → Except String (IO Output))
  | "objective" => some (wrap objective)
  | "jacobian" => some (fun _ => do .ok <| do
    return default)
  | _ => none
