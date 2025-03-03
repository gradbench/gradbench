import Gradbench.Util
import Gradbench.GMM.ObjectiveDirect

namespace Gradbench

open Lean ToJson FromJson

open SciLean GMM

def gmm : String → Option (Json → Except String (IO Output))
  | "objective" => some (wrap objective)
  | "jacobian" => some (wrap jacobian)
  | _ => none
