import Gradbench.Util
import Gradbench.GMM.Basic

namespace Gradbench

open Lean ToJson FromJson

open SciLean

def gmm : String → Option (Json → Except String (IO Output))
  | "objective" => some (wrap GMM.objective)
  | "jacobian" => some (wrap GMM.jacobian)
  | _ => none
