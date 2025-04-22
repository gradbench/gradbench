import Gradbench.Util
import Gradbench.LLSQ.Basic

namespace Gradbench

open Lean

open SciLean

def llsq : String → Option (Json → Except String (IO Output))
  | "primal" => some (wrap LLSQ.primal)
  | "gradient" => some (wrap LLSQ.gradient)
  | _ => none
