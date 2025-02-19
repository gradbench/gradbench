import Lean
import SciLean

import Gradbench.Util

namespace Gradbench

open Lean
open Except FromJson ToJson

set_default_scalar Float

open SciLean

def square (x : Float) : Float := x * x

def double (x : Float) : Float := (∂ square x) rewrite_by unfold square; autodiff

def hello : String → Option (Json → Except String (IO Output))
  | "square" => some (wrap square)
  | "double" => some (wrap double)
  | _ => none
