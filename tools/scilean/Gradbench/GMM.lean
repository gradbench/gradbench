import Lean
import SciLean

import Gradbench.Util

namespace Gradbench

open Lean
open Except FromJson ToJson

set_default_scalar Float

open SciLean


structure GMMData (D K N : Nat) where
  x : Float^[D]^[N]
  alpha : Float^[K]
deriving ToJson

-- structure GMMData (D K N : Nat) where
--   m : Float
--   x : Float^[D]^[N]
--   α : Float^[K]
--   μ : Float^[D]^[K]
--   q : Float^[D]^[K]
--   l : Float^[((D-1)*D)/2]^[K]



def gmm : String → Option (Json → Except String (IO Output)) := fun f =>
  .some fun input => .ok <| do
    IO.eprintln s!"function: {f}"
    IO.eprintln s!"input:\n{input}"
    return default
