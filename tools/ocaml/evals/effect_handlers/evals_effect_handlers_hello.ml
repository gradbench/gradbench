open Gradbench_shared
open Evals_effect_handlers_evaluate_tensor
open Evals_effect_handlers_reverse_tensor

module FloatScalar : Shared_hello.HELLO_SCALAR
       with type t = float
  = struct
  type t = float
  let ( *. ) = Stdlib.( *. )
end

module EvaluateScalar : Shared_hello.HELLO_SCALAR
       with type t = Evaluate.scalar
  = struct
  include Evaluate
  type t = Evaluate.scalar
end


module ReverseEvaluate = Reverse (Evaluate)

module ReverseScalar : Shared_hello.HELLO_SCALAR
       with type t = ReverseEvaluate.scalar
  = struct
  include ReverseEvaluate
  type t = ReverseEvaluate.scalar
end

module type HELLO = sig
  val square : float -> float
  val double : float -> float
end

module Hello : HELLO = struct
  let square x =
    let module Objective = Shared_hello.Make (EvaluateScalar)
    in Objective.square x

  let double x =
    let module Objective =
      Shared_hello.Make (ReverseScalar) in
    let square' (x' : Evaluate.tensor prop array) =
      Objective.square (ReverseEvaluate.get (x'.(0)) [|0|])
    in
    let grads = Effect.Deep.match_with (fun p ->
                    ReverseEvaluate.grad square' p
                  ) [|Evaluate.create [|1|] x|] Evaluate.evaluate
    in Evaluate.get (grads.(0)) [|0|]
end
