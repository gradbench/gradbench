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
  type input
  type output

  val square : input -> output
  val double : input -> output

  val input_of_json : Yojson.Basic.t -> input
  val json_of_output : output -> Yojson.Basic.t
end

module Hello : HELLO = struct
  type input = float
  type output = float

  let square x =
    let module Objective = Shared_hello.Make (EvaluateScalar)
    in Effect.Deep.match_with
         Objective.square
         x
         Evaluate.evaluate

  let double x =
    let module Objective = Shared_hello.Make (ReverseScalar) in
    Effect.Deep.match_with
      (ReverseEvaluate.grads Objective.square)
      x Evaluate.evaluate

  let input_of_json = function
    | `Float x -> x
    | _ -> failwith "input_from_json: unexpected JSON"

  let json_of_output x = `Float x

end
