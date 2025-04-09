open Gradbench_shared
open Owl.Dense.Ndarray.Generic
open Evals_effect_handlers_evaluate_tensor
open Evals_effect_handlers_reverse_tensor

module FloatScalar : Shared_logsumexp.LOGSUMEXP_SCALAR
       with type t = float
  = struct
  type t = float

  let float x = x
  let log = Stdlib.log
  let ( +. ) = Stdlib.( +. )
  let ( -. ) = Stdlib.( -. )
  let ( *. ) = Stdlib.( *. )
  let ( /. ) = Stdlib.( /. )
end

module EvaluateScalar : Shared_logsumexp.LOGSUMEXP_SCALAR
       with type t = Evaluate.scalar
  = struct
  include Evaluate
  type t = Evaluate.scalar

  let float = c
end

module ReverseEvaluate = Reverse (Evaluate)

module ReverseScalar : Shared_logsumexp.LOGSUMEXP_SCALAR
       with type t = ReverseEvaluate.scalar
  = struct
  include ReverseEvaluate
  type t = ReverseEvaluate.scalar

  let float = c
end

module OwlFloatTensor : Shared_logsumexp.LOGSUMEXP_TENSOR
       with type t = (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
       with type scalar = float
  = struct
  type t = (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
  type scalar = float

  open Owl.Dense.Ndarray.Generic

  let tensor x = x
  let shape = shape
  let zeros = zeros Bigarray.Float64
  let create = create Bigarray.Float64
  let concatenate = concatenate
  let stack = stack
  let squeeze = squeeze
  let get_slice = get_slice
  let slice_left = slice_left
  let get = get
  let exp = exp
  let add = add
  let sub = sub
  let mul = mul
  let sum_reduce = sum_reduce
  let log_sum_exp = log_sum_exp
  let scalar_mul = scalar_mul
  let sub_scalar = sub_scalar
  let pow_const = pow_scalar
end

module EvaluateTensor : Shared_logsumexp.LOGSUMEXP_TENSOR
       with type t = Evaluate.tensor
       with type scalar = Evaluate.scalar
  = struct
  include Evaluate
  type t = Evaluate.tensor
  type scalar = Evaluate.scalar

  let tensor x = x
  let add = ( + )
  let sub = ( - )
  let mul = ( * )
end

module ReverseTensor : Shared_logsumexp.LOGSUMEXP_TENSOR
       with type t = ReverseEvaluate.tensor
       with type scalar = ReverseEvaluate.scalar
  = struct
  include ReverseEvaluate
  type t = ReverseEvaluate.tensor
  type scalar = ReverseEvaluate.scalar

  let tensor x = {
      v = x;
      dv = Evaluate.zeros (Owl.Dense.Ndarray.Generic.shape x)
    }
  let create ia f = create ia (Evaluate.c f)
  let add = ( + )
  let sub = ( - )
  let mul = ( * )
end

module type LOGSUMEXP = sig
  type input
  type primal_output
  type gradient_output

  val input_of_json : Yojson.Basic.t -> input
  val json_of_primal : primal_output -> Yojson.Basic.t
  val json_of_gradient : gradient_output -> Yojson.Basic.t

  val primal: input -> primal_output
  val gradient: input -> gradient_output
end

module LOGSUMEXP : LOGSUMEXP
  = struct
  type input = (float, Bigarray.float64_elt) t
  type primal_output = float
  type gradient_output = (float, Bigarray.float64_elt) t

  let primal (param: input) =
    let module Objective =
      Shared_logsumexp.Make (EvaluateScalar) (EvaluateTensor)
    in
    Effect.Deep.match_with
      Objective.primal
      param
      Evaluate.evaluate

  let gradient (param: input) =
    let module Objective =
      Shared_logsumexp.Make (ReverseScalar) (ReverseTensor)
    in
    let grads =
      Effect.Deep.match_with (fun p ->
          ReverseEvaluate.grad (fun ta ->
              Objective.primal (ta.(0))
            ) p
        ) [|param|] Evaluate.evaluate
    in concatenate (Array.map flatten grads)

  let input_of_json json =
    let module U = Yojson.Basic.Util in
    let module A = Owl.Dense.Ndarray.Generic in
    let floats x = x
                   |> U.to_list
                   |> Array.of_list
                   |> Array.map U.to_float in
    let x_a = json |> U.member "x" |> floats in
    let n = Array.length x_a in
    let x = A.init Bigarray.Float64 [|n|] (Array.get x_a) in
    x

  let json_of_primal x = `Float x
  let json_of_gradient (x: gradient_output) : Yojson.Basic.t =
    let n = (Bigarray.Genarray.dims x).(0) in
    `List (List.init n (fun i -> `Float (Bigarray.Genarray.get x [|i|])))
end
