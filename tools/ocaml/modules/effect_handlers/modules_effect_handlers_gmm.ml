open Gradbench_shared
open Owl.Dense.Ndarray.Generic
open Modules_effect_handlers_evaluate_tensor
open Modules_effect_handlers_reverse_tensor

module FloatScalar : Shared_gmm_types.GMM_SCALAR
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

module EvaluateScalar : Shared_gmm_types.GMM_SCALAR
  with type t = Evaluate.scalar
= struct
  include Evaluate
  type t = Evaluate.scalar

  let float = c
end

module ReverseEvaluate = Reverse (Evaluate)

module ReverseScalar : Shared_gmm_types.GMM_SCALAR
  with type t = ReverseEvaluate.scalar
= struct
  include ReverseEvaluate
  type t = ReverseEvaluate.scalar

  let float = c
end

module OwlFloatTensor : Shared_gmm_types.GMM_TENSOR
  with type t = (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
  with type scalar = float
= struct
  type t = (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
  type scalar = float

  open Owl.Dense.Ndarray.Generic

  let einsum_ijk_mik_to_mij a x =
    let ( - ) = Stdlib.( - ) in
    let (n, k) = ((shape x).(0), (shape x).(1)) in
    let y = empty Bigarray.Float64 (shape x) in
    for i = 0 to n - 1 do
      for j = 0 to k - 1 do
        (* Slice left are views, i.e. memory is shared. *)
        let sa = slice_left a [|j|] in
        let sx = slice_left x [|i; j|] in
        let sy = slice_left y [|i; j|] in
        Owl.Cblas.gemv ~trans:false ~incx:1 ~incy:1 ~alpha:1.0 ~beta:0.0
                       ~a:sa ~x:sx ~y:sy;
      done;
    done;
    y

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
  let einsum_ijk_mik_to_mij = einsum_ijk_mik_to_mij
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

module EvaluateTensor : Shared_gmm_types.GMM_TENSOR
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

module ReverseTensor : Shared_gmm_types.GMM_TENSOR
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

module GMMTest () : Shared_test_interface.TEST
  with type input =
    (float, (float, Bigarray.float64_elt) t)
      Shared_gmm_data.gmm_input
  with type output =
   (float, (float, Bigarray.float64_elt) t)
     Shared_gmm_data.gmm_output
= struct
  type input =
    (float, (float, Bigarray.float64_elt) t)
      Shared_gmm_data.gmm_input
  type output =
    (float, (float, Bigarray.float64_elt) t)
      Shared_gmm_data.gmm_output

  let input = ref None
  let objective = ref 0.0
  let gradient = ref (zeros Bigarray.Float64 [|0|])
  let _grads = ref (Array.init 3 (fun _ -> zeros Bigarray.Float64 [|0|]))

  let prepare input' =
    input := Some input'
  let calculate_objective times =
    let module Objective =
      Shared_gmm_objective.Make (EvaluateScalar) (EvaluateTensor)
    in
    match !input with
      | None -> ()
      | Some param ->
        for _ = 1 to times do
          objective :=
            Effect.Deep.match_with
              Objective.gmm_objective
              param
              Evaluate.evaluate
        done
  let calculate_jacobian times =
    let module Objective =
      Shared_gmm_objective.Make (ReverseScalar) (ReverseTensor)
    in
    match !input with
      | None -> ()
      | Some param ->
        for _ = 1 to times do
          let grads = Effect.Deep.match_with (fun p ->
            ReverseEvaluate.grad (fun ta ->
              let (alphas, means, icfs) = (ta.(0), ta.(1), ta.(2)) in
              Objective.gmm_objective
                { alphas = alphas;
                  means = means;
                  icfs = icfs;
                  x = ReverseTensor.tensor param.x;
                  wishart = {
                    gamma = ReverseScalar.float param.wishart.gamma;
                    m = param.wishart.m
                  }
                }
            ) p
          ) [|param.alphas; param.means; param.icfs|] Evaluate.evaluate in
          _grads := grads
        done
  let output _ =
    let flattened = Array.map flatten !_grads in
    gradient := concatenate flattened;
    {
      Shared_gmm_data.objective = !objective;
      Shared_gmm_data.gradient = !gradient
    }
end
