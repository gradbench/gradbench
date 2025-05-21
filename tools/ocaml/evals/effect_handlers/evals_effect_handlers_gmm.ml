(* https://github.com/jasigal/ADBench/blob/b98752f96a3b785e07ff6991853dc1073e6bf075/src/ocaml/modules/effect_handlers/modules_effect_handlers_gmm.ml *)

open Gradbench_shared
open Owl.Dense.Ndarray.Generic
open Evals_effect_handlers_evaluate_tensor
open Evals_effect_handlers_reverse_tensor

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

module type GMM = sig
  type input
  type objective_output
  type jacobian_output

  val input_of_json : Yojson.Basic.t -> input
  val json_of_objective : objective_output -> Yojson.Basic.t
  val json_of_jacobian : jacobian_output -> Yojson.Basic.t

  val objective: input -> objective_output
  val jacobian: input -> jacobian_output
end

module GMM : GMM
  = struct
  type input = (float, (float, Bigarray.float64_elt) t) Shared_gmm_data.gmm_input
  type objective_output = float
  type jacobian_output = (float, Bigarray.float64_elt) t

  let objective (param: input) =
    let module Objective =
      Shared_gmm_objective.Make (EvaluateScalar) (EvaluateTensor)
    in
    Effect.Deep.match_with
      Objective.gmm_objective
      param
      Evaluate.evaluate

  let jacobian (param: input) =
    let module Objective =
      Shared_gmm_objective.Make (ReverseScalar) (ReverseTensor)
    in
    let grads =
      Effect.Deep.match_with (fun p ->
          ReverseEvaluate.grad (fun ta ->
              let (alphas, mu, q, l) = (ta.(0), ta.(1), ta.(2), ta.(3)) in
              Objective.gmm_objective
                { alphas = alphas;
                  mu = mu;
                  q = q;
                  l = l;
                  x = ReverseTensor.tensor param.x;
                  wishart = {
                      gamma = ReverseScalar.float param.wishart.gamma;
                      m = param.wishart.m
                    }
                }
            ) p
        ) [|param.alphas; param.mu; param.q; param.l|] Evaluate.evaluate
    in concatenate (Array.map flatten grads)

  let input_of_json json =
    let module U = Yojson.Basic.Util in
    let module A = Owl.Dense.Ndarray.Generic in
    let floats x = x
                   |> U.to_list
                   |> Array.of_list
                   |> Array.map U.to_float in
    let floats_2d x = x
                      |> U.to_list
                      |> U.flatten
                      |> Array.of_list
                      |> Array.map U.to_float in
    let d = json |> U.member "d" |> U.to_int in
    let k = json |> U.member "k" |> U.to_int in
    let n = json |> U.member "n" |> U.to_int in
    let alphas_a = json |> U.member "alpha" |> floats in
    let mu_a = json |> U.member "mu" |> floats_2d in
    let q_a = json |> U.member "q" |> floats_2d in
    let l_a = json |> U.member "l" |> floats_2d in
    let x_a = json |> U.member "x" |> floats_2d in
    let gamma = json |> U.member "gamma" |> U.to_float in
    let m = json |> U.member "m" |> U.to_int in

    let lsz = Stdlib.(d*(d-1)/2) in
    let alphas = A.init Bigarray.Float64 [|k|] (Array.get alphas_a) in
    let mu = A.init Bigarray.Float64 [|k;d|] (Array.get mu_a) in
    let q = A.init Bigarray.Float64 [|k;d|] (Array.get q_a) in
    let l = A.init Bigarray.Float64 [|k;lsz|] (Array.get l_a) in
    let x = A.init Bigarray.Float64 [|n;d|] (Array.get x_a) in

    let open Shared_gmm_data in

    {alphas = alphas;
     mu = mu;
     q = q;
     l = l;
     x = x;
     wishart = {gamma; m;}
    }

  let json_of_objective x = `Float x
  let json_of_jacobian (x: jacobian_output) : Yojson.Basic.t =
    let n = (Bigarray.Genarray.dims x).(0) in
    `List (List.init n (fun i -> `Float (Bigarray.Genarray.get x [|i|])))
end
