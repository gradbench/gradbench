(* https://github.com/jasigal/ADBench/blob/b98752f96a3b785e07ff6991853dc1073e6bf075/src/ocaml/shared/shared_gmm_objective.ml *)

open Shared_gmm_data
open Shared_gmm_types

module type GMM_OBJECTIVE = sig
  type tensor
  type scalar

  val gmm_objective : (scalar, tensor) gmm_input -> scalar
end

module Make
  (S : GMM_SCALAR)
  (T : GMM_TENSOR with type scalar = S.t) : GMM_OBJECTIVE
  with type tensor = T.t
  with type scalar = S.t
= struct
  type tensor = T.t
  type scalar = S.t
  open S
  open T

  let ( + ) = add
  let ( - ) = sub
  let ( * ) = mul

  let constructl d icfs =
    let lparamidx = ref d in

    let make_l_col i =
      let nelems = Stdlib.(d - i - 1) in
      (* Slicing in Owl requires inculsive indices, so will not create
      * an empty tensor. Thus we have two cases. 
      *)
      let max_lparamidx = (shape icfs).(0) in
      let col =
        if Stdlib.(!lparamidx >= max_lparamidx) then
          zeros [|Stdlib.(i + 1)|]
        else concatenate ~axis:0 [|
          zeros [|Stdlib.(i + 1)|];
          get_slice [[!lparamidx; Stdlib.(!lparamidx + nelems - 1)]] icfs;
      |] in
      lparamidx := Stdlib.(!lparamidx + nelems);
      col
    in

    let columns = Array.init d make_l_col in
    stack ~axis:1 columns

  let qtimesx qdiag l x =
    let y = einsum_ijk_mik_to_mij l x in
    (qdiag * x) + y

  let log_gamma_distrib a p =
    Stdlib.(
    let scalar = (0.25 *. (float_of_int (p * (p - 1))) *. log Float.pi) in
    let summed = Array.fold_left (+.) 0.0
      (Array.init p (fun i ->
        Owl.Maths.loggamma (a +. 0.5 *. (float_of_int (1 - (i + 1))))
      ))
    in
    scalar +. summed
    )

  let log_wishart_prior p wishart sum_qs qdiags icf =
    let n = float_of_int (Stdlib.(p + wishart.m + 1)) in
    let k = float_of_int ((shape icf).(0)) in
    
    let out = sum_reduce (
      (
        scalar_mul (S.float 0.5 *. wishart.gamma *. wishart.gamma)
          (squeeze (
            sum_reduce ~axis:[|1|] (pow_const qdiags 2.0) +
            sum_reduce ~axis:[|1|] (pow_const (get_slice [[]; [p;-1]] icf) 2.0)
          ))
      )
      - (scalar_mul (S.float (float_of_int wishart.m)) sum_qs)
    ) in

    let c =
      S.float n
      *. S.float (float_of_int p)
      *. (log (wishart.gamma /. S.float (Stdlib.sqrt 2.0)))
    in
    sub_scalar
      out
      (S.float k *. (c -. S.float (log_gamma_distrib Stdlib.(0.5 *. n) p)))

  let gmm_objective param =
    let xshape = shape param.x in
    let n = xshape.(0) in
    let d = xshape.(1) in
    let k = (shape param.means).(0) in

    let qdiags = exp (get_slice [[]; [0; Stdlib.(d - 1)]] param.icfs) in
    let sqdiags = stack (Array.make n qdiags) in
    let sum_qs = squeeze (
      sum_reduce ~axis:[|1|] (get_slice [[]; [0; Stdlib.(d - 1)]] param.icfs)
    ) in
    (* Prevent implicit broadcasting *)
    let ssum_qs = stack (Array.make n sum_qs) in

    let icf_sz = (shape param.icfs).(0) in
    let ls = stack (Array.init icf_sz (fun i ->
      constructl d (slice_left param.icfs [|i|]))
    ) in

    let xcentered = squeeze (stack (Array.init n (fun i ->
      let sx = slice_left param.x [|i|] in
      (* Prevent implicit broadcasting *)
      let ssx = stack (Array.make k sx) in
      ssx - param.means
    ))) in
    let lxcentered = qtimesx sqdiags ls xcentered in
    let sqsum_lxcentered = squeeze (
      sum_reduce ~axis:[|2|] (pow_const lxcentered 2.0)
    ) in
    (* Prevent implicit broadcasting *)
    let salphas = stack (Array.make n param.alphas) in
    let inner_term =
      salphas + ssum_qs - (scalar_mul (S.float 0.5) sqsum_lxcentered)
    in
    (* Uses the stable version as in the paper, i.e. max-shifted *)
    let lse = squeeze (log_sum_exp ~axis:1 inner_term) in
    let slse = sum_reduce lse in

    let const = create [||] Stdlib.(
      -. (float_of_int n) *. (float_of_int d) *. 0.5 *. log (2.0 *. Float.pi)
    ) in
    
    let wish = log_wishart_prior d param.wishart sum_qs qdiags param.icfs in
    get (
      const + slse
            - scalar_mul (S.float (float_of_int n)) (squeeze (log_sum_exp param.alphas))
            + wish
    ) [|0|]
end
