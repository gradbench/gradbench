open Effect.Deep
open Modules_effect_handlers_smooth_tensor

module Evaluate_Non_Diff : SMOOTH_NON_DIFF
  with type scalar = float
  with type tensor = (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
= struct
  type scalar = float
  type tensor = (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
  
  let shape = Owl.Dense.Ndarray.Generic.shape
  let add_ x dx = Owl.Dense.Ndarray.Generic.add_ ~out:x x dx
end

module T = Owl.Dense.Ndarray.Generic

module Evaluate = struct
  include Smooth (Evaluate_Non_Diff)

  let _contract_einsum_ijk_mik_to_mij a x =
    let open T in
    let (n, k, d) = ((shape x).(0), (shape x).(1), (shape x).(2)) in
    get_slice [[]; [0;-1;Stdlib.(k+1)]] (
      reshape (
        contract2 [|(2,2)|] x a
      ) [|n;Stdlib.(k*k);d|]
    )

  let einsum_ijk_mik_to_mij a x =
    let open T in
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

  let _contract_einsum_ijk_mij_to_mik a y =
    let open T in
    let (n, k, d) = ((shape y).(0), (shape y).(1), (shape y).(2)) in
    get_slice [[]; [0;-1;Stdlib.(k+1)]] (
      reshape (
        transpose ~axis:[|0;1;3;2|] (
          contract2 [|(2,1)|] y a
        )
      ) [|n;Stdlib.(k*k);d|]
    )

  let einsum_ijk_mij_to_mik a y =
    let open T in
    let ( - ) = Stdlib.( - ) in
    let (n, k) = ((shape y).(0), (shape y).(1)) in
    let x = empty Bigarray.Float64 (shape y) in
    let at = transpose ~axis:[|0;2;1|] a in
    (* ikj,mij -> mik *)
    for i = 0 to n - 1 do
      for j = 0 to k - 1 do
        (* Slice left are views, i.e. memory is shared. *)
        let sat = slice_left at [|j|] in
        let sy  = slice_left y  [|i; j|] in
        let sx  = slice_left x  [|i; j|] in
        Owl.Cblas.gemv ~trans:false ~incx:1 ~incy:1 ~alpha:1.0 ~beta:0.0
                      ~a:sat ~x:sy ~y:sx;
      done;
    done;
    x

  let _contract_einsum_mij_mik_to_ijk y x =
    let open T in
    let (k, d) = ((shape x).(1), (shape x).(2)) in
    get_slice [[0;-1;Stdlib.(k+1)]] (
      reshape (
        transpose ~axis:[|1;3;0;2|] (
          contract2 [|(0,0)|] y x
        )
      ) [|Stdlib.(k*k);d;d|]
    )

  let einsum_mij_mik_to_ijk y x =
    let open T in
    let ( - ) = Stdlib.( - ) in
    let (k, d) = ((shape x).(1), (shape x).(2)) in
    let a = empty Bigarray.Float64 [|k;d;d|] in
    let yt = transpose ~axis:[|1;2;0|] y in
    let xt = transpose ~axis:[|1;0;2|] x in
    for i = 0 to k - 1 do
      let syt = slice_left yt [|i|] in
      let sxt = slice_left xt [|i|] in
      let sa  = slice_left a  [|i|] in
      Owl.Cblas.gemm ~transa:false ~transb:false ~alpha:1.0 ~beta:0.0
                      ~a:syt ~b:sxt ~c:sa
    done;
    a

  let evaluate = {
    retc = (fun x -> x);
    exnc = raise;
    effc = (fun (type a) (eff : a Effect.t) ->
      match eff with
      | Ap_u_to_s o -> Some (fun (k : (a, _) continuation) ->
          match o with
            | Const f -> continue k f
        )
      | Ap_s_to_s (o, s) -> Some (fun k ->
          match o with
            | Negate -> continue k Stdlib.(-. s)
            | Log -> continue k Stdlib.(log s)
        )
      | Ap_s's_to_s (o, s1, s2) -> Some (fun k ->
          match o with
            | Add -> continue k Stdlib.(s1 +. s2)
            | Subtract -> continue k Stdlib.(s1 -. s2)
            | Multiply -> continue k Stdlib.(s1 *. s2)
            | Divide -> continue k Stdlib.(s1 /. s2)
        )
      | Ap_u_to_t o -> Some (fun k ->
          match o with
            | Zeros ia -> continue k T.(zeros Bigarray.Float64 ia)
            | Create (ia, f) -> continue k T.(create Bigarray.Float64 ia f)
        )
      | Ap_t_to_t (o, t) -> Some (fun k ->
          match o with
            | Squeeze iao -> continue k T.(squeeze ?axis:iao t)
            | Reshape ia -> continue k T.(reshape t ia)
            | GetSlice ill -> continue k T.(get_slice ill t)
            | SliceLeft ia -> continue k T.(slice_left t ia)
            | Transpose iao -> continue k T.(transpose ?axis:iao t)
            | Exp -> continue k T.(exp t)
            | Negate -> continue k T.(neg t)
            | PowerConst f -> continue k T.(pow_scalar t f)
            | SumReduce iao -> continue k T.(sum_reduce ?axis:iao t)
            | LogSumExp (io, bo) ->
              continue k T.(log_sum_exp ?axis:io ?keep_dims:bo t)
            | Softmax io -> continue k T.(softmax ?axis:io t)
        )
      | Ap_t't_to_t (o, t1, t2) -> Some (fun k ->
          match o with
            | Add -> continue k T.(t1 + t2)
            | Subtract -> continue k T.(t1 - t2)
            | Multiply -> continue k T.(t1 * t2)
            | Divide -> continue k T.(t1 / t2)
            | Einsum_ijk_mik_to_mij -> continue k (einsum_ijk_mik_to_mij t1 t2)
            | Einsum_ijk_mij_to_mik -> continue k (einsum_ijk_mij_to_mik t1 t2)
            | Einsum_mij_mik_to_ijk -> continue k (einsum_mij_mik_to_ijk t1 t2)
            | SetSlice ill ->
              let tout = T.copy t1 in
              T.set_slice ill tout t2;
              continue k tout
        )
      | Ap_t_to_s (o, t) -> Some (fun k ->
          match o with
            | Get ia -> continue k T.(get t ia)
            | Sum -> continue k T.(sum' t)
        )
      | Ap_s't_to_t (o, s, t) -> Some (fun k ->
          match o with
            | ScalarMultiply -> continue k T.(scalar_mul s t)
            | SubtractScalar -> continue k T.(sub_scalar t s)
        )
      | Ap_ta_to_t (o, ta) -> Some (fun k ->
          match o with
            | Concatenate io -> continue k T.(concatenate ?axis:io ta)
            | Stack io -> continue k T.(stack ?axis:io ta)
        )
      | Ap_t_to_ta (o, t) -> Some (fun k ->
          match o with
            | Split (io, ia) -> continue k T.(split ?axis:io ia t)
        )
      | _ -> None
    )
  }
end
