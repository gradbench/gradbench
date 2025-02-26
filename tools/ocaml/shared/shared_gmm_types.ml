module type GMM_SCALAR = sig
  type t

  val float : float -> t

  val log : t -> t
  val ( +. ) : t -> t -> t
  val ( -. ) : t -> t -> t
  val ( *. ) : t -> t -> t
  val ( /. ) : t -> t -> t
end

module type GMM_TENSOR = sig
  type t
  type scalar

  val tensor : (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t -> t

  (* Shape of the tensor *)
  val shape : t -> int array

  (* Creating constant tensors *)
  val zeros : int array -> t
  val create : int array -> float -> t

  (* Combining tensors *)
  val concatenate : ?axis:int -> t array -> t
  val stack : ?axis:int -> t array -> t

  (* Shrinking and slicing tensors *)
  val squeeze : ?axis:int array -> t -> t
  val get_slice : int list list -> t -> t
  val slice_left : t -> int array -> t
  val get : t -> int array -> scalar

  (* Einsum operation *)
  val einsum_ijk_mik_to_mij : t -> t -> t

  (* Pointwise tensor operations *)
  val exp : t -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t

  (* Reduction operations *)
  val sum_reduce : ?axis:int array -> t -> t
  val log_sum_exp : ?axis:int -> ?keep_dims:bool -> t -> t

  (* Scalar-tensor operations *)
  val scalar_mul : scalar -> t -> t
  val sub_scalar : t -> scalar -> t

  (* Tensor-float operations *)
  val pow_const : t -> float -> t
end