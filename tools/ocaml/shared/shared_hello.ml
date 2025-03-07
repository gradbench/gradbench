module type HELLO_SCALAR = sig
  type t
  val ( *. ) : t -> t -> t
end


module type HELLO_OBJECTIVE = sig
  type scalar

  val square : scalar -> scalar
end

module Make
  (S : HELLO_SCALAR) : HELLO_OBJECTIVE
  with type scalar = S.t
= struct
  type scalar = S.t

  let square x = S.(x *. x)
end
