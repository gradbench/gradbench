module type TEST = sig
  type input
  type output

  val prepare : input -> unit
  val calculate_objective : int -> unit
  val calculate_jacobian : int -> unit
  val output : unit -> output
end

module type MAKE = functor () -> TEST
