! Enzyme reverse mode over the shared Fortran ode primal (see grad_lse.f90).
! The gradient is that of the last component of the final state. yf must be
! enzyme_dup, not enzyme_dupnoneed: the primal reads yf between steps, and
! with dupnoneed Enzyme may drop those stores (wrong gradient).
include '../../cpp/gradbench/evals/fortran/ode.f90'

subroutine ode_enzyme_gradient(n, xi, xib, s) bind(c, name="ode_enzyme_gradient")
  use iso_c_binding, only: c_int, c_double
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup
  implicit none
  integer(c_int), intent(in) :: n, s
  real(c_double), intent(in) :: xi(n)
  real(c_double), intent(inout) :: xib(n)
  real(c_double), allocatable :: yf(:), yfb(:)
  external :: ode_primal_f
  allocate(yf(n), yfb(n))
  yfb = 0.0d0
  yfb(n) = 1.0d0
  call enzyme_autodiff(ode_primal_f, enzyme_const, n, enzyme_dup, xi, xib, &
                       enzyme_const, s, enzyme_dup, yf, yfb)
end subroutine ode_enzyme_gradient
