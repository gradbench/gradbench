! Enzyme reverse mode over the shared Fortran llsq primal (see grad_lse.f90).
include '../../cpp/gradbench/evals/fortran/llsq.f90'

subroutine llsq_enzyme_gradient(n, m, x, xb) bind(c, name="llsq_enzyme_gradient")
  use iso_c_binding, only: c_int, c_double
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup, enzyme_dupnoneed
  implicit none
  integer(c_int), intent(in) :: n, m
  real(c_double), intent(in) :: x(m)
  real(c_double), intent(inout) :: xb(m)
  real(c_double) :: out, outb
  external :: llsq_primal_f
  outb = 1.0d0
  call enzyme_autodiff(llsq_primal_f, enzyme_const, n, enzyme_const, m, &
                       enzyme_dup, x, xb, enzyme_dupnoneed, out, outb)
end subroutine llsq_enzyme_gradient
