! Enzyme reverse mode over the shared Fortran lse primal. The primal is
! included so that it is in the same translation unit as the
! enzyme_autodiff call, where the flang Enzyme plugin can see its body.
include '../../cpp/gradbench/evals/fortran/lse.f90'

subroutine lse_enzyme_gradient(n, x, xb) bind(c, name="lse_enzyme_gradient")
  use iso_c_binding, only: c_int, c_double
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup, enzyme_dupnoneed
  implicit none
  integer(c_int), intent(in) :: n
  real(c_double), intent(in) :: x(n)
  real(c_double), intent(inout) :: xb(n)
  real(c_double) :: out, outb
  external :: lse_primal_f
  outb = 1.0d0
  call enzyme_autodiff(lse_primal_f, enzyme_const, n, enzyme_dup, x, xb, &
                       enzyme_dupnoneed, out, outb)
end subroutine lse_enzyme_gradient
