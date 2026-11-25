! Enzyme reverse mode over the shared Fortran det primal (see grad_lse.f90).
include '../../cpp/gradbench/evals/fortran/det.f90'

subroutine det_enzyme_gradient(ell, a, ab) bind(c, name="det_enzyme_gradient")
  use iso_c_binding, only: c_int, c_double
  use enzyme, only: enzyme_autodiff, enzyme_const, enzyme_dup, enzyme_dupnoneed
  implicit none
  integer(c_int), intent(in) :: ell
  real(c_double), intent(in) :: a(ell*ell)
  real(c_double), intent(inout) :: ab(ell*ell)
  real(c_double) :: out, outb
  integer :: r(0:ell), c(0:ell)
  external :: det_primal_f
  outb = 1.0d0
  call enzyme_autodiff(det_primal_f, enzyme_const, ell, enzyme_dup, a, ab, &
                       enzyme_dupnoneed, out, outb, enzyme_const, r, &
                       enzyme_const, c)
end subroutine det_enzyme_gradient
