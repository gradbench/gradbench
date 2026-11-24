! Fortran primal for the lse eval. Shared by the Fortran AD tools
! (tapenade, enzyme-fortran).
subroutine lse_primal_f(n, x, out)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  double precision, intent(out) :: out
  double precision :: a, s
  integer :: i
  a = x(1)
  do i = 2, n
    if (x(i) > a) a = x(i)
  end do
  s = 0.0d0
  do i = 1, n
    s = s + exp(x(i) - a)
  end do
  out = log(s) + a
end subroutine lse_primal_f
