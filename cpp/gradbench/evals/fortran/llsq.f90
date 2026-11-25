! Fortran primal for the llsq eval. Shared by the Fortran AD tools
! (tapenade, enzyme-fortran).
subroutine llsq_primal_f(n, m, x, out)
  implicit none
  integer, intent(in) :: n, m
  double precision, intent(in) :: x(m)
  double precision, intent(out) :: out
  double precision :: total, ti, inner, mul
  integer :: i, j
  total = 0.0d0
  do i = 0, n - 1
    ti = -1.0d0 + i * 2.0d0 / (n - 1)
    if (ti > 0.0d0) then
      inner = 1.0d0
    else if (ti < 0.0d0) then
      inner = -1.0d0
    else
      inner = 0.0d0
    end if
    mul = 1.0d0
    do j = 1, m
      inner = inner - x(j) * mul
      mul = mul * ti
    end do
    total = total + inner * inner
  end do
  out = total / 2.0d0
end subroutine llsq_primal_f
