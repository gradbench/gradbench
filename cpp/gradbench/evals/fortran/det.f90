! Fortran primal for the det eval (cmpad det_of_minor, recursive),
! shared by the Fortran AD tools (tapenade, enzyme-fortran). 0-based
! arrays so the index bookkeeping is identical. det_of_minor lives in a module
! so that Tapenade's recursive adjoint sees its interface.
module det_mod
contains
recursive function det_of_minor(n, a, m, r, c) result(detm)
  implicit none
  integer, intent(in) :: n, m
  double precision, intent(in) :: a(0:n*n-1)
  integer, intent(inout) :: r(0:n), c(0:n)
  double precision :: detm, m0j, dets
  integer :: r0, cj, cj1, j, sgn
  r0 = r(n)
  cj = c(n)
  if (m == 1) then
    detm = a(r0*n + cj)
    return
  end if
  detm = 0.0d0
  sgn = 1
  r(n) = r(r0)
  cj1 = n
  do j = 0, m - 1
    m0j = a(r0*n + cj)
    c(cj1) = c(cj)
    dets = det_of_minor(n, a, m - 1, r, c)
    c(cj1) = cj
    if (sgn > 0) then
      detm = detm + m0j*dets
    else
      detm = detm - m0j*dets
    end if
    cj1 = cj
    cj = c(cj)
    sgn = -sgn
  end do
  r(n) = r0
end function det_of_minor
end module det_mod

subroutine det_primal_f(ell, a, out, r, c)
  use det_mod
  implicit none
  integer, intent(in) :: ell
  double precision, intent(in) :: a(0:ell*ell-1)
  double precision, intent(out) :: out
  integer, intent(inout) :: r(0:ell), c(0:ell)
  integer :: i
  do i = 0, ell - 1
    r(i) = i + 1
    c(i) = i + 1
  end do
  r(ell) = 0
  c(ell) = 0
  out = det_of_minor(ell, a, ell, r, c)
end subroutine det_primal_f
