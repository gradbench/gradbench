! Fortran primal for the ode eval (cmpad Runge-Kutta).
! Shared by the Fortran AD tools (tapenade, enzyme-fortran).
subroutine ode_fun(n, x, y, z)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x(n), y(n)
  double precision, intent(out) :: z(n)
  integer :: i
  z(1) = x(1)
  do i = 2, n
    z(i) = x(i) * y(i - 1)
  end do
end subroutine ode_fun

subroutine ode_primal_f(n, xi, s, yf)
  implicit none
  integer, intent(in) :: n, s
  double precision, intent(in) :: xi(n)
  double precision, intent(out) :: yf(n)
  double precision :: k1(n), k2(n), k3(n), k4(n), y_tmp(n)
  double precision :: h
  integer :: i_step
  h = 2.0d0 / s
  yf = 0.0d0
  do i_step = 1, s
    call ode_fun(n, xi, yf, k1)
    y_tmp = yf + h * k1 / 2.0d0
    call ode_fun(n, xi, y_tmp, k2)
    y_tmp = yf + h * k2 / 2.0d0
    call ode_fun(n, xi, y_tmp, k3)
    y_tmp = yf + h * k3
    call ode_fun(n, xi, y_tmp, k4)
    yf = yf + h * (k1 + 2.0d0 * k2 + 2.0d0 * k3 + k4) / 6.0d0
  end do
end subroutine ode_primal_f
