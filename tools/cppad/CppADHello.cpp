#include "CppADHello.h"
#include "adbench/shared/hello.h"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

CppADHello::CppADHello(HelloInput& input) : ITest(input) {
  std::vector<ADdouble> X(1);
  std::vector<ADdouble> Y(1);
  X[0] = _input.x;
  CppAD::Independent(X);

  Y[0] = hello_objective(X[0]);
  _tape = new CppAD::ADFun<double>(X, Y);
}

void CppADHello::calculate_objective() {
  _output.objective = hello_objective(_input.x);
}

void CppADHello::calculate_jacobian() {
  std::vector<double> dx(1);
  dx[0] = 1;
  std::vector<double> dy(1);

  dy = _tape->Forward(1, dx);
  _output.gradient = dy[0];
}
