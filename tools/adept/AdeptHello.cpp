#include "AdeptHello.h"
#include "adbench/shared/hello.h"
#include "adept.h"

AdeptHello::AdeptHello(HelloInput& input) : ITest(input) {}

void AdeptHello::calculate_objective() {
  _output.objective = hello_objective(_input.x);
}

void AdeptHello::calculate_jacobian() {
  using adept::adouble;
  using adept::Real;
  adept::Stack s;
  adouble x = _input.x;

  s.new_recording();
  adouble y = hello_objective<adouble>(x);
  y.set_gradient(1.0);
  s.reverse();
  s.independent(&x, 1);
  s.dependent(y);
  s.jacobian(&_output.gradient);
}
