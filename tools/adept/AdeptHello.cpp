#include "AdeptHello.h"
#include "adbench/shared/hello.h"
#include "adept.h"

void AdeptHello::prepare(HelloInput&& input)
{
    _input = input;
}

HelloOutput AdeptHello::output()
{
    return _output;
}

void AdeptHello::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
      _output.objective = hello_objective(_input.x);
    }
}

void AdeptHello::calculate_jacobian(int times)
{
  using adept::adouble;
  using adept::Real;
  adept::Stack s;
  adouble x = _input.x;

  for (int i = 0; i < times; ++i) {
    s.new_recording();
    adouble y = hello_objective<adouble>(x);
    y.set_gradient(1.0);
    s.reverse();
    s.independent(&x, 1);
    s.dependent(y);
    s.jacobian(&_output.gradient);
  }
}
