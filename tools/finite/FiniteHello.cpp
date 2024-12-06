#include "FiniteHello.h"
#include "adbench/shared/hello.h"
#include "finite.h"

void FiniteHello::prepare(HelloInput&& input)
{
    _input = input;
    engine.set_max_output_size(1);
}

HelloOutput FiniteHello::output()
{
    return _output;
}

void FiniteHello::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
      _output.objective = hello_objective(_input.x);
    }
}

void FiniteHello::calculate_jacobian(int times)
{
  for (int i = 0; i < times; ++i) {
    engine.finite_differences([&](double* x, double *out) {
      *out = hello_objective(*x);
    }, &_input.x, 1, 1, &_output.gradient);
  }
}
