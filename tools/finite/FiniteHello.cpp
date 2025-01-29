#include "FiniteHello.h"
#include "adbench/shared/hello.h"
#include "finite.h"

FiniteHello::FiniteHello(HelloInput& input) : ITest(input) {
  engine.set_max_output_size(1);
}

void FiniteHello::calculate_objective() {
  _output.objective = hello_objective(_input.x);
}

void FiniteHello::calculate_jacobian() {
  engine.finite_differences([&](double* x, double *out) {
    *out = hello_objective(*x);
  }, &_input.x, 1, 1, &_output.gradient);
}
