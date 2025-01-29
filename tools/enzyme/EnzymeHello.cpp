#include "EnzymeHello.h"
#include "adbench/shared/hello.h"

EnzymeHello::EnzymeHello(HelloInput& input) : ITest(input) {}

void EnzymeHello::calculate_objective() {
  _output.objective = hello_objective(_input.x);
}

extern double __enzyme_autodiff(void*, double);

void EnzymeHello::calculate_jacobian() {
  _output.gradient = __enzyme_autodiff((void*)hello_objective<double>, _input.x);
}
