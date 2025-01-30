#include "ManualHello.h"
#include "adbench/shared/hello.h"
#include "hello_d.h"

ManualHello::ManualHello(HelloInput& input) : ITest(input) {}

void ManualHello::calculate_objective() {

  _output.objective = hello_objective(_input.x);
}

void ManualHello::calculate_jacobian() {
  _output.gradient = hello_objective_d(_input.x);
}
