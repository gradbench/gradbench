#include "AdolCHello.h"
#include "adbench/shared/hello.h"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

AdolCHello::AdolCHello(HelloInput& input) : ITest(input) {
  // Construct tape.
  adouble ax;
  trace_on(tapeTag);
  ax <<= _input.x;
  hello_objective<adouble>(ax) >>= _output.objective;
  trace_off();
}

void AdolCHello::calculate_objective() {
  _output.objective = hello_objective(_input.x);
}

void AdolCHello::calculate_jacobian() {
  double in = _input.x;
  gradient(tapeTag, 1, &in, &_output.gradient);
}
