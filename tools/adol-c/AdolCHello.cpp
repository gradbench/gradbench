#include "AdolCHello.h"
#include "adbench/shared/hello.h"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

void AdolCHello::prepare(HelloInput&& input) {
  _input = input;
  adouble ax;
  trace_on(tapeTag);
  ax <<= _input.x;
  hello_objective<adouble>(ax) >>= _output.objective;
  trace_off();
}

HelloOutput AdolCHello::output() {
  return _output;
}

void AdolCHello::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    _output.objective = hello_objective(_input.x);
  }
}

void AdolCHello::calculate_jacobian(int times) {
  for (int i = 0; i < times; ++i) {
    double in = _input.x;
    gradient(tapeTag, 1, &in, &_output.gradient);
  }
}
