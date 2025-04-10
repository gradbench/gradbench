#include <iostream>
#include "gradbench/main.hpp"
#include "gradbench/evals/hello.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Double : public Function<hello::Input, hello::DoubleOutput> {
public:
  Double(hello::Input& input) : Function(input) {
    // Construct tape.
    adouble ax;
    double output;
    trace_on(tapeTag);
    ax <<= _input;
    hello::square<adouble>(ax) >>= output;
    trace_off();
  }

  void compute(hello::DoubleOutput& output) {
    gradient(tapeTag, 1, &_input, &output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"square", function_main<hello::Square>},
      {"double", function_main<Double>}
    });;
}
