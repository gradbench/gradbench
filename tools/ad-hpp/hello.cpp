#include "gradbench/evals/hello.hpp"
#include "ad.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

class Double : public Function<hello::Input, hello::DoubleOutput> {
public:
  Double(hello::Input& input) : Function(input) {}

  void compute(hello::DoubleOutput& output) {
    adjoint::global_tape = adjoint::tape_t::create();

    adjoint_t x = _input;
    adjoint::global_tape->register_variable(x);

    adjoint_t y = hello::square(x);

    ad::derivative(y) = 1.0;
    adjoint::global_tape->interpret_adjoint();
    output = ad::derivative(x);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"square", function_main<hello::Square>},
                       {"double", function_main<Double>}});
  ;
}
