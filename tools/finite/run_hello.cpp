#include "gradbench/main.hpp"
#include "gradbench/evals/hello.hpp"
#include "finite.h"

class Double : public Function<hello::Input, hello::DoubleOutput> {
  FiniteDifferencesEngine<double> engine;
public:
  Double(hello::Input& input) : Function(input) {
    engine.set_max_output_size(1);
  }

  void compute(hello::DoubleOutput& output) {
    engine.finite_differences([&](double* x, double *out) {
      *out = hello::square(*x);
    }, &_input, 1, 1, &output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"square", function_main<hello::Square>},
      {"double", function_main<Double>}
    });;
}
