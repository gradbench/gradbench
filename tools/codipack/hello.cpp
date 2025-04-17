#include "gradbench/main.hpp"
#include "gradbench/evals/hello.hpp"
#include "codi_impl.hpp"

class Double : public Function<hello::Input, hello::DoubleOutput>, CoDiForwardRunner {
  using Real = typename CoDiForwardRunner::Real;
public:
  Double(hello::Input& input) : Function(input) {}

  void compute(hello::DoubleOutput& output) {
    Real x(_input);
    codiSetGradient(x, 1.0);
    auto y = hello::square<Real>(x);
    output = codiGetGradient(y);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"square", function_main<hello::Square>},
      {"double", function_main<Double>}
    });;
}
