#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"
#include "ad.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) :
    Function(input) { }

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();

    output.resize(m);

    adjoint::global_tape = adjoint::tape_t::create();

    std::vector<adjoint_t> x(m);

    for (size_t i = 0; i < m; i++) {
      x[i] = _input.x[i];
      adjoint::global_tape->register_variable(x[i]);
    }

    adjoint_t y;

    llsq::primal<adjoint_t>(n, m, x.data(), &y);

    ad::derivative(y) = 1.0;
    adjoint::global_tape->interpret_adjoint();

    for (size_t i = 0; i < m; i++) {
      output[i] = ad::derivative(x[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
