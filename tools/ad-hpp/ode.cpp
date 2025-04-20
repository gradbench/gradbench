#include <algorithm>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/ode.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000;  // Determined experimentally.

class Gradient : public Function<ode::Input, ode::GradientOutput> {
  std::vector<adjoint_t> _x;

public:
  Gradient(ode::Input& input) : Function(input), _x(_input.x.size()) {
    adjoint::global_tape = adjoint::tape_t::create(TAPE_SIZE);
    std::copy(_input.x.begin(), _input.x.end(), _x.begin());
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    adjoint::global_tape->reset();

    for (auto& x : _x) {
      adjoint::global_tape->register_variable(x);
    }

    std::vector<adjoint_t> primal_out(n);
    ode::primal(n, _x.data(), _input.s, primal_out.data());

    ad::derivative(primal_out[n - 1]) = 1.0;
    adjoint::global_tape->interpret_adjoint();

    for (size_t i = 0; i < n; i++) {
      output[i] = ad::derivative(_x[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<ode::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}
