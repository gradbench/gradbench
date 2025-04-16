#include <algorithm>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/lse.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

class Gradient : public Function<lse::Input, lse::GradientOutput> {
  std::vector<adjoint_t> _x_d;

public:
  Gradient(lse::Input& input) : Function(input), _x_d(_input.x.size()) {
    std::copy(_input.x.begin(), _input.x.end(), _x_d.begin());
    adjoint::global_tape = adjoint::tape_t::create();
  }

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    adjoint::global_tape->reset();

    for (auto& x : _x_d) {
      adjoint::global_tape->register_variable(x);
    }

    adjoint_t error;
    lse::primal(n, _x_d.data(), &error);

    ad::derivative(error) = 1;

    adjoint::global_tape->interpret_adjoint();

    for (size_t i = 0; i < n; i++) {
      output[i] = ad::derivative(_x_d[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<lse::Primal>},
                       {"gradient", function_main<Gradient>}});
}
