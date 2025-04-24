#include <algorithm>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/det.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000;  // Determined experimentally.

class Gradient : public Function<det::Input, det::GradientOutput> {
  std::vector<adjoint_t> _A;

  ad::shared_global_tape_ptr<adjoint> _tape;

public:
  Gradient(det::Input& input)
      : Function(input), _A(_input.A.size()),
        _tape(adjoint::tape_options_t(TAPE_SIZE)) {
    std::copy(_input.A.begin(), _input.A.end(), _A.begin());
  }

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell * ell);

    _tape->reset();

    for (auto& x : _A) {
      adjoint::global_tape->register_variable(x);
    }

    adjoint_t error;
    det::primal(ell, _A.data(), &error);

    ad::derivative(error) = 1.0;
    _tape->interpret_adjoint();

    for (size_t i = 0; i < ell * ell; i++) {
      output[i] = ad::derivative(_A[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<det::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}
