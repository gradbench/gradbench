#include <vector>

#include "ad.hpp"
#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
  std::vector<adjoint_t> _x;

  ad::shared_global_tape_ptr<adjoint> _tape;

public:
  Gradient(llsq::Input& input) : Function(input), _x(_input.x.size()) {
    size_t m = _input.x.size();
    for (size_t i = 0; i < m; i++) {
      _x[i] = _input.x[i];
    }
  }

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();

    output.resize(m);

    _tape->reset();

    for (size_t i = 0; i < m; i++) {
      _tape->register_variable(_x[i]);
    }

    adjoint_t y;

    llsq::primal<adjoint_t>(n, m, _x.data(), &y);

    ad::derivative(y) = 1.0;
    _tape->interpret_adjoint();

    for (size_t i = 0; i < m; i++) {
      output[i] = ad::derivative(_x[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<llsq::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}
