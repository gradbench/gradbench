#include <algorithm>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/ode.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000;  // Determined experimentally.

class Gradient : public Function<ode::Input, ode::GradientOutput> {
  std::vector<adjoint_t>              _x;
  ad::shared_global_tape_ptr<adjoint> _tape;
  adjoint::tape_t::position_t         _input_pos;

public:
  Gradient(ode::Input& input)
      : Function(input), _x(_input.x.size()),
        _tape(adjoint::tape_options_t(TAPE_SIZE)) {
    std::copy(_input.x.begin(), _input.x.end(), _x.begin());
    _tape->register_variable(_x);
    _input_pos = _tape->get_position();
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);
    std::for_each(_x.begin(), _x.end(),
                  [&](adjoint_t& v) -> void { ad::derivative(v) = 0.0; });
    _tape->reset_to(_input_pos);

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
