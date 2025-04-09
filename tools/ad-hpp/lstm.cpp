#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"
#include "ad.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000; // Determined experimentally.

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  std::vector<adjoint_t> _main_params;
  std::vector<adjoint_t> _extra_params;
  std::vector<adjoint_t> _state;
  std::vector<adjoint_t> _sequence;
public:
  Jacobian(lstm::Input& input) :
    Function(input),
    _main_params(_input.main_params.size()),
    _extra_params(_input.extra_params.size()),
    _state(_input.state.size()),
    _sequence(_input.sequence.size()) {

    for (size_t i = 0; i < _main_params.size(); i++) {
      _main_params[i] = _input.main_params[i];
    }

    for (size_t i = 0; i < _extra_params.size(); i++) {
      _extra_params[i] = _input.extra_params[i];
    }

    for (size_t i = 0; i < _state.size(); i++) {
      _state[i] = _input.state[i];
    }

    for (size_t i = 0; i < _sequence.size(); i++) {
      _sequence[i] = _input.sequence[i];
    }
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);

    adjoint::global_tape = adjoint::tape_t::create(TAPE_SIZE);

    for (size_t i = 0; i < _main_params.size(); i++) {
      adjoint::global_tape->register_variable(_main_params[i]);
    }

    for (size_t i = 0; i < _extra_params.size(); i++) {
      adjoint::global_tape->register_variable(_extra_params[i]);
    }

    adjoint_t loss;
    lstm::objective(_input.l, _input.c, _input.b,
                    _main_params.data(), _extra_params.data(),
                    _state.data(), _sequence.data(),
                    &loss);

    ad::derivative(loss) = 1.0;
    adjoint::global_tape->interpret_adjoint();

    int o = 0;
    for (size_t i = 0; i < _main_params.size(); i++) {
      output[o++] = ad::derivative(_main_params[i]);
    }
    for (size_t i = 0; i < _extra_params.size(); i++) {
      output[o++] = ad::derivative(_extra_params[i]);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
