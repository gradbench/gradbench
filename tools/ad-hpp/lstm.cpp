#include <algorithm>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/lstm.hpp"
#include "gradbench/main.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

static const int TAPE_SIZE = 1000000000;  // Determined experimentally.

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  std::vector<adjoint_t> _main_params;
  std::vector<adjoint_t> _extra_params;
  std::vector<adjoint_t> _state;
  std::vector<adjoint_t> _sequence;

  ad::shared_global_tape_ptr<adjoint> _tape;
  adjoint::tape_t::position_t         _input_pos;

public:
  Jacobian(lstm::Input& input)
      : Function(input), _main_params(_input.main_params.size()),
        _extra_params(_input.extra_params.size()), _state(_input.state.size()),
        _sequence(_input.sequence.size()),
        _tape(adjoint::tape_options_t(TAPE_SIZE)) {

    std::copy(_input.main_params.begin(), _input.main_params.end(),
              _main_params.begin());
    std::copy(_input.extra_params.begin(), _input.extra_params.end(),
              _extra_params.begin());
    std::copy(_input.state.begin(), _input.state.end(), _state.begin());
    std::copy(_input.sequence.begin(), _input.sequence.end(),
              _sequence.begin());

    _tape->register_variable(_main_params);
    _tape->register_variable(_extra_params);
    _input_pos = _tape->get_position();
    std::cout << "Tape memory is: " << _tape->get_allocated_tape_memory_size()
              << "\n";
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    _tape->reset_to(_input_pos);

    std::for_each(_main_params.begin(), _extra_params.end(),
                  [&](adjoint_t& v) -> void { ad::derivative(v) = 0.0; });

    std::for_each(_extra_params.begin(), _extra_params.end(),
                  [&](adjoint_t& v) -> void { ad::derivative(v) = 0.0; });

    adjoint_t loss;
    lstm::objective(_input.l, _input.c, _input.b, _main_params.data(),
                    _extra_params.data(), _state.data(), _sequence.data(),
                    &loss);

    ad::derivative(loss) = 1.0;
    _tape->interpret_adjoint();

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
  return generic_main(argc, argv,
                      {{"objective", function_main<lstm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
