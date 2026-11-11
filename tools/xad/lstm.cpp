#include "gradbench/evals/lstm.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  using mode = xad::adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;

  // Passive (never registered) copies, as in the other C++ tools.
  std::vector<AD> _state;
  std::vector<AD> _sequence;

public:
  Jacobian(lstm::Input& input)
      : Function(input), _state(_input.state.begin(), _input.state.end()),
        _sequence(_input.sequence.begin(), _input.sequence.end()) {}

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);

    _tape.clearAll();
    std::vector<AD> main_params(_input.main_params.begin(),
                                _input.main_params.end());
    std::vector<AD> extra_params(_input.extra_params.begin(),
                                 _input.extra_params.end());
    _tape.registerInputs(main_params);
    _tape.registerInputs(extra_params);
    _tape.newRecording();

    AD loss;
    lstm::objective(_input.l, _input.c, _input.b, main_params.data(),
                    extra_params.data(), _state.data(), _sequence.data(),
                    &loss);

    _tape.registerOutput(loss);
    xad::derivative(loss) = 1.0;
    _tape.computeAdjoints();

    size_t o = 0;
    for (auto& p : main_params) {
      output[o++] = xad::derivative(p);
    }
    for (auto& p : extra_params) {
      output[o++] = xad::derivative(p);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<lstm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
