#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"
#include <codi.hpp>

using Real = codi::RealReverse;
using Tape = typename Real::Tape;

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  std::vector<Real> main_params_d;
  std::vector<Real> extra_params_d;
  std::vector<Real> state_d;
  std::vector<Real> sequence_d;
public:
  Jacobian(lstm::Input& input) :
    Function(input),
    main_params_d(_input.main_params.size()),
    extra_params_d(_input.extra_params.size()),
    state_d(_input.state.size()),
    sequence_d(_input.sequence.size()) {

    Tape& tape = Real::getTape();
    tape.setActive();

    for (size_t i = 0; i < main_params_d.size(); i++) {
      main_params_d[i] = _input.main_params[i];
      tape.registerInput(main_params_d[i]);
    }

    for (size_t i = 0; i < extra_params_d.size(); i++) {
      extra_params_d[i] = _input.extra_params[i];
      tape.registerInput(extra_params_d[i]);
    }


    for (size_t i = 0; i < state_d.size(); i++) {
      state_d[i] = _input.state[i];
    }

    for (size_t i = 0; i < sequence_d.size(); i++) {
      sequence_d[i] = _input.sequence[i];
    }
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);

    Real loss;
    Tape& tape = Real::getTape();

    lstm::objective(_input.l, _input.c, _input.b,
                    main_params_d.data(), extra_params_d.data(),
                    state_d.data(), sequence_d.data(),
                    &loss);

    tape.registerOutput(loss);
    tape.setPassive();
    loss.setGradient(1.0);
    tape.evaluate();

    int o = 0;
    for (size_t i = 0; i < main_params_d.size(); i++) {
      output[o++] = main_params_d[i].getGradient();
    }
    for (size_t i = 0; i < extra_params_d.size(); i++) {
      output[o++] = extra_params_d[i].getGradient();
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
