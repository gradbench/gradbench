#include "gradbench/evals/lstm.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include "codi_impl.hpp"

class Jacobian : public Function<lstm::Input, lstm::JacOutput>,
                 CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> main_params_d;
  std::vector<Real> extra_params_d;
  std::vector<Real> state_d;
  std::vector<Real> sequence_d;

  Real loss;

public:
  Jacobian(lstm::Input& input)
      : Function(input), main_params_d(_input.main_params.size()),
        extra_params_d(_input.extra_params.size()),
        state_d(_input.state.size()), sequence_d(_input.sequence.size()),
        loss() {

    for (size_t i = 0; i < main_params_d.size(); i++) {
      main_params_d[i] = _input.main_params[i];
    }

    for (size_t i = 0; i < extra_params_d.size(); i++) {
      extra_params_d[i] = _input.extra_params[i];
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

    codiStartRecording();

    for (size_t i = 0; i < main_params_d.size(); i++) {
      codiAddInput(main_params_d[i]);
    }

    for (size_t i = 0; i < extra_params_d.size(); i++) {
      codiAddInput(extra_params_d[i]);
    }

    lstm::objective(_input.l, _input.c, _input.b, main_params_d.data(),
                    extra_params_d.data(), state_d.data(), sequence_d.data(),
                    &loss);

    codiAddOutput(loss);
    codiStopRecording();

    codiSetGradient(loss, 1.0);
    codiEval();

    int o = 0;
    for (size_t i = 0; i < main_params_d.size(); i++) {
      output[o++] = codiGetGradient(main_params_d[i]);
    }
    for (size_t i = 0; i < extra_params_d.size(); i++) {
      output[o++] = codiGetGradient(extra_params_d[i]);
    }

    codiCleanup();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<lstm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
