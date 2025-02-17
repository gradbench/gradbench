// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeLSTM.cpp

#include <algorithm>

#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"

#include "lstm/lstm.h"
#include "lstm/lstm_b.h"

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  std::vector<double> _state;

public:
  Jacobian(lstm::Input& input) : Function(input) {}

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    double* main_params_gradient_part = output.data();
    double* extra_params_gradient_part = output.data() + _input.main_params.size();

    double loss = 0.0;      // stores fictive output
    // (Tapenade doesn't calculate an original function in reverse mode)

    double lossb = 1.0;     // stores dY
    // (equals to 1.0 for gradient calculation)

    _state = _input.state;
    lstm_objective_b(
                     _input.l,
                     _input.c,
                     _input.b,
                     _input.main_params.data(),
                     main_params_gradient_part,
                     _input.extra_params.data(),
                     extra_params_gradient_part,
                     _state.data(),
                     _input.sequence.data(),
                     &loss,
                     &lossb
                     );

  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
