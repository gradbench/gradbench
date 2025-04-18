// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteLSTM.cpp

#include "gradbench/evals/lstm.hpp"
#include "finite.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  FiniteDifferencesEngine<double> _engine;

public:
  Jacobian(lstm::Input& input) : Function(input) {
    _engine.set_max_output_size(1);
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);

    _engine.finite_differences(
        1,
        [&](double* main_params_in, double* loss) {
          lstm::objective(_input.l, _input.c, _input.b, main_params_in,
                          _input.extra_params.data(), _input.state.data(),
                          _input.sequence.data(), loss);
        },
        _input.main_params.data(), _input.main_params.size(), 1, output.data());

    _engine.finite_differences(
        1,
        [&](double* extra_params_in, double* loss) {
          lstm::objective(_input.l, _input.c, _input.b,
                          _input.main_params.data(), extra_params_in,
                          _input.state.data(), _input.sequence.data(), loss);
        },
        _input.extra_params.data(), _input.extra_params.size(), 1,
        &output.data()[2 * _input.l * 4 * _input.b]);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<lstm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
