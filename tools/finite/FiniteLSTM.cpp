// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteLSTM.cpp

#include "FiniteLSTM.h"
#include "adbench/shared/lstm.h"
#include "finite.h"

FiniteLSTM::FiniteLSTM(LSTMInput& input) : ITest(input) {
  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0, std::vector<double>(Jcols) };
  engine.set_max_output_size(1);
}

void FiniteLSTM::calculate_objective() {
  lstm_objective(_input.l, _input.c, _input.b, _input.main_params.data(), _input.extra_params.data(), _input.state.data(), _input.sequence.data(), &_output.objective);
}

void FiniteLSTM::calculate_jacobian() {
  engine.finite_differences([&](double* main_params_in, double* loss) {
    lstm_objective(_input.l, _input.c, _input.b, main_params_in,
                   _input.extra_params.data(), _input.state.data(), _input.sequence.data(), loss);
  }, _input.main_params.data(), _input.main_params.size(), 1, _output.gradient.data());

  engine.finite_differences([&](double* extra_params_in, double* loss) {
    lstm_objective(_input.l, _input.c, _input.b, _input.main_params.data(),
                   extra_params_in, _input.state.data(), _input.sequence.data(), loss);
  }, _input.extra_params.data(), _input.extra_params.size(), 1, &_output.gradient.data()[2 * _input.l * 4 * _input.b]);
}
