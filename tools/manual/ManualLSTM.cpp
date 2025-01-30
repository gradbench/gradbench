// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualLSTM.cpp

// ManualLSTM.cpp : Defines the exported functions for the DLL.
#include "ManualLSTM.h"
#include "adbench/shared/lstm.h"
#include "lstm_d.h"

ManualLSTM::ManualLSTM(LSTMInput& input) : ITest(input) {
  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0, std::vector<double>(Jcols) };
}

void ManualLSTM::calculate_objective() {
  lstm_objective(_input.l, _input.c, _input.b,
                 _input.main_params.data(), _input.extra_params.data(),
                 _input.state.data(), _input.sequence.data(),
                 &_output.objective);
}

void ManualLSTM::calculate_jacobian() {
  lstm_objective_d(_input.l, _input.c, _input.b,
                   _input.main_params.data(), _input.extra_params.data(),
                   _input.state, _input.sequence.data(),
                   &_output.objective, _output.gradient.data());
}
