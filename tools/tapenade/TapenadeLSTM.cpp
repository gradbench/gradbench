// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeLSTM.cpp

#include "TapenadeLSTM.h"

TapenadeLSTM::TapenadeLSTM(LSTMInput& input) : ITest(input) {
  state = std::vector<double>(_input.state.size());

  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0, std::vector<double>(Jcols) };
}

void TapenadeLSTM::calculate_objective() {
  state = _input.state;
  lstm_objective(_input.l,
                 _input.c,
                 _input.b,
                 _input.main_params.data(),
                 _input.extra_params.data(),
                 state.data(),
                 _input.sequence.data(),
                 &_output.objective
                 );
}

void TapenadeLSTM::calculate_jacobian() {
  double* main_params_gradient_part = _output.gradient.data();
  double* extra_params_gradient_part = _output.gradient.data() + _input.main_params.size();

  double loss = 0.0;      // stores fictive _output
  // (Tapenade doesn't calculate an original function in reverse mode)

  double lossb = 1.0;     // stores dY
  // (equals to 1.0 for gradient calculation)

  state = _input.state;
  lstm_objective_b(
                   _input.l,
                   _input.c,
                   _input.b,
                   _input.main_params.data(),
                   main_params_gradient_part,
                   _input.extra_params.data(),
                   extra_params_gradient_part,
                   state.data(),
                   _input.sequence.data(),
                   &loss,
                   &lossb
                   );
}
