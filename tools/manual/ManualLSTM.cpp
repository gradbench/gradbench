// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualLSTM.cpp

// ManualLSTM.cpp : Defines the exported functions for the DLL.
#include "ManualLSTM.h"
#include "adbench/shared/lstm.h"
#include "lstm_d.h"

// This function must be called before any other function.
void ManualLSTM::prepare(LSTMInput&& input)
{
    this->input = input;
    int Jcols = 8 * this->input.l * this->input.b + 3 * this->input.b;
    result = { 0, std::vector<double>(Jcols) };
}

LSTMOutput ManualLSTM::output()
{
    return result;
}

void ManualLSTM::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
      lstm_objective(input.l, input.c, input.b, input.main_params.data(), input.extra_params.data(), input.state.data(), input.sequence.data(), &result.objective);
    }
}

void ManualLSTM::calculate_jacobian(int times)
{
    for (int i = 0; i < times; ++i) {
      lstm_objective_d(input.l, input.c, input.b, input.main_params.data(), input.extra_params.data(), input.state, input.sequence.data(), &result.objective, result.gradient.data());
    }
}
