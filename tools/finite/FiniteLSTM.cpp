// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteLSTM.cpp

#include "FiniteLSTM.h"
#include "adbench/shared/lstm.h"
#include "finite.h"

// This function must be called before any other function.
void FiniteLSTM::prepare(LSTMInput&& input)
{
    this->input = input;
    int Jcols = 8 * this->input.l * this->input.b + 3 * this->input.b;
    result = { 0, std::vector<double>(Jcols) };
    engine.set_max_output_size(1);
}

LSTMOutput FiniteLSTM::output()
{
    return result;
}

void FiniteLSTM::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
        lstm_objective(input.l, input.c, input.b, input.main_params.data(), input.extra_params.data(), input.state.data(), input.sequence.data(), &result.objective);
    }
}

void FiniteLSTM::calculate_jacobian(int times)
{
    for (int i = 0; i < times; ++i) {
        engine.finite_differences([&](double* main_params_in, double* loss) {
            lstm_objective(input.l, input.c, input.b, main_params_in,
                           input.extra_params.data(), input.state.data(), input.sequence.data(), loss);
            }, input.main_params.data(), input.main_params.size(), 1, result.gradient.data());

        engine.finite_differences([&](double* extra_params_in, double* loss) {
            lstm_objective(input.l, input.c, input.b, input.main_params.data(),
                           extra_params_in, input.state.data(), input.sequence.data(), loss);
            }, input.extra_params.data(), input.extra_params.size(), 1, &result.gradient.data()[2 * input.l * 4 * input.b]);
    }
}
