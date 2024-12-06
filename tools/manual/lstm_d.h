// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/lstm_d.h

#pragma once

#include <vector>

// Manual gradient of lstm_objective loss (output)
// with relation to main_params and extra_params (inputs)
void lstm_objective_d(int l, int c, int b,
    const double* main_params, const double* extra_params,
    std::vector<double> state, const double* sequence,
    double* loss, double* J);
