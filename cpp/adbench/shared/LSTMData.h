// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/LSTMData.h

#pragma once

#include <vector>

struct LSTMInput
{
    int l;
    int c;
    int b;
    std::vector<double> main_params;
    std::vector<double> extra_params;
    std::vector<double> state;
    std::vector<double> sequence;
};

struct LSTMOutput {
    double objective;
    std::vector<double> gradient;
};
