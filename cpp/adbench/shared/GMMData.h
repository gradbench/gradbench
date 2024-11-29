// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/GMMData.h

#pragma once 

#include <vector>

#include "defs.h"

struct GMMInput {
    int d, k, n;
    std::vector<double> alphas, means, icf, x;
    Wishart wishart;
};

struct GMMOutput {
    double objective;
    std::vector<double> gradient;
};

struct GMMParameters {
    bool replicate_point;
};
