// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/HandData.h

#pragma once

#include <vector>

#include "utils.h"

struct HandInput
{
    std::vector<double> theta;
    HandDataLightMatrix data;
    std::vector<double> us;
};

struct HandOutput {
    std::vector<double> objective;
    int jacobian_ncols, jacobian_nrows;
    std::vector<double> jacobian;
};

struct HandParameters {
    bool is_complicated;
};
