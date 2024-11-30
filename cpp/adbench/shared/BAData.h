// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/BAData.h

#pragma once

#include <vector>

#include "defs.h"
#include "utils.h"

struct BAInput {
    int n = 0, m = 0, p = 0;
    std::vector<double> cams, X, w, feats;
    std::vector<int> obs;
};

struct BAOutput {
    std::vector<double> reproj_err;
    std::vector<double> w_err;
    BASparseMat J;
};
