// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/gmm/gmm_b.h

// Changes made:
//
// * modified #include paths.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>

#include "../utils/adBuffer.h"
#include "adbench/shared/defs.h"

// GMM function differentiated in reverse mode by Tapenade.
void gmm_objective_b(
    int d,
    int k,
    int n,
    double const* alphas,
    double* alphasb,
    double const* means,
    double* meansb,
    double const* icf,
    double* icfb,
    double const* x,
    Wishart wishart,
    double* err,
    double* errb
);

#ifdef __cplusplus
}
#endif
