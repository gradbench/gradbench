// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/lstm/lstm.h

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <math.h>

// LSTM objective (loss function)
// Input variables: main_params (8 * l * b), extra_params (3 * b)
// Output variable: loss (scalar)
// Parameters:
//      state (2 * l * b)
//      sequence (c * b)
void lstm_objective(
    int l,
    int c,
    int b,
    double const* main_params,
    double const* extra_params,
    double* state,
    double const* sequence,
    double* loss
);

#ifdef __cplusplus
}
#endif
