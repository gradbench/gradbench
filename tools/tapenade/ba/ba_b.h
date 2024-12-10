// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/ba/ba_b.h

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
    
#include "../utils/adBuffer.h"
#include "adbench/shared/defs.h"



// Reprojection error function differentiated in reverse mode by Tapenade.
void compute_reproj_error_b(
    const double* cam,
    double* camb,
    const double* X,
    double* Xb,
    const double* w,
    double* wb,
    const double* feat,
    double* err,
    double* errb
);



// Weight error function differentiated in reverse mode by Tapenade.
void compute_zach_weight_error_b(
    const double* w,
    double* wb,
    double* err,
    double* errb
);

#ifdef __cplusplus
}
#endif
