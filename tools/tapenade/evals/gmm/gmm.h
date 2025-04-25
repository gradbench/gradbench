// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/gmm/gmm.h

// Changes made:
//
// * modified #include paths.

#pragma once

#include "adbench/shared/defs.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <stdlib.h>

// d: dim
// k: number of gaussians
// n: number of points
// alphas: k logs of mixture weights (unnormalized), so
//          weights = exp(log_alphas) / sum(exp(log_alphas))
// means: d*k component means
// icf: (d*(d+1)/2)*k inverse covariance factors
//                  every icf entry stores firstly log of diagonal and then
//          columnwise other entris
//          To generate icf in MATLAB given covariance C :
//              L = inv(chol(C, 'lower'));
//              inv_cov_factor = [log(diag(L)); L(au_tril_indices(d, -1))]
// x: d*n points
// wishart: wishart distribution parameters
// err: 1 output
void gmm_objective(int d, int k, int n, double const* alphas,
                   double const* means, double const* icf, double const* x,
                   Wishart wishart, double* err);

#ifdef __cplusplus
}
#endif
