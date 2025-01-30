// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteBA.cpp

#include "FiniteBA.h"
#include "adbench/shared/ba.h"
#include "finite.h"

FiniteBA::FiniteBA(BAInput& input) : ITest(input) {
  _output = {
    std::vector<double>(2 * _input.p),
    std::vector<double>(_input.p),
    BASparseMat(_input.n, _input.m, _input.p)
  };
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;
  reproj_err_d = std::vector<double>(2 * n_new_cols);
  engine.set_max_output_size(2);
}

void FiniteBA::calculate_objective() {
  ba_objective(_input.n, _input.m, _input.p, _input.cams.data(), _input.X.data(), _input.w.data(),
               _input.obs.data(), _input.feats.data(), _output.reproj_err.data(), _output.w_err.data());
}

void FiniteBA::calculate_jacobian() {
  _output.J.clear();
  for (int j = 0; j < _input.p; ++j) {
    std::fill(reproj_err_d.begin(), reproj_err_d.end(), (double)0);

    int camIdx = _input.obs[2 * j + 0];
    int ptIdx = _input.obs[2 * j + 1];

    engine.finite_differences([&](double* cam_in, double* reproj_err) {
      computeReprojError(cam_in, &_input.X[ptIdx * 3], &_input.w[j], &_input.feats[2 * j], reproj_err);
    }, &_input.cams[camIdx * BA_NCAMPARAMS], BA_NCAMPARAMS, 2, reproj_err_d.data());

    engine.finite_differences([&](double* X_in, double* reproj_err) {
      computeReprojError(&_input.cams[camIdx * BA_NCAMPARAMS], X_in, &_input.w[j], &_input.feats[2 * j], reproj_err);
    }, &_input.X[ptIdx * 3], 3, 2, &reproj_err_d.data()[2 * BA_NCAMPARAMS]);

    engine.finite_differences([&](double* w_in, double* reproj_err) {
      computeReprojError(&_input.cams[camIdx * BA_NCAMPARAMS], &_input.X[ptIdx * 3], w_in, &_input.feats[2 * j], reproj_err);
    }, &_input.w[j], 1, 2, &reproj_err_d.data()[2 * (BA_NCAMPARAMS + 3)]);

    _output.J.insert_reproj_err_block(j, camIdx, ptIdx, reproj_err_d.data());
  }

  double w_d;

  for (int j = 0; j < _input.p; ++j) {
    engine.finite_differences([&](double* w_in, double* w_er) {
      computeZachWeightError(w_in, w_er);
    }, &_input.w[j], 1, 1, &w_d);

    _output.J.insert_w_err_block(j, w_d);
  }
}
