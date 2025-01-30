// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Adept/main.cpp

#include "AdeptBA.h"
#include "adbench/shared/ba.h"
#include "adept.h"

using adept::adouble;

void compute_ba_J(int n, int m, int p,
                  double *cams, double *X, double *w, int *obs, double *feats,
                  double *reproj_err, double *w_err, BASparseMat *J) {

  adept::Stack stack;
  adouble acam[BA_NCAMPARAMS],
    aX[3], aw, areproj_err[2], aw_err;
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;
  std::vector<double> reproj_err_d(2 * n_new_cols);

  for (int i = 0; i < p; i++) {
    std::fill(reproj_err_d.begin(), reproj_err_d.end(), (double)0);

    int camIdx = obs[2 * i + 0];
    int ptIdx = obs[2 * i + 1];
    adept::set_values(acam, BA_NCAMPARAMS, &cams[BA_NCAMPARAMS*camIdx]);
    adept::set_values(aX, 3, &X[ptIdx * 3]);
    aw.set_value(w[i]);

    stack.new_recording();
    computeReprojError(acam, aX, &aw, &feats[2 * i], areproj_err);
    stack.independent(acam, BA_NCAMPARAMS);
    stack.independent(aX, 3);
    stack.independent(aw);
    stack.dependent(areproj_err, 2);
    stack.jacobian_reverse(reproj_err_d.data());

    adept::get_values(areproj_err, 2, &reproj_err[2 * i]);

    J->insert_reproj_err_block(i, camIdx, ptIdx, reproj_err_d.data());
  }

  for (int i = 0; i < p; i++) {
    aw.set_value(w[i]);

    stack.new_recording();
    computeZachWeightError(&aw, &aw_err);
    aw_err.set_gradient(1.);
    stack.reverse();

    w_err[i] = aw_err.value();
    double err_d = aw.get_gradient();

    J->insert_w_err_block(i, err_d);
  }
}

AdeptBA::AdeptBA(BAInput& input) : ITest(input) {
  _output = {
    std::vector<double>(2 * _input.p),
    std::vector<double>(_input.p),
    BASparseMat(_input.n, _input.m, _input.p)
  };
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;
  _reproj_err_d = std::vector<double>(2 * n_new_cols);
}

void AdeptBA::calculate_objective() {
  ba_objective(_input.n, _input.m, _input.p, _input.cams.data(), _input.X.data(), _input.w.data(),
               _input.obs.data(), _input.feats.data(), _output.reproj_err.data(), _output.w_err.data());
}

void AdeptBA::calculate_jacobian() {
  _output.J.clear();
  compute_ba_J(_input.n, _input.m, _input.p,
               _input.cams.data(), _input.X.data(), _input.w.data(),
               _input.obs.data(), _input.feats.data(), _output.reproj_err.data(), _output.w_err.data(),
               &_output.J);
}
