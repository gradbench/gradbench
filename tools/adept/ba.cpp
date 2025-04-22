// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Adept/main.cpp

#include "gradbench/evals/ba.hpp"
#include "adept.h"
#include "gradbench/main.hpp"
#include <algorithm>

using adept::adouble;

void compute_ba_J(int n, int m, int p, double* cams, double* X, double* w,
                  int* obs, double* feats, double* reproj_err, double* w_err,
                  ba::SparseMat* J) {

  adept::Stack        stack;
  adouble             acam[BA_NCAMPARAMS], aX[3], aw, areproj_err[2], aw_err;
  int                 n_new_cols = BA_NCAMPARAMS + 3 + 1;
  std::vector<double> reproj_err_d(2 * n_new_cols);

  for (int i = 0; i < p; i++) {
    std::fill(reproj_err_d.begin(), reproj_err_d.end(), (double)0);

    int camIdx = obs[2 * i + 0];
    int ptIdx  = obs[2 * i + 1];
    adept::set_values(acam, BA_NCAMPARAMS, &cams[BA_NCAMPARAMS * camIdx]);
    adept::set_values(aX, 3, &X[ptIdx * 3]);
    aw.set_value(w[i]);

    stack.new_recording();
    ba::computeReprojError(acam, aX, &aw, &feats[2 * i], areproj_err);
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
    ba::computeZachWeightError(&aw, &aw_err);
    aw_err.set_gradient(1.);
    stack.reverse();

    w_err[i]     = aw_err.value();
    double err_d = aw.get_gradient();

    J->insert_w_err_block(i, err_d);
  }
}

class Jacobian : public Function<ba::Input, ba::JacOutput> {
private:
  std::vector<double> _reproj_err;
  std::vector<double> _w_err;
  std::vector<double> _reproj_err_d;

public:
  Jacobian(ba::Input& input)
      : Function(input), _reproj_err(2 * input.p), _w_err(input.p),
        _reproj_err_d(2 * (BA_NCAMPARAMS + 3 + 1)) {}

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    compute_ba_J(_input.n, _input.m, _input.p, _input.cams.data(),
                 _input.X.data(), _input.w.data(), _input.obs.data(),
                 _input.feats.data(), _reproj_err.data(), _w_err.data(),
                 &output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ba::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
