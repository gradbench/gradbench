// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteBA.cpp

#include "gradbench/evals/ba.hpp"
#include "finite.h"
#include "gradbench/main.hpp"
#include <algorithm>

static int n_new_cols = BA_NCAMPARAMS + 3 + 1;

class Jacobian : public Function<ba::Input, ba::JacOutput> {
private:
  std::vector<double>             _reproj_err_d;
  FiniteDifferencesEngine<double> _engine;

public:
  Jacobian(ba::Input& input) : Function(input), _reproj_err_d(2 * n_new_cols) {
    _engine.set_max_output_size(2);
  }

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    for (int j = 0; j < _input.p; ++j) {
      std::fill(_reproj_err_d.begin(), _reproj_err_d.end(), (double)0);

      int camIdx = _input.obs[2 * j + 0];
      int ptIdx  = _input.obs[2 * j + 1];

      _engine.finite_differences(
          1,
          [&](double* cam_in, double* reproj_err) {
            ba::computeReprojError(cam_in, &_input.X[ptIdx * 3], &_input.w[j],
                                   &_input.feats[2 * j], reproj_err);
          },
          &_input.cams[camIdx * BA_NCAMPARAMS], BA_NCAMPARAMS, 2,
          _reproj_err_d.data());

      _engine.finite_differences(
          1,
          [&](double* X_in, double* reproj_err) {
            ba::computeReprojError(&_input.cams[camIdx * BA_NCAMPARAMS], X_in,
                                   &_input.w[j], &_input.feats[2 * j],
                                   reproj_err);
          },
          &_input.X[ptIdx * 3], 3, 2, &_reproj_err_d.data()[2 * BA_NCAMPARAMS]);

      _engine.finite_differences(
          1,
          [&](double* w_in, double* reproj_err) {
            ba::computeReprojError(&_input.cams[camIdx * BA_NCAMPARAMS],
                                   &_input.X[ptIdx * 3], w_in,
                                   &_input.feats[2 * j], reproj_err);
          },
          &_input.w[j], 1, 2, &_reproj_err_d.data()[2 * (BA_NCAMPARAMS + 3)]);

      output.insert_reproj_err_block(j, camIdx, ptIdx, _reproj_err_d.data());
    }

    double w_d;

    for (int j = 0; j < _input.p; ++j) {
      _engine.finite_differences(
          1,
          [&](double* w_in, double* w_er) {
            ba::computeZachWeightError(w_in, w_er);
          },
          &_input.w[j], 1, 1, &w_d);

      output.insert_w_err_block(j, w_d);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ba::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
