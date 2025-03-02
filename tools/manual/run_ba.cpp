// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualBA.cpp

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/ba.hpp"
#include "ba_d.hpp"

class Jacobian : public Function<ba::Input, ba::JacOutput> {
private:
  std::vector<double> _reproj_err;
  std::vector<double> _w_err;
  std::vector<double> _reproj_err_d;

public:
  Jacobian(ba::Input& input) :
    Function(input),
    _reproj_err(2 * input.p),
    _w_err(input.p),
    _reproj_err_d(2 * (BA_NCAMPARAMS + 3 + 1))
  {}

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    for (int i = 0; i < _input.p; i++) {
      std::fill(_reproj_err_d.begin(), _reproj_err_d.end(), (double)0);

      int camIdx = _input.obs[2 * i + 0];
      int ptIdx = _input.obs[2 * i + 1];
      compute_reproj_error_d(
                             &_input.cams[BA_NCAMPARAMS * camIdx],
                             &_input.X[ptIdx * 3],
                             _input.w[i],
                             _input.feats[2 * i + 0], _input.feats[2 * i + 1],
                             &_reproj_err[2 * i],
                             _reproj_err_d.data());

      output.insert_reproj_err_block(i, camIdx, ptIdx, _reproj_err_d.data());
    }

    for (int i = 0; i < _input.p; i++) {
      double w_d = 0;
      compute_zach_weight_error_d(_input.w[i], &_w_err[i], &w_d);
      output.insert_w_err_block(i, w_d);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ba::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
