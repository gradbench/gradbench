// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeBA.cpp

#include "ba/ba.h"
#include "ba/ba_b.h"

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/ba.hpp"

class Jacobian : public Function<ba::Input, ba::JacOutput> {
private:
  std::vector<double> _reproj_err;
  std::vector<double> _w_err;

  std::vector<double> _state;

  // buffer for reprojection error jacobian part holding (column-major)
  std::vector<double> _reproj_err_d;

  // buffer for reprojection error jacobian block row holding
  std::vector<double> _reproj_err_d_row;

public:
  Jacobian(ba::Input& input) :
    Function(input),
    _reproj_err(2 * input.p),
    _w_err(input.p),
    _reproj_err_d(2 * (BA_NCAMPARAMS + 3 + 1)),
    _reproj_err_d_row(BA_NCAMPARAMS + 3 + 1)
  {}

  void calculate_reproj_error_jacobian_part(ba::JacOutput& output) {
    double errb[2];     // stores dY
    // (i-th element equals to 1.0 for calculating i-th jacobian row)

    double err[2];      // stores fictive _output
    // (Tapenade doesn't calculate an original function in reverse mode)

    double* cam_gradient_part = _reproj_err_d_row.data();
    double* x_gradient_part = _reproj_err_d_row.data() + BA_NCAMPARAMS;
    double* weight_gradient_part = _reproj_err_d_row.data() + BA_NCAMPARAMS + 3;

    for (int i = 0; i < _input.p; i++) {
      int camIdx = _input.obs[2 * i + 0];
      int ptIdx = _input.obs[2 * i + 1];

      // calculate first row
      errb[0] = 1.0;
      errb[1] = 0.0;
      compute_reproj_error_b(
                             &_input.cams[camIdx * BA_NCAMPARAMS],
                             cam_gradient_part,
                             &_input.X[ptIdx * 3],
                             x_gradient_part,
                             &_input.w[i],
                             weight_gradient_part,
                             &_input.feats[i * 2],
                             err,
                             errb
                             );

      // fill first row elements
      for (int j = 0; j < BA_NCAMPARAMS + 3 + 1; j++) {
        _reproj_err_d[2 * j] = _reproj_err_d_row[j];
      }

      // calculate second row
      errb[0] = 0.0;
      errb[1] = 1.0;
      compute_reproj_error_b(
                             &_input.cams[camIdx * BA_NCAMPARAMS],
                             cam_gradient_part,
                             &_input.X[ptIdx * 3],
                             x_gradient_part,
                             &_input.w[i],
                             weight_gradient_part,
                             &_input.feats[i * 2],
                             err,
                             errb
                             );

      // fill second row elements
      for (int j = 0; j < BA_NCAMPARAMS + 3 + 1; j++) {
        _reproj_err_d[2 * j + 1] = _reproj_err_d_row[j];
      }

      output.insert_reproj_err_block(i, camIdx, ptIdx, _reproj_err_d.data());
    }
  }

  void calculate_weight_error_jacobian_part(ba::JacOutput& output) {
    for (int j = 0; j < _input.p; j++) {
      double wb;              // stores calculated derivative

      double err = 0.0;       // stores fictive _output
      // (Tapenade doesn't calculate an original function in reverse mode)

      double errb = 1.0;      // stores dY
      // (equals to 1.0 for derivative calculation)

      compute_zach_weight_error_b(&_input.w[j], &wb, &err, &errb);
      output.insert_w_err_block(j, wb);
    }
  }

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);
    calculate_reproj_error_jacobian_part(output);
    calculate_weight_error_jacobian_part(output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ba::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
