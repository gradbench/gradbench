// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/ADOLC/main.cpp
//
// In particular, this is the "DO_BA_BLOCK" version. The
// "DO_BA_SPARSE" seems to use facilities no longer available in
// ADOL-C.
//
// With this implementation it is not possible to separate out the
// tape construction in the constructor, as it is too interleaved with
// the actual computation.

#include "gradbench/evals/ba.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include <set>
#include <string.h>
#include <vector>

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>
#include <adolc/taping.h>

void compute_reproj_error_J_block(int tapeTag, double* cam, double* X, double w,
                                  double* feat, double* err, double* J) {
  adouble acam[BA_NCAMPARAMS], aX[3], aw, areproj_err[2], aw_err;

  int keepValues = 1;
  trace_on(tapeTag, keepValues);
  for (int i = 0; i < BA_NCAMPARAMS; i++)
    acam[i] <<= cam[i];
  for (int i = 0; i < 3; i++)
    aX[i] <<= X[i];
  aw <<= w;
  ba::computeReprojError(acam, aX, &aw, feat, areproj_err);
  areproj_err[0] >>= err[0];
  areproj_err[1] >>= err[1];
  trace_off();

  int      n_new_cols = BA_NCAMPARAMS + 3 + 1;
  double** eye;
  eye           = new double*[2];
  double** Jtmp = new double*[2];
  for (int i = 0; i < 2; i++) {
    eye[i]  = new double[2];
    Jtmp[i] = new double[n_new_cols];
  }

  eye[0][0] = 1;
  eye[0][1] = 0;
  eye[1][0] = 0;
  eye[1][1] = 1;
  fov_reverse(tapeTag, 2, n_new_cols, 2, eye, Jtmp);

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < n_new_cols; j++)
      J[j * 2 + i] = Jtmp[i][j];

  for (int i = 0; i < 2; i++) {
    delete[] eye[i];
    delete[] Jtmp[i];
  }
  delete[] eye;
  delete[] Jtmp;
}

void compute_ba_J(int n, int m, int p, double* cams, double* X, double* w,
                  int* obs, double* feats, double* reproj_err, double* w_err,
                  ba::SparseMat* J) {
  const int tapeTagReprojErr = 1;
  const int tapeTagWeightErr = 2;
  *J                         = ba::SparseMat(n, m, p);

  int                 n_new_cols = BA_NCAMPARAMS + 3 + 1;
  std::vector<double> reproj_err_d(2 * n_new_cols);
  for (int i = 0; i < p; i++) {
    std::fill(reproj_err_d.begin(), reproj_err_d.end(), (double)0);

    int camIdx = obs[2 * i + 0];
    int ptIdx  = obs[2 * i + 1];
    compute_reproj_error_J_block(
        tapeTagReprojErr, &cams[BA_NCAMPARAMS * camIdx], &X[ptIdx * 3], w[i],
        &feats[2 * i], &reproj_err[2 * i], reproj_err_d.data());

    J->insert_reproj_err_block(i, camIdx, ptIdx, reproj_err_d.data());
  }

  adouble aw, aw_err;
  trace_on(tapeTagWeightErr);
  aw <<= w[0];
  ba::computeZachWeightError(&aw, &aw_err);
  aw_err >>= w_err[0];
  trace_off();
  int keepValues = 0;

  for (int i = 0; i < p; i++) {
    double err_d = 0.;
    double w_d   = 1.;
    fos_forward(tapeTagWeightErr, 1, 1, keepValues, &w[i], &w_d, &w_err[i],
                &err_d);

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
