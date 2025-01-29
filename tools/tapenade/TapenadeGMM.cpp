// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeGMM.cpp

#include "TapenadeGMM.h"

TapenadeGMM::TapenadeGMM(GMMInput& input) : ITest(input) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    _output = { 0, std::vector<double>(Jcols) };
}

void TapenadeGMM::calculate_objective() {
  gmm_objective(
                _input.d,
                _input.k,
                _input.n,
                _input.alphas.data(),
                _input.means.data(),
                _input.icf.data(),
                _input.x.data(),
                _input.wishart,
                &_output.objective
                );
}

void TapenadeGMM::calculate_jacobian() {
  double* alphas_gradient_part = _output.gradient.data();
  double* means_gradient_part = _output.gradient.data() + _input.alphas.size();
  double* icf_gradient_part =
    _output.gradient.data() +
    _input.alphas.size() +
    _input.means.size();

  double tmp = 0.0;       // stores fictive _output
  // (Tapenade doesn't calculate an original function in reverse mode)

  double errb = 1.0;      // stores dY
  // (equals to 1.0 for gradient calculation)

  gmm_objective_b(_input.d,
                  _input.k,
                  _input.n,
                  _input.alphas.data(),
                  alphas_gradient_part,
                  _input.means.data(),
                  means_gradient_part,
                  _input.icf.data(),
                  icf_gradient_part,
                  _input.x.data(),
                  _input.wishart,
                  &tmp,
                  &errb
                  );
}
