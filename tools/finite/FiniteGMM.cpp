// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteGMM.cpp

#include "FiniteGMM.h"
#include "adbench/shared/gmm.h"
#include "finite.h"

FiniteGMM::FiniteGMM(GMMInput& input) : ITest(input) {
  int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };
  engine.set_max_output_size(1);
}

void FiniteGMM::calculate_objective() {
  gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective);
}

void FiniteGMM::calculate_jacobian() {
  engine.finite_differences([&](double* alphas_in, double* err) {
    gmm_objective(_input.d, _input.k, _input.n, alphas_in, _input.means.data(), _input.icf.data(), _input.x.data(), _input.wishart, err);
  }, _input.alphas.data(), _input.alphas.size(), 1, _output.gradient.data());

  engine.finite_differences([&](double* means_in, double* err) {
    gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), means_in, _input.icf.data(), _input.x.data(), _input.wishart, err);
  }, _input.means.data(), _input.means.size(), 1, &_output.gradient.data()[_input.k]);

  engine.finite_differences([&](double* icf_in, double* err) {
    gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(), icf_in, _input.x.data(), _input.wishart, err);
  }, _input.icf.data(), _input.icf.size(), 1, &_output.gradient.data()[_input.k + _input.d * _input.k]);
}
