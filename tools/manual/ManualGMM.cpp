// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualGMM.cpp

#include "ManualGMM.h"
#include "adbench/shared/gmm.h"
#include "gmm_d.h"

#include <iostream>

ManualGMM::ManualGMM(GMMInput& input) : ITest(input) {
  int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };
}

void ManualGMM::calculate_objective() {
  gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective);
}

void ManualGMM::calculate_jacobian() {
  gmm_objective_d(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                  _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective, _output.gradient.data());
}
