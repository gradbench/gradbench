// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualHand.cpp

#include "ManualHT.h"

#include "adbench/shared/ht_light_matrix.h"
#include "ht_d.h"

ManualHand::ManualHand(HandInput& input) : ITest(input) {
  _complicated = _input.us.size() != 0;
  int err_size = 3 * _input.data.correspondences.size();
  int ncols = (_complicated ? 2 : 0) + _input.theta.size();
  _output = { std::vector<double>(err_size), ncols, err_size, std::vector<double>(err_size * ncols) };
}

void ManualHand::calculate_objective() {
  if (_complicated) {
    hand_objective(_input.theta.data(), _input.us.data(), &_input.data, _output.objective.data());
  } else {
    hand_objective(_input.theta.data(), &_input.data, _output.objective.data());
  }
}

void ManualHand::calculate_jacobian() {
  if (_complicated) {
    hand_objective_d(_input.theta.data(), _input.us.data(), _input.data, _output.objective.data(), _output.jacobian.data());
  }
  else {
    hand_objective_d(_input.theta.data(), _input.data, _output.objective.data(), _output.jacobian.data());
  }
}
