// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteHand.cpp

// Changes made: fixed trivial warnings.

#include "FiniteHT.h"

#include "adbench/shared/ht_light_matrix.h"
#include "finite.h"

FiniteHand::FiniteHand(HandInput& input) : ITest(input) {
  complicated = _input.us.size() != 0;
  int err_size = 3 * _input.data.correspondences.size();
  int ncols = (complicated ? 2 : 0) + _input.theta.size();
  _output = { std::vector<double>(err_size), ncols, err_size, std::vector<double>(err_size * ncols) };
  engine.set_max_output_size(err_size);
  if (complicated)
    jacobian_by_us = std::vector<double>(2 * err_size);
}

void FiniteHand::calculate_objective() {
  if (complicated) {
    hand_objective(_input.theta.data(), _input.us.data(), &_input.data, _output.objective.data());
  }
  else {
    hand_objective(_input.theta.data(), &_input.data, _output.objective.data());
  }
}

void FiniteHand::calculate_jacobian() {
  if (complicated) {
    engine.finite_differences([&](double* theta_in, double* err) {
      hand_objective(theta_in, _input.us.data(), &_input.data, err);
    }, _input.theta.data(), _input.theta.size(), _output.objective.size(), &_output.jacobian.data()[6 * _input.data.correspondences.size()]);

    for (unsigned int j = 0; j < _input.us.size() / 2; ++j) {
      engine.finite_differences([&](double* us_in, double* err) {
        // us_in points into the middle of __input.us.data()
        hand_objective(_input.theta.data(), _input.us.data(), &_input.data, err);
      }, &_input.us.data()[j * 2], 2, _output.objective.size(), jacobian_by_us.data());

      for (int k = 0; k < 3; ++k) {
        _output.jacobian[j * 3 + k] = jacobian_by_us[j * 3 + k];
        _output.jacobian[j * 3 + k + _output.objective.size()] = jacobian_by_us[j * 3 + k + _output.objective.size()];
      }
    }
  } else {
    engine.finite_differences([&](double* theta_in, double* err) {
      hand_objective(theta_in, &_input.data, err);
    }, _input.theta.data(), _input.theta.size(), _output.objective.size(), _output.jacobian.data());
  }
}
