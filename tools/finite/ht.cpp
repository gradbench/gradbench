// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteHand.cpp

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/ht.hpp"
#include "finite.h"

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  bool _complicated = false;
  std::vector<double> _objective;
  std::vector<double> _jacobian_by_us;
  FiniteDifferencesEngine<double> _engine;
public:
  Jacobian(ht::Input& input) :
    Function(input),
    _complicated(input.us.size() != 0),
    _objective(3 * input.data.correspondences.size()),
    _jacobian_by_us(2 * _objective.size())
  {
    _engine.set_max_output_size(_objective.size());
  }

  void compute(ht::JacOutput& output) {
    int err_size = 3 * _input.data.correspondences.size();
    int ncols = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (_complicated) {
      _engine.finite_differences(1, [&](double* theta_in, double* err) {
        ht::objective(theta_in, _input.us.data(), &_input.data, err);
      }, _input.theta.data(), _input.theta.size(),
        _objective.size(),
        &output.jacobian.data()[6 * _input.data.correspondences.size()]);

      for (unsigned int j = 0; j < _input.us.size() / 2; ++j) {
        _engine.finite_differences(1, [&](double* us_in, double* err) {
          ht::objective(_input.theta.data(), us_in, &_input.data, err);
        }, _input.us.data(), _input.us.size(), j * 2, 2,
          _objective.size(),
          _jacobian_by_us.data());

        for (int k = 0; k < 3; ++k) {
          output.jacobian[j * 3 + k] = _jacobian_by_us[j * 3 + k];
          output.jacobian[j * 3 + k + _objective.size()] = _jacobian_by_us[j * 3 + k + _objective.size()];
        }
      }
    } else {
      _engine.finite_differences(1, [&](double* theta_in, double* err) {
        ht::objective(theta_in, &_input.data, err);
      }, _input.theta.data(), _input.theta.size(), _objective.size(), output.jacobian.data());
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ht::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
