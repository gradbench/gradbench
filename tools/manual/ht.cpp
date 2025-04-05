// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualHand.cpp

#include "gradbench/main.hpp"
#include "gradbench/evals/ht.hpp"

#include "ht_d.hpp"

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  bool _complicated = false;
  std::vector<double> _objective;
public:
  Jacobian(ht::Input& input) :
    Function(input),
    _complicated(input.us.size() != 0),
    _objective(3 * input.data.correspondences.size())
  {}

  void compute(ht::JacOutput& output) {
    int err_size = 3 * _input.data.correspondences.size();
    int ncols = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (_complicated) {
      ht_objective_d(_input.theta.data(), _input.us.data(), _input.data, _objective.data(), output.jacobian.data());
    } else {
      ht_objective_d(_input.theta.data(), _input.data, _objective.data(), output.jacobian.data());
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ht::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
