// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualGMM.cpp

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/gmm.hpp"
#include "gmm_d.hpp"

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    double error;
    gmm_objective_d(_input.d, _input.k, _input.n,
                    _input.alphas.data(), _input.means.data(),
                    _input.icf.data(), _input.x.data(), _input.wishart,
                    &error, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<gmm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
