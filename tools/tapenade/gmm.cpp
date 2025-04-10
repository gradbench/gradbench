// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeGMM.cpp

#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/gmm.hpp"

#include "evals/gmm/gmm_b.h"

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    double* alphas_gradient_part = output.data();
    double* means_gradient_part = output.data() + _input.alphas.size();
    double* icf_gradient_part =
      output.data() +
      _input.alphas.size() +
      _input.means.size();

    double tmp = 0.0;       // stores fictive output
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
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<gmm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
