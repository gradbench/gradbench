// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteGMM.cpp

#include "gradbench/evals/gmm.hpp"
#include "finite.h"
#include "gradbench/main.hpp"
#include <algorithm>

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  FiniteDifferencesEngine<double> _engine;

public:
  Jacobian(gmm::Input& input) : Function(input) {
    _engine.set_max_output_size(1);
  }

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    _engine.finite_differences(
        1,
        [&](double* alphas_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, alphas_in,
                         _input.means.data(), _input.icf.data(),
                         _input.x.data(), _input.wishart, err);
        },
        _input.alphas.data(), _input.alphas.size(), 1, output.data());

    _engine.finite_differences(
        1,
        [&](double* means_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, _input.alphas.data(),
                         means_in, _input.icf.data(), _input.x.data(),
                         _input.wishart, err);
        },
        _input.means.data(), _input.means.size(), 1, &output.data()[_input.k]);

    _engine.finite_differences(
        1,
        [&](double* icf_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, _input.alphas.data(),
                         _input.means.data(), icf_in, _input.x.data(),
                         _input.wishart, err);
        },
        _input.icf.data(), _input.icf.size(), 1,
        &output.data()[_input.k + _input.d * _input.k]);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
