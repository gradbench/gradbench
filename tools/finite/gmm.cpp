// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Largely derived from
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteGMM.cpp

#include "gradbench/evals/gmm.hpp"
#include "finite.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  FiniteDifferencesEngine<double> _engine;

public:
  Jacobian(gmm::Input& input) : Function(input) {
    _engine.set_max_output_size(1);
  }

  void compute(gmm::JacOutput& output) {
    const int l_sz = _input.d * (_input.d - 1) / 2;

    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    output.alpha.resize(output.k);
    output.mu.resize(output.k * output.d);
    output.q.resize(output.k * output.d);
    output.l.resize(output.k * l_sz);

    _engine.finite_differences(
        1,
        [&](double* alpha_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, alpha_in,
                         _input.mu.data(), _input.q.data(), _input.l.data(),
                         _input.x.data(), _input.wishart, err);
        },
        _input.alpha.data(), _input.alpha.size(), 1, output.alpha.data());

    _engine.finite_differences(
        1,
        [&](double* mu_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, _input.alpha.data(),
                         mu_in, _input.q.data(), _input.l.data(),
                         _input.x.data(), _input.wishart, err);
        },
        _input.mu.data(), _input.mu.size(), 1, output.mu.data());

    _engine.finite_differences(
        1,
        [&](double* q_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, _input.alpha.data(),
                         _input.mu.data(), q_in, _input.l.data(),
                         _input.x.data(), _input.wishart, err);
        },
        _input.q.data(), _input.q.size(), 1, output.q.data());
    _engine.finite_differences(
        1,
        [&](double* l_in, double* err) {
          gmm::objective(_input.d, _input.k, _input.n, _input.alpha.data(),
                         _input.mu.data(), _input.q.data(), l_in,
                         _input.x.data(), _input.wishart, err);
        },
        _input.l.data(), _input.l.size(), 1, output.l.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
