// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeGMM.cpp

#include "gradbench/evals/gmm.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

#include "evals/gmm/gmm_b.h"

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  std::vector<double> _J;

public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    const int l_sz = _input.d * (_input.d - 1) / 2;

    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    output.alpha.resize(output.k);
    output.mu.resize(output.k * output.d);
    output.q.resize(output.k * output.d);
    output.l.resize(output.k * l_sz);

    // Because the Tapenade program is difficult to modify, we use the
    // ADBench-like approach of merging the Q and L arrays into a single icf
    // array.
    int k = _input.k, d = _input.d;
    int icf_sz = d * (d + 1) / 2;

    std::vector<double> icf(k * icf_sz);
    std::vector<double> icf_d(k * icf_sz);

    double* alpha_gradient_part = output.alpha.data();
    double* mu_gradient_part  = output.mu.data();
    double* icf_gradient_part = icf_d.data();

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < d; j++) {
        icf[i * icf_sz + j] = _input.q[i*d+j];
      }
      for (int j = d; j < icf_sz; j++) {
        icf[i * icf_sz + j] = _input.l[i*l_sz+j-d];
      }
    }

    double tmp = 0.0;  // stores fictive output
    // (Tapenade doesn't calculate an original function in reverse mode)

    double errb = 1.0;  // stores dY
    // (equals to 1.0 for gradient calculation)

    gmm_objective_b(d, k, _input.n, _input.alpha.data(),
                    alpha_gradient_part, _input.mu.data(),
                    mu_gradient_part, icf.data(), icf_gradient_part,
                    _input.x.data(), _input.wishart, &tmp, &errb);

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < d; j++) {
        output.q[i*d+j] = icf_d[i * icf_sz + j];
      }
      for (int j = d; j < icf_sz; j++) {
        output.l[i*l_sz+j-d] = icf_d[i * icf_sz + j];
      }
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
