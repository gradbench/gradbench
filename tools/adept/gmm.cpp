// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Heavily based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/Adept/main.cpp

#include "gradbench/evals/gmm.hpp"
#include "adept.h"
#include "gradbench/main.hpp"
#include <algorithm>
#include <vector>

using adept::adouble;

void compute_gmm_J(const gmm::Input& input, gmm::JacOutput& output) {
  adept::Stack stack;
  adouble*     aalpha = new adouble[input.alpha.size()];
  adouble*     amu    = new adouble[input.mu.size()];
  adouble*     aq     = new adouble[input.q.size()];
  adouble*     al     = new adouble[input.l.size()];

  adept::set_values(aalpha, input.alpha.size(), input.alpha.data());
  adept::set_values(amu, input.mu.size(), input.mu.data());
  adept::set_values(aq, input.q.size(), input.q.data());
  adept::set_values(al, input.l.size(), input.l.data());

  stack.new_recording();
  adouble aerr;
  gmm::objective<adouble>(input.d, input.k, input.n, aalpha, amu, aq, al,
                          input.x.data(), input.wishart, &aerr);
  aerr.set_gradient(1.);  // only one J row here
  stack.reverse();

  adept::get_gradients(aalpha, output.alpha.size(), output.alpha.data());
  adept::get_gradients(amu, output.mu.size(), output.mu.data());
  adept::get_gradients(aq, output.q.size(), output.q.data());
  adept::get_gradients(al, output.l.size(), output.l.data());

  delete[] aalpha;
  delete[] amu;
  delete[] aq;
  delete[] al;
}

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
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

    compute_gmm_J(_input, output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
