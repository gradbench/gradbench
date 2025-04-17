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
  int icf_sz = input.d * (input.d + 1) / 2;

  adept::Stack stack;
  adouble*     aalphas = new adouble[input.k];
  adouble*     ameans  = new adouble[input.d * input.k];
  adouble*     aicf    = new adouble[icf_sz * input.k];

  adept::set_values(aalphas, input.k, input.alphas.data());
  adept::set_values(ameans, input.d * input.k, input.means.data());
  adept::set_values(aicf, icf_sz * input.k, input.icf.data());

  stack.new_recording();
  adouble aerr;
  gmm::objective<adouble>(input.d, input.k, input.n, aalphas, ameans,
                          aicf, input.x.data(), input.wishart, &aerr);
  aerr.set_gradient(1.); // only one J row here
  stack.reverse();

  adept::get_gradients(aalphas, input.k, output.data());
  adept::get_gradients(ameans, input.d * input.k, &output.data()[input.k]);
  adept::get_gradients(aicf, icf_sz * input.k,
                       &output.data()[input.k + input.d * input.k]);

  delete[] aalphas;
  delete[] ameans;
  delete[] aicf;
}

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    compute_gmm_J(_input, output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
