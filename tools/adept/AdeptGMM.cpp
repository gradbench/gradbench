// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "AdeptGMM.h"
#include "adbench/shared/gmm.h"
#include "adept.h"

#include <iostream>

using adept::adouble;

void compute_gmm_J(const GMMInput &input, GMMOutput &output) {
  int icf_sz = input.d*(input.d + 1) / 2;

  adept::Stack stack;
  adouble *aalphas = new adouble[input.k];
  adouble *ameans = new adouble[input.d*input.k];
  adouble *aicf = new adouble[icf_sz*input.k];

  adept::set_values(aalphas, input.k, input.alphas.data());
  adept::set_values(ameans, input.d*input.k, input.means.data());
  adept::set_values(aicf, icf_sz*input.k, input.icf.data());

  stack.new_recording();
  adouble aerr;
  gmm_objective(input.d, input.k, input.n, aalphas, ameans,
                aicf, input.x.data(), input.wishart, &aerr);
  aerr.set_gradient(1.); // only one J row here
  stack.reverse();

  adept::get_gradients(aalphas, input.k, output.gradient.data());
  adept::get_gradients(ameans, input.d*input.k, &output.gradient.data()[input.k]);
  adept::get_gradients(aicf, icf_sz*input.k, &output.gradient.data()[input.k + input.d * input.k]);

  delete[] aalphas;
  delete[] ameans;
  delete[] aicf;
}

// This function must be called before any other function.
void AdeptGMM::prepare(GMMInput&& input)
{
  _input = input;
  int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };
}

GMMOutput AdeptGMM::output()
{
  return _output;
}

void AdeptGMM::calculate_objective(int times)
{
  for (int i = 0; i < times; ++i) {
    gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                  _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective);
  }
}

void AdeptGMM::calculate_jacobian(int times)
{
  for (int i = 0; i < times; ++i) {
    compute_gmm_J(_input, _output);
  }
}
