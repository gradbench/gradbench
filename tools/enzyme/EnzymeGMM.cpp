// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "EnzymeGMM.h"
#include "adbench/shared/gmm.h"

#include <iostream>

// This function must be called before any other function.
void EnzymeGMM::prepare(GMMInput&& input)
{
    _input = input;
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    _output = { 0,  std::vector<double>(Jcols) };
}

GMMOutput EnzymeGMM::output()
{
    return _output;
}

void EnzymeGMM::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
        gmm_objective(_input.d, _input.k, _input.n, _input.alphas.data(), _input.means.data(),
                      _input.icf.data(), _input.x.data(), _input.wishart, &_output.objective);
    }
}

extern int enzyme_dup;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;

void EnzymeGMM::calculate_jacobian(int times)
{
  _output.gradient.resize(_input.alphas.size()+
                          _input.means.size()+
                          _input.icf.size());


  double* d_alphas = _output.gradient.data();
  double* d_means = d_alphas+_input.alphas.size();
  double* d_icf = d_means+_input.means.size();

  for (int i = 0; i < times; ++i) {
    __enzyme_autodiff
      (gmm_objective<double>,
       enzyme_const, _input.d,
       enzyme_const, _input.k,
       enzyme_const, _input.n,

       enzyme_dup, _input.alphas.data(), d_alphas,

       enzyme_dup, _input.means.data(), d_means,

       enzyme_dup, _input.icf.data(), d_icf,

       enzyme_const, _input.x.data(),
       enzyme_const, _input.wishart,
       enzyme_const, &_output.objective);
  }

}