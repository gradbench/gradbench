#include "CppADGMM.h"
#include "adbench/shared/gmm.h"
#include <iostream>

typedef CppAD::AD<double> ADdouble;

CppADGMM::CppADGMM(GMMInput& input) : ITest(input) {
  int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };

  _input_flat.insert(_input_flat.end(), _input.alphas.begin(), _input.alphas.end());
  _input_flat.insert(_input_flat.end(), _input.means.begin(), _input.means.end());
  _input_flat.insert(_input_flat.end(), _input.icf.begin(), _input.icf.end());

  std::vector<ADdouble> X(_input_flat.size());

  ADdouble* aalphas = &X[0];
  ADdouble* ameans = aalphas + _input.alphas.size();
  ADdouble* aicf = ameans + _input.means.size();

  std::copy(_input_flat.begin(), _input_flat.end(), X.data());

  CppAD::Independent(X);

  std::vector<ADdouble> Y(1);

  gmm_objective<ADdouble>(_input.d, _input.k, _input.n,
                          aalphas, ameans, aicf,
                          _input.x.data(), _input.wishart, &Y[0]);

  _tape = new CppAD::ADFun<double>(X, Y);

  _tape->optimize();
}

void CppADGMM::calculate_objective() {
  gmm_objective(_input.d, _input.k, _input.n,
                _input.alphas.data(), _input.means.data(), _input.icf.data(),
                _input.x.data(), _input.wishart, &_output.objective);
}

void CppADGMM::calculate_jacobian() {
  _output.gradient = _tape->Jacobian(_input_flat);
}
