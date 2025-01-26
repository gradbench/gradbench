#include "CppADGMM.h"
#include "adbench/shared/gmm.h"
#include <iostream>

typedef CppAD::AD<double> ADdouble;

void CppADGMM::prepare(GMMInput&& input) {
  _input = input;
  int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };

  input_flat.insert(input_flat.end(), _input.alphas.begin(), _input.alphas.end());
  input_flat.insert(input_flat.end(), _input.means.begin(), _input.means.end());
  input_flat.insert(input_flat.end(), _input.icf.begin(), _input.icf.end());

  std::vector<ADdouble> X(_input.alphas.size() +
                          _input.means.size() +
                          _input.icf.size());

  ADdouble* aalphas = &X[0];
  ADdouble* ameans = aalphas + _input.alphas.size();
  ADdouble* aicf = ameans + _input.means.size();

  std::copy(_input.alphas.begin(), _input.alphas.end(), aalphas);
  std::copy(_input.means.begin(), _input.means.end(), ameans);
  std::copy(_input.icf.begin(), _input.icf.end(), aicf);

  CppAD::Independent(X);

  std::vector<ADdouble> Y(1);

  gmm_objective<ADdouble>(_input.d, _input.k, _input.n,
                          aalphas, ameans, aicf,
                          _input.x.data(), _input.wishart, &Y[0]);

  _tape = new CppAD::ADFun<double>(X, Y);
}

GMMOutput CppADGMM::output() {
  return _output;
}

void CppADGMM::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    gmm_objective(_input.d, _input.k, _input.n,
                  _input.alphas.data(), _input.means.data(), _input.icf.data(),
                  _input.x.data(), _input.wishart, &_output.objective);
  }
}

void CppADGMM::calculate_jacobian(int times) {
  for (int i = 0; i < times; ++i) {
    _output.gradient = _tape->Jacobian(input_flat);
  }
}
