#include "CppADLSTM.h"
#include "adbench/shared/lstm.h"
#include <iostream>

typedef CppAD::AD<double> ADdouble;

void CppADLSTM::prepare(LSTMInput&& input) {
  _input = input;
  int Jcols = 8 * _input.l * _input.b + 3 * _input.b;
  _output = { 0,  std::vector<double>(Jcols) };

  _input_flat.insert(_input_flat.end(), _input.main_params.begin(), _input.main_params.end());
  _input_flat.insert(_input_flat.end(), _input.extra_params.begin(), _input.extra_params.end());

  std::vector<ADdouble> astate(_input.state.size());
  std::vector<ADdouble> asequence(_input.sequence.size());
  astate.insert(astate.begin(), _input.state.begin(), _input.state.end());
  asequence.insert(asequence.begin(), _input.sequence.begin(), _input.sequence.end());

  std::vector<ADdouble> X(_input_flat.size());
  std::copy(_input_flat.begin(), _input_flat.end(), X.data());
  ADdouble* amain_params = &X[0];
  ADdouble* aextra_params = amain_params + _input.main_params.size();

  CppAD::Independent(X);

  std::vector<ADdouble> Y(1);

  lstm_objective(input.l, input.c, input.b,
                 amain_params, aextra_params,
                 astate.data(), asequence.data(),
                 &Y[0]);

  _tape = new CppAD::ADFun<double>(X, Y);

  _tape->optimize();
}

LSTMOutput CppADLSTM::output() {
  return _output;
}

void CppADLSTM::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    lstm_objective(_input.l, _input.c, _input.b,
                   _input.main_params.data(), _input.extra_params.data(),
                   _input.state.data(), _input.sequence.data(),
                   &_output.objective);
  }
}

void CppADLSTM::calculate_jacobian(int times) {
  for (int i = 0; i < times; ++i) {
    _output.gradient = _tape->Jacobian(_input_flat);
  }
}
