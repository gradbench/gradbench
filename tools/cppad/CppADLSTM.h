#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"
#include <cppad/cppad.hpp>

class CppADLSTM : public ITest<LSTMInput, LSTMOutput> {
  LSTMInput _input;
  LSTMOutput _output;

  std::vector<double> _input_flat;
  CppAD::ADFun<double> *_tape;

public:
  void prepare(LSTMInput&& input) override;

  void calculate_objective(int times) override;
  void calculate_jacobian(int times) override;
  LSTMOutput output() override;

  ~CppADLSTM() = default;
};
