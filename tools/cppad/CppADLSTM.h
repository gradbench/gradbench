#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"
#include <cppad/cppad.hpp>

class CppADLSTM : public ITest<LSTMInput, LSTMOutput> {
  std::vector<double> _input_flat;
  CppAD::ADFun<double> *_tape;

public:
  CppADLSTM(LSTMInput&);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
