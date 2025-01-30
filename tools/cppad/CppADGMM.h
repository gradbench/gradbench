#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"
#include <cppad/cppad.hpp>

class CppADGMM : public ITest<GMMInput, GMMOutput> {
  std::vector<double> _input_flat;
  CppAD::ADFun<double> *_tape;

public:
  CppADGMM(GMMInput&);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
