#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"
#include <cppad/cppad.hpp>

class CppADGMM : public ITest<GMMInput, GMMOutput> {
  GMMInput _input;
  GMMOutput _output;

  std::vector<double> _input_flat;
  CppAD::ADFun<double> *_tape;

public:
  void prepare(GMMInput&& input) override;

  void calculate_objective(int times) override;
  void calculate_jacobian(int times) override;
  GMMOutput output() override;

  ~CppADGMM() = default;
};
