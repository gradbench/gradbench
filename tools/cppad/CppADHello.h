#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"
#include <cppad/cppad.hpp>
#include <vector>

class CppADHello : public ITest<HelloInput, HelloOutput> {
private:
  HelloInput _input;
  HelloOutput _output;
  CppAD::ADFun<double> *_tape;

public:
  virtual void prepare(HelloInput&& input) override;

  virtual void calculate_objective(int times) override;
  virtual void calculate_jacobian(int times) override;
  virtual HelloOutput output() override;

  ~CppADHello() = default;
};
