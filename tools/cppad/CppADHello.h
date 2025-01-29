#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"
#include <cppad/cppad.hpp>

class CppADHello : public ITest<HelloInput, HelloOutput> {
private:
  CppAD::ADFun<double> *_tape;

public:
  CppADHello(HelloInput&);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
