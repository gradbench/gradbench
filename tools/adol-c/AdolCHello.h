#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"

#include <vector>

class AdolCHello : public ITest<HelloInput, HelloOutput> {
public:
  AdolCHello(HelloInput& input);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
