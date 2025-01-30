#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"

class AdolCHand : public ITest<HandInput, HandOutput> {
  bool _complicated = false;

public:
  AdolCHand(HandInput&);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
