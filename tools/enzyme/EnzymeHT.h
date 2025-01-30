#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"

class EnzymeHand : public ITest<HandInput, HandOutput> {
  bool _complicated = false;

public:
  EnzymeHand(HandInput& input);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
