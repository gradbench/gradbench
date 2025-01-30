#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"

class AdeptHand : public ITest<HandInput, HandOutput> {
  bool _complicated = false;

public:
  AdeptHand(HandInput&);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
