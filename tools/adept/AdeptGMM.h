#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class AdeptGMM : public ITest<GMMInput, GMMOutput> {
public:
  AdeptGMM(GMMInput&);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
