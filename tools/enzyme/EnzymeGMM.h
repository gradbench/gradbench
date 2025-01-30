#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class EnzymeGMM : public ITest<GMMInput, GMMOutput> {
 public:
  EnzymeGMM(GMMInput& input);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
