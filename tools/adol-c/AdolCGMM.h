#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class AdolCGMM : public ITest<GMMInput, GMMOutput> {
public:
  AdolCGMM(GMMInput&);

  void prepare_jacobian() override;
  void calculate_objective() override;
  void calculate_jacobian() override;
};
