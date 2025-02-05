#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/KMeansData.h"

#include <vector>

class ManualKMeans : public ITest<KMeansInput, KMeansOutput> {
public:
  ManualKMeans(KMeansInput& input);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
