#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"

#include <vector>

class AdeptBA : public ITest<BAInput, BAOutput> {
private:
  std::vector<double> _reproj_err_d;

public:
  AdeptBA(BAInput&);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
