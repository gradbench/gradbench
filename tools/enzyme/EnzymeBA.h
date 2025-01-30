#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"

#include <vector>

class EnzymeBA : public ITest<BAInput, BAOutput> {
private:
  std::vector<double> _reproj_err_d;

public:
  EnzymeBA(BAInput& input);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
