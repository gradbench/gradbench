#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"

#include <vector>

class EnzymeBA : public ITest<BAInput, BAOutput> {
private:
  BAInput _input;
  BAOutput _output;
  std::vector<double> _reproj_err_d;

public:
  virtual void prepare(BAInput&& input) override;

  virtual void calculate_objective(int times) override;
  virtual void calculate_jacobian(int times) override;
  virtual BAOutput output() override;

  ~EnzymeBA() {}
};
