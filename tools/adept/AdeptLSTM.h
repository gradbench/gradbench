#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"

#include <vector>

class AdeptLSTM : public ITest<LSTMInput, LSTMOutput> {
public:
  AdeptLSTM(LSTMInput&);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
