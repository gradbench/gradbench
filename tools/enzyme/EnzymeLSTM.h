#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"

#include <vector>

class EnzymeLSTM : public ITest<LSTMInput, LSTMOutput> {
public:
  EnzymeLSTM(LSTMInput& input);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
