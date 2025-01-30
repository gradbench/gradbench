#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"

class AdolCLSTM : public ITest<LSTMInput, LSTMOutput> {
public:
  AdolCLSTM(LSTMInput& input);

  void prepare_jacobian() override;
  void calculate_objective() override;
  void calculate_jacobian() override;
};
