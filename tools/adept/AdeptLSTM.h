// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"

#include <vector>

class AdeptLSTM : public ITest<LSTMInput, LSTMOutput> {
private:
  LSTMInput input;
  LSTMOutput result;
  std::vector<double> state;

public:
  // This function must be called before any other function.
  virtual void prepare(LSTMInput&& input) override;

  virtual void calculate_objective(int times) override;
  virtual void calculate_jacobian(int times) override;
  virtual LSTMOutput output() override;

  ~AdeptLSTM() {}
};