// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"

class AdolCHand : public ITest<HandInput, HandOutput> {
  HandInput _input;
  HandOutput _output;
  bool _complicated = false;

public:
  // This function must be called before any other function.
  void prepare(HandInput&& input) override;

  void calculate_objective(int times) override;
  void calculate_jacobian(int times) override;
  HandOutput output() override;

  ~AdolCHand() = default;
};
