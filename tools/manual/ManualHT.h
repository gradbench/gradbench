// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualHand.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"

class ManualHand : public ITest<HandInput, HandOutput> {
  bool _complicated = false;

public:
  ManualHand(HandInput& input);

  void calculate_objective() override;
  void calculate_jacobian() override;
};
