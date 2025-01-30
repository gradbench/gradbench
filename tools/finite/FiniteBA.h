// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteBA.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"
#include "finite.h"

#include <vector>

class FiniteBA : public ITest<BAInput, BAOutput> {
private:
  std::vector<double> reproj_err_d;
  FiniteDifferencesEngine<double> engine;

public:
  FiniteBA(BAInput&);

  virtual void calculate_objective() override;
  virtual void calculate_jacobian() override;
};
