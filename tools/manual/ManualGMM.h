// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualGMM.h

// ManualGMM.h - Contains declarations of GMM tester functions
#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class ManualGMM : public ITest<GMMInput, GMMOutput> {
public:
    ManualGMM(GMMInput& input);

    void calculate_objective() override;
    void calculate_jacobian() override;
};
