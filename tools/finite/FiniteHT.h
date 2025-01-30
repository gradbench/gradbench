// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteHand.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"
#include "finite.h"

class FiniteHand : public ITest<HandInput, HandOutput> {
    bool complicated = false;
    std::vector<double> jacobian_by_us;
    FiniteDifferencesEngine<double> engine;

public:
    FiniteHand(HandInput& input);

    void calculate_objective() override;
    void calculate_jacobian() override;
};
