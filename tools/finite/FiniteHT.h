// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteHand.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"
#include "finite.h"

class FiniteHand : public ITest<HandInput, HandOutput> {
    HandInput input;
    HandOutput result;
    bool complicated = false;
    std::vector<double> jacobian_by_us;
    FiniteDifferencesEngine<double> engine;

public:
    // This function must be called before any other function.
    void prepare(HandInput&& input) override;

    void calculate_objective(int times) override;
    void calculate_jacobian(int times) override;
    HandOutput output() override;

    ~FiniteHand() = default;
};
