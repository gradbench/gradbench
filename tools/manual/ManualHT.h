// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualHand.h

// ManualHand.h - Contains declarations of GMM tester functions
#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HTData.h"

class ManualHand : public ITest<HandInput, HandOutput> {
    HandInput _input;
    HandOutput _output;
    bool _complicated = false;

public:
    // This function must be called before any other function.
    void prepare(HandInput&& input) override;

    void calculate_objective(int times) override;
    void calculate_jacobian(int times) override;
    HandOutput output() override;

    ~ManualHand() = default;
};