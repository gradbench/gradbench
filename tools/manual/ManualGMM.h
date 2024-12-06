// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualGMM.h

// ManualGMM.h - Contains declarations of GMM tester functions
#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class ManualGMM : public ITest<GMMInput, GMMOutput> {
    GMMInput _input;
    GMMOutput _output;

public:
    // This function must be called before any other function.
    void prepare(GMMInput&& input) override;

    void calculate_objective(int times) override;
    void calculate_jacobian(int times) override;
    GMMOutput output() override; 

    ~ManualGMM() = default;
};
