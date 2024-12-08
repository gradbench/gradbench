// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include <vector>

class EnzymeGMM : public ITest<GMMInput, GMMOutput> {
    GMMInput _input;
    GMMOutput _output;

public:
    // This function must be called before any other function.
    void prepare(GMMInput&& input) override;

    void calculate_objective(int times) override;
    void calculate_jacobian(int times) override;
    GMMOutput output() override;

    ~EnzymeGMM() = default;
};
