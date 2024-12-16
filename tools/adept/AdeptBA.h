// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// AdeptBA.h - Contains declarations of GMM tester functions
#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"

#include <vector>

class AdeptBA : public ITest<BAInput, BAOutput> {
private:
    BAInput _input;
    BAOutput _output;
    std::vector<double> _reproj_err_d;

public:
    // This function must be called before any other function.
    virtual void prepare(BAInput&& input) override;

    virtual void calculate_objective(int times) override;
    virtual void calculate_jacobian(int times) override;
    virtual BAOutput output() override;

    ~AdeptBA() {}
};
