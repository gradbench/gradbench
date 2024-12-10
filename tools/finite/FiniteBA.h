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
    BAInput input;
    BAOutput result;
    std::vector<double> reproj_err_d;
    FiniteDifferencesEngine<double> engine;

public:
    // This function must be called before any other function.
    virtual void prepare(BAInput&& input) override;

    virtual void calculate_objective(int times) override;
    virtual void calculate_jacobian(int times) override;
    virtual BAOutput output() override;

    ~FiniteBA() = default;
};
