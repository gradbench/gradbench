// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/manual/ManualBA.h

// ManualBA.h - Contains declarations of GMM tester functions
#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"

#include <vector>

class ManualBA : public ITest<BAInput, BAOutput> {
private:
    std::vector<double> _reproj_err_d;

public:
    ManualBA(BAInput&);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};
