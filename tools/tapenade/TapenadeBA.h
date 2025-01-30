// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeBA.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/BAData.h"
#include "adbench/shared/defs.h"

#include "ba/ba.h"
#include "ba/ba_b.h"

#include <vector>

class TapenadeBA : public ITest<BAInput, BAOutput>
{
private:
    std::vector<double> state;

    // buffer for reprojection error jacobian part holding (column-major)
    std::vector<double> reproj_err_d;

    // buffer for reprojection error jacobian block row holding
    std::vector<double> reproj_err_d_row;

public:
    TapenadeBA(BAInput& input);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;

private:
    void calculate_weight_error_jacobian_part();
    void calculate_reproj_error_jacobian_part();
};
