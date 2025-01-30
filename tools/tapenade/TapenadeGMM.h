// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeGMM.h

// Changes made:
//
// * modified #include paths.

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/GMMData.h"

#include "gmm/gmm.h"
#include "gmm/gmm_b.h"

#include <vector>

class TapenadeGMM : public ITest<GMMInput, GMMOutput>
{
private:
    std::vector<double> state;

public:
    TapenadeGMM(GMMInput&);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};

