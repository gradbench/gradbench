// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/finite/FiniteLSTM.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"
#include "finite.h"

#include <vector>

class FiniteLSTM : public ITest<LSTMInput, LSTMOutput> {
private:
    FiniteDifferencesEngine<double> engine;

public:
    FiniteLSTM(LSTMInput&);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};
