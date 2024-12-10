// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/modules/tapenade/TapenadeLSTM.h

#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/LSTMData.h"

#include "lstm/lstm.h"
#include "lstm/lstm_b.h"

#include <vector>

class TapenadeLSTM : public ITest<LSTMInput, LSTMOutput>
{
private:
    LSTMInput input;
    LSTMOutput result;
    std::vector<double> state;

public:
    // This function must be called before any other function.
    virtual void prepare(LSTMInput&& input) override;

    virtual void calculate_objective(int times) override;
    virtual void calculate_jacobian(int times) override;
    virtual LSTMOutput output() override;

    ~TapenadeLSTM() {}
};

