#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"

#include <vector>

class EnzymeHello : public ITest<HelloInput, HelloOutput> {
private:
    HelloInput _input;
    HelloOutput _output;

public:
    EnzymeHello(HelloInput& input);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};
