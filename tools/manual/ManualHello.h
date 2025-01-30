#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"

#include <vector>

class ManualHello : public ITest<HelloInput, HelloOutput> {
public:
    ManualHello(HelloInput&);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};
