#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"

#include <vector>

class AdeptHello : public ITest<HelloInput, HelloOutput> {
public:
    AdeptHello(HelloInput&);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};
