#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"
#include "finite.h"

#include <vector>

class FiniteHello : public ITest<HelloInput, HelloOutput> {
private:
    FiniteDifferencesEngine<double> engine;

public:
    FiniteHello(HelloInput& input);

    virtual void calculate_objective() override;
    virtual void calculate_jacobian() override;
};
