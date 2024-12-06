#pragma once

#include "adbench/shared/ITest.h"
#include "adbench/shared/HelloData.h"
#include "finite.h"

#include <vector>

class FiniteHello : public ITest<HelloInput, HelloOutput> {
private:
    HelloInput _input;
    HelloOutput _output;
    FiniteDifferencesEngine<double> engine;

public:
    // This function must be called before any other function.
    virtual void prepare(HelloInput&& input) override;

    virtual void calculate_objective(int times) override;
    virtual void calculate_jacobian(int times) override;
    virtual HelloOutput output() override;

    ~FiniteHello() = default;
};
