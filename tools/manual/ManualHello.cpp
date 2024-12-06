#include "ManualHello.h"
#include "adbench/shared/hello.h"
#include "hello_d.h"

void ManualHello::prepare(HelloInput&& input)
{
    _input = input;
}

HelloOutput ManualHello::output()
{
    return _output;
}

void ManualHello::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
      _output.objective = hello_objective(_input.x);
    }
}

void ManualHello::calculate_jacobian(int times)
{
    for (int i = 0; i < times; ++i) {
      _output.gradient = hello_objective_d(_input.x);
    }
}

extern "C" DLL_PUBLIC ITest<HelloInput, HelloOutput>* get_ba_test()
{
    return new ManualHello();
}
