#include "EnzymeHello.h"
#include "adbench/shared/hello.h"

void EnzymeHello::prepare(HelloInput&& input)
{
    _input = input;
}

HelloOutput EnzymeHello::output()
{
    return _output;
}

void EnzymeHello::calculate_objective(int times)
{
    for (int i = 0; i < times; ++i) {
      _output.objective = hello_objective(_input.x);
    }
}

extern double __enzyme_autodiff(void*, double);

void EnzymeHello::calculate_jacobian(int times)
{
    for (int i = 0; i < times; ++i) {
      _output.gradient = __enzyme_autodiff((void*)hello_objective, _input.x);
    }
}

extern "C" DLL_PUBLIC ITest<HelloInput, HelloOutput>* get_ba_test()
{
    return new EnzymeHello();
}
