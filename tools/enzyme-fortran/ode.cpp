#include "enzyme_fortran.h"
#include "gradbench/evals/ode.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<ode::Input, ode::GradientOutput> {
public:
  Gradient(ode::Input& input) : Function(input) {}

  void compute(ode::GradientOutput& output) {
    int n = _input.x.size(), s = _input.s;
    output.resize(n);
    std::fill(output.begin(), output.end(), 0.0);
    ode_enzyme_gradient(&n, _input.x.data(), output.data(), &s);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<ode::Primal>},
                       {"gradient", function_main<Gradient>}});
}
