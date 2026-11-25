#include "enzyme_fortran.h"
#include "gradbench/evals/lse.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<lse::Input, lse::GradientOutput> {
public:
  Gradient(lse::Input& input) : Function(input) {}

  void compute(lse::GradientOutput& output) {
    int n = _input.x.size();
    output.resize(n);
    std::fill(output.begin(), output.end(), 0.0);
    lse_enzyme_gradient(&n, _input.x.data(), output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<lse::Primal>},
                       {"gradient", function_main<Gradient>}});
}
