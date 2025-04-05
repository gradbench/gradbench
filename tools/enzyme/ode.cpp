#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/ode.hpp"
#include "enzyme.h"

class Gradient : public Function<ode::Input, ode::GradientOutput> {
public:
  Gradient(ode::Input& input) : Function(input) {}

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);
    std::fill(output.begin(), output.end(), 0);

    std::vector<double> dummy(n), adj(n);
    adj[n-1] = 1;
    __enzyme_autodiff(ode::primal<double>,
                      enzyme_const, n,
                      enzyme_dup, _input.x.data(), output.data(),
                      enzyme_const, _input.s,
                      enzyme_dupnoneed, dummy.data(), adj.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<ode::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
