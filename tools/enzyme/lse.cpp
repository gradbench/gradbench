#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lse.hpp"
#include "enzyme.h"

class Gradient : public Function<lse::Input, lse::GradientOutput> {
public:
  Gradient(lse::Input& input) : Function(input) {}

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    std::fill(output.begin(), output.end(), 0);

    double dummy, unit = 1;
    __enzyme_autodiff(lse::primal<double>,
                      enzyme_const, n,
                      enzyme_dup, _input.x.data(), output.data(),
                      enzyme_dupnoneed, &dummy, &unit);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<lse::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
