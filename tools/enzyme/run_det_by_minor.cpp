#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/det_by_minor.hpp"
#include "enzyme.h"

class Gradient : public Function<det_by_minor::Input, det_by_minor::GradientOutput> {
public:
  Gradient(det_by_minor::Input& input) : Function(input) {}

  void compute(det_by_minor::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);
    std::fill(output.begin(), output.end(), 0);

    double dummy, unit = 1;
    __enzyme_autodiff(det_by_minor::primal<double>,
                      enzyme_const, ell,
                      enzyme_dup, _input.A.data(), output.data(),
                      enzyme_dupnoneed, &dummy, &unit);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det_by_minor::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
