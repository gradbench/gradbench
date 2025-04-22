#include "gradbench/evals/det.hpp"
#include "enzyme.h"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<det::Input, det::GradientOutput> {
public:
  Gradient(det::Input& input) : Function(input) {}

  void compute(det::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell * ell);
    std::fill(output.begin(), output.end(), 0);

    double dummy, unit = 1;
    __enzyme_autodiff(det::primal<double>, enzyme_const, ell, enzyme_dup,
                      _input.A.data(), output.data(), enzyme_dupnoneed, &dummy,
                      &unit);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {
                          {"primal", function_main<det::Primal>},
                          {"gradient", function_main<Gradient>},
                      });
  ;
}
