#include "enzyme_fortran.h"
#include "gradbench/evals/det.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<det::Input, det::GradientOutput> {
public:
  Gradient(det::Input& input) : Function(input) {}

  void compute(det::GradientOutput& output) {
    int ell = _input.ell;
    output.resize(ell * ell);
    std::fill(output.begin(), output.end(), 0.0);
    det_enzyme_gradient(&ell, _input.A.data(), output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<det::Primal>},
                       {"gradient", function_main<Gradient>}});
}
