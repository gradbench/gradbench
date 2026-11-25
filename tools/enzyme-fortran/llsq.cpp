#include "enzyme_fortran.h"
#include "gradbench/evals/llsq.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
public:
  Gradient(llsq::Input& input) : Function(input) {}

  void compute(llsq::GradientOutput& output) {
    int n = _input.n, m = _input.x.size();
    output.resize(m);
    std::fill(output.begin(), output.end(), 0.0);
    llsq_enzyme_gradient(&n, &m, _input.x.data(), output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<llsq::Primal>},
                       {"gradient", function_main<Gradient>}});
}
