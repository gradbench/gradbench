#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq_obj.hpp"
#include "enzyme.h"

class Gradient : public Function<llsq_obj::Input, llsq_obj::GradientOutput> {
public:
  Gradient(llsq_obj::Input& input) : Function(input) {}

  void compute(llsq_obj::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);
    std::fill(output.begin(), output.end(), 0);

    double dummy, unit = 1;
    __enzyme_autodiff(llsq_obj::primal<double>,
                      enzyme_const, n,
                      enzyme_const, m,
                      enzyme_dup, _input.x.data(), output.data(),
                      enzyme_dupnoneed, &dummy, &unit);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq_obj::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
