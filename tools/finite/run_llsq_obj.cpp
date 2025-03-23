#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq_obj.hpp"
#include "finite.h"

class Gradient : public Function<llsq_obj::Input, llsq_obj::GradientOutput> {
  FiniteDifferencesEngine<double> _engine;
public:
  Gradient(llsq_obj::Input& input) : Function(input), _engine(1) {}

  void compute(llsq_obj::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    _engine.finite_differences(1, [&](double *in, double *out) {
      llsq_obj::primal<double>(n, m, in, out);
    }, _input.x.data(), m, 1, output.data());
  }
};


int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq_obj::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
