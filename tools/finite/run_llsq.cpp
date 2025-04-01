#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/llsq.hpp"
#include "finite.h"

class Gradient : public Function<llsq::Input, llsq::GradientOutput> {
  FiniteDifferencesEngine<double> _engine;
public:
  Gradient(llsq::Input& input) : Function(input), _engine(1) {}

  void compute(llsq::GradientOutput& output) {
    size_t n = _input.n;
    size_t m = _input.x.size();
    output.resize(m);

    _engine.finite_differences(1, [&](double *in, double *out) {
      llsq::primal<double>(n, m, in, out);
    }, _input.x.data(), m, 1, output.data());
  }
};


int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<llsq::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
