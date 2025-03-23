#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/det_by_minor.hpp"
#include "finite.h"

class Gradient : public Function<det_by_minor::Input, det_by_minor::GradientOutput> {
  FiniteDifferencesEngine<double> _engine;
public:
  Gradient(det_by_minor::Input& input) : Function(input), _engine(1) {}

  void compute(det_by_minor::GradientOutput& output) {
    size_t ell = _input.ell;
    output.resize(ell*ell);

    _engine.finite_differences(1, [&](double *in, double *out) {
      det_by_minor::primal<double>(ell, in, out);
    }, _input.A.data(), ell*ell, 1, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<det_by_minor::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
