#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/an_ode.hpp"
#include "finite.h"

class Gradient : public Function<an_ode::Input, an_ode::GradientOutput> {
  FiniteDifferencesEngine<double> _engine;
public:
  Gradient(an_ode::Input& input) : Function(input) {
    _engine.set_max_output_size(1);
  }

  void compute(an_ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    _engine.finite_differences(1, [&](double *in, double *out) {
      std::vector<double> tmp(n);
      an_ode::primal(n, in, _input.s, tmp.data());
      *out = tmp[n-1];
    }, _input.x.data(), n, 1, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<an_ode::Primal>},
      {"gradient", function_main<Gradient>}
    });;
}
