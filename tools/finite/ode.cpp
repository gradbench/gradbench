#include "gradbench/evals/ode.hpp"
#include "finite.hpp"
#include "gradbench/main.hpp"
#include <algorithm>

class Gradient : public Function<ode::Input, ode::GradientOutput> {
  FiniteDifferencesEngine<double> _engine;

public:
  Gradient(ode::Input& input) : Function(input) {
    _engine.set_max_output_size(1);
  }

  void compute(ode::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    _engine.finite_differences(
        1,
        [&](double* in, double* out) {
          std::vector<double> tmp(n);
          ode::primal(n, in, _input.s, tmp.data());
          *out = tmp[n - 1];
        },
        _input.x.data(), n, 1, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"primal", function_main<ode::Primal>},
                       {"gradient", function_main<Gradient>}});
  ;
}
