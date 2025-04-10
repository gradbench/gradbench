#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lse.hpp"
#include "finite.h"

class Gradient : public Function<lse::Input, lse::GradientOutput> {
private:
  FiniteDifferencesEngine<double> _engine;
public:
  Gradient(lse::Input& input) : Function(input), _engine(_input.x.size()) {}

  void compute(lse::GradientOutput& output) {
    size_t n = _input.x.size();
    output.resize(n);

    _engine.finite_differences(1, [&](double* in, double* out) {
      lse::primal<double>(n, in, out);
    }, _input.x.data(), n, 1, output.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"primal", function_main<lse::Primal>},
      {"gradient", function_main<Gradient>},
    });;
}
